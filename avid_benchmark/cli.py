"""Three deliberately separate entry points."""
import argparse
import copy
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = ('ldm','mindeye1','mindeye2','mindbridge')


def read_config(path):
    path=Path(path).expanduser().resolve()
    if path.suffix in ('.yaml','.yml'):
        import yaml
        payload=yaml.safe_load(path.read_text(encoding='utf-8'))
    else:
        payload=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload,dict):
        raise ValueError(f'Config must be a mapping: {path}')
    return payload


def merge(base,override):
    out=copy.deepcopy(base)
    for key,value in override.items():
        out[key]=merge(out[key],value) if isinstance(value,dict) and isinstance(out.get(key),dict) else copy.deepcopy(value)
    return out


def model_parser(stage):
    p=argparse.ArgumentParser(description=f'{stage.capitalize()} one AVID baseline. This command only performs {stage}.')
    p.add_argument('--model',required=True,choices=MODELS)
    p.add_argument('--config',default='config.yaml',help='Paths and existing local assets; relative to your working directory.')
    p.add_argument('--recipe',help='Optional model JSON/YAML recipe; defaults to the bundled recipes/<model>.json.')
    p.add_argument('--subject',type=int,default=1,help='Prediction subject; held-out subject for MindEye2 LOO.')
    p.add_argument('--stimulus',choices=('complex','simple'),default='complex')
    p.add_argument('--run-name')
    if stage=='train':
        p.add_argument('--epochs',type=int,help='Override training epochs for smoke runs.')
        p.add_argument('--component',choices=('highlevel','lowlevel'),default='highlevel',
                       help='MindEye1: train the high-level or low-level fMRI model independently.')
    else:
        p.add_argument('--checkpoint',help='Existing checkpoint. Required unless train_result declares one.')
        p.add_argument('--max-samples',type=int,default=0,help='0 evaluates the full subject set; small values are smoke only.')
    return p


def model_config(args):
    from .config import parse_path_config
    common=read_config(args.config)
    from .assets import configure_assets
    configure_assets(common.get('assets',{}))
    os.environ['PYTHONPATH']=str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH','')
    component=getattr(args,'component','highlevel')
    if component=='lowlevel' and args.model!='mindeye1':
        raise ValueError('--component lowlevel is supported only for MindEye1.')
    recipe_name='mindeye1_lowlevel' if component=='lowlevel' else args.model
    cfg=read_config(args.recipe or Path(__file__).parent/'recipes'/f'{recipe_name}.json')
    cfg=merge(cfg,common.get('model_overrides',{}).get(args.model,{}))
    cfg['model_name']='ldm_himalaya' if args.model=='ldm' else args.model
    cfg['paths']=parse_path_config(common['paths'],base_dir=Path(args.config).resolve().parent,repo_root=ROOT)
    train=cfg.setdefault('train',{})
    train['component']=component
    if train.get('subjects_by_stimulus'):
        train['subjects']=train['subjects_by_stimulus'][args.stimulus]
        if args.subject not in train['subjects']:
            raise ValueError(f'Subject {args.subject} is outside the released {args.stimulus} participant set.')
    if train.get('mode')=='multi_subject_leave_one_out':
        train['held_out_subject']=args.subject
    elif args.model!='mindbridge' or len(train.get('subjects',[]))<=1:
        train['subjects']=[args.subject]
    train['prediction_subjects']=[args.subject]
    train['stimulus_types']=[args.stimulus]
    suffix='_lowlevel' if component=='lowlevel' else ''
    train['run_name']=args.run_name or f'{args.model}_sub{args.subject:02d}_{args.stimulus}{suffix}'
    # No image generation or benchmark evaluation inside training.
    train.update(run_validation_prediction_after_train=False,generate_val_preview_every_epoch=False,
                 training_preview_full_generation=False,preview_every_n_epochs=0,
                 prediction_train_samples=0)
    if args.model=='ldm':
        train['train_validation_num_samples']=0
    if args.model=='mindeye1':
        cfg['model']['n_samples_save']=0
        cfg['model']['plot_umap']=False
    cfg['evaluation']={'run_after_prediction':False}
    if getattr(args,'epochs',None) is not None:
        if args.model=='ldm':
            raise ValueError('LDM uses ridge fitting; omit --epochs for this model.')
        if args.epochs<1:
            raise ValueError('epochs must be positive.')
        if component=='lowlevel':
            cfg['lowlevel']['num_epochs']=args.epochs
        else:
            cfg['model']['num_epochs']=args.epochs
        if args.model=='mindeye2':
            cfg['model']['loo_finetune_epochs']=args.epochs
    if hasattr(args,'max_samples'):
        if args.max_samples < 0:
            raise ValueError('max-samples must be nonnegative; use 0 for the full set.')
        train['max_prediction_samples']=args.max_samples
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    return cfg


def run_root(cfg):
    return cfg['paths'].train_output_root/cfg['model_name']/cfg['train']['run_name']


def resolve_checkpoint(cfg,override=None):
    if override:
        path=Path(override).expanduser().resolve()
    else:
        train_result=run_root(cfg)/'logs/train_result.json'
        if not train_result.exists():
            train_result=run_root(cfg)/'train_result.json'
        if not train_result.exists():
            raise ValueError('Supply --checkpoint, or train this run first.')
        payload=json.loads(train_result.read_text(encoding='utf-8'))
        if cfg['train'].get('mode')=='multi_subject_leave_one_out':
            # Held-out adapted checkpoint, never the source-only pretrain.
            selected=payload.get('finetunes',{}).get('pretrain_best',{}).get('last_checkpoint')
        else:
            selected=payload.get('best_checkpoint') or payload.get('last_checkpoint') or payload.get('checkpoint_path')
        if not selected:
            raise ValueError('Training result does not declare a compatible checkpoint; supply --checkpoint.')
        path=Path(selected).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _finish_distributed():
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def train_main():
    args=model_parser('train').parse_args()
    cfg=model_config(args)
    if (run_root(cfg)/'logs/train_result.json').exists():
        raise ValueError('This run already has completed training; use a new --run-name to preserve its checkpoints.')
    from .models import create_model_adapter
    from .runtime import save_json
    try:
        output=create_model_adapter(cfg['model_name'],cfg).train()
        if int(os.environ.get('RANK','0'))==0:
            checkpoint_paths=[v for k,v in output.items() if k.endswith('checkpoint') and isinstance(v,str)]
            checkpoint_paths += [v['last_checkpoint'] for v in output.get('finetunes',{}).values()]
            if not checkpoint_paths or any(not Path(p).is_file() for p in checkpoint_paths):
                raise RuntimeError('Training returned without the declared checkpoints.')
            save_json(run_root(cfg)/'logs/train_result.json',output)
            print(json.dumps(output,ensure_ascii=False,indent=2))
    finally:
        _finish_distributed()


def predict_main():
    args=model_parser('predict').parse_args()
    cfg=model_config(args)
    checkpoint=resolve_checkpoint(cfg,args.checkpoint)
    from .models import create_model_adapter
    try:
        output=create_model_adapter(cfg['model_name'],cfg).predict_batch(str(checkpoint))
        if int(os.environ.get('RANK','0'))==0:
            print(json.dumps(output,ensure_ascii=False,indent=2))
    finally:
        _finish_distributed()


def evaluate_main():
    from .evaluation import METRICS
    p=argparse.ArgumentParser(description='Compute selected metrics only; never train or generate predictions.')
    p.add_argument('--predictions',help='JSONL prediction manifest, with one record per stimulus.')
    p.add_argument('--answers',help='Frozen Qwen answer JSONL; attribute scoring only.')
    p.add_argument('--save-answers',help='Save extracted Qwen answers for later scoring without loading Qwen.')
    p.add_argument('--inputs',help='NPZ arrays for precomputed embeddings or image/caption scoring.')
    p.add_argument('--metrics',nargs='+',required=True,choices=sorted(METRICS))
    p.add_argument('--config',help='Optional encoder/path config; defaults to config.yaml when present.')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    from .evaluation.runner import evaluate
    config_path=Path(args.config or 'config.yaml').expanduser().resolve()
    config=read_config(config_path) if args.config or config_path.is_file() else {}
    # Question banks are relative to the config, independent of the caller's cwd.
    for key in ('attribute_bank','tifa_bank'):
        bank=config.get('evaluation',{}).get(key)
        if bank:
            config['evaluation'][key]=str((config_path.parent/Path(bank).expanduser()).resolve())
    report=evaluate(args,config)
    from .runtime import save_json
    save_json(Path(args.output),report)
    print(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
