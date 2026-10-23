import argparse
import json
from pathlib import Path
import numpy as np
import pytest
from avid_benchmark.evaluation import compute_clip_t_recall_at_1
from avid_benchmark.evaluation.runner import evaluate


@pytest.mark.parametrize('backend',['last_position','tiny_siglip'])
def test_siglip_text_features_do_not_depend_on_batch_composition(monkeypatch,backend):
    import torch
    from avid_benchmark.evaluation.encoders import EncoderPool

    lengths=[]
    class Processor:
        def __call__(self,text,padding,truncation,return_tensors,max_length=64):
            # Deterministic token IDs; exercise actual batching without downloads.
            rows=[[2+len(word)%29 for word in caption.split()]+[1] for caption in text]
            length=max_length if padding=='max_length' else max(map(len,rows))
            lengths.append(length)
            return {'input_ids':torch.tensor([row[:length]+[0]*max(0,length-len(row)) for row in rows])}

    if backend=='tiny_siglip':
        pytest.importorskip('transformers')
        from transformers import SiglipConfig,SiglipModel
        torch.manual_seed(7)
        config=SiglipConfig(
            text_config=dict(vocab_size=32,hidden_size=16,intermediate_size=32,
                             num_hidden_layers=1,num_attention_heads=2,max_position_embeddings=64),
            vision_config=dict(hidden_size=16,intermediate_size=32,num_hidden_layers=1,
                               num_attention_heads=2,image_size=16,patch_size=8),
        )
        model=SiglipModel(config).eval()
    else:
        class LastPositionModel:
            def get_text_features(self,input_ids):
                # SigLIP pools the last position; its position changes with dynamic padding.
                return torch.stack((input_ids[:,-1].float()+1,
                                    torch.full((len(input_ids),),float(input_ids.shape[1]))),dim=1)
        model=LastPositionModel()

    pool=EncoderPool({'device':'cpu','batch_size':2})
    monkeypatch.setattr(pool,'hf',lambda name:(model,Processor()))
    short='there is a knife'
    long='at the bottom there is a yellow three by three dot array'
    alone=pool.texts('siglip',[short])[0]
    mixed=pool.texts('siglip',[short,long])[0]
    np.testing.assert_allclose(alone,mixed,rtol=1e-5,atol=1e-6)
    all_texts=[short,long,'a cat '*50]
    batched=pool.texts('siglip',all_texts)
    pool.batch_size=1
    single=pool.texts('siglip',all_texts)
    np.testing.assert_allclose(batched,single,rtol=1e-5,atol=1e-6)
    assert set(lengths)=={64}


@pytest.mark.parametrize('name',['altclip'])
def test_other_text_encoders_keep_dynamic_padding(monkeypatch,name):
    import torch
    from avid_benchmark.evaluation.encoders import EncoderPool

    calls=[]
    def processor(**kwargs):
        calls.append(kwargs)
        return {'input_ids':torch.ones((1,4),dtype=torch.long)}
    class Model:
        def get_text_features(self,**inputs):
            return inputs['input_ids'].float()
    pool=EncoderPool({'device':'cpu'})
    monkeypatch.setattr(pool,'hf',lambda encoder:(Model(),processor))
    pool.texts(name,['a cat'])
    assert calls==[dict(text=['a cat'],padding=True,truncation=True,return_tensors='pt')]


def test_cli_npz_equals_direct_function(tmp_path):
    features=np.eye(4)
    path=tmp_path/'features.npz'
    np.savez(path,image_embeddings=features,text_embeddings=features,sample_ids=['a','b','c','d'])
    args=argparse.Namespace(inputs=str(path),answers=None,predictions=None,metrics=['clip_t_recall_at_1'])
    assert evaluate(args,{})['metrics']['clip_t_recall_at_1']==compute_clip_t_recall_at_1(features,features,['a','b','c','d']).to_dict()


def test_caption_manifest_keeps_empty_prediction_and_subject_denominator(tmp_path):
    path=tmp_path/'manifest.jsonl'
    rows=[{'subject_id':sid,'image_name':name,'reference_caption':'a cat','generated_text':text} for sid in [1,2] for name,text in [('x','a cat'),('y','')]]
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    args=argparse.Namespace(inputs=None,answers=None,predictions=str(path),metrics=['token_f1'])
    report=evaluate(args,{})
    assert len(report['subjects'])==2
    for group in report['subjects'].values():
        assert group['n']==2
        assert group['metrics']['token_f1']['score']==.5


def test_tifa_scores_equal_mean_of_image_means():
    from avid_benchmark.evaluation import compute_tifa
    assert compute_tifa([[1,0],[1]]).score==.75


def test_precomputed_cli_runs_without_path_or_encoder_config(tmp_path,monkeypatch):
    from avid_benchmark.cli import evaluate_main
    features=np.eye(4)
    np.savez(tmp_path/'features.npz',image_embeddings=features,text_embeddings=features)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('sys.argv',['avid-evaluate','--inputs','features.npz','--metrics','clip_t_recall_at_1','--output','scores.json'])
    evaluate_main()
    assert json.loads((tmp_path/'scores.json').read_text())['metrics']['clip_t_recall_at_1']['score']==1


def test_answers_cli_runs_without_encoder_config(tmp_path,monkeypatch):
    from avid_benchmark.cli import evaluate_main
    row={'subject_id':1,'attribute':'color','image_name':'red.jpg','answer':'red','choices':['red','blue'],'mapped_answer':'red'}
    (tmp_path/'answers.jsonl').write_text(json.dumps(row)+'\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('sys.argv',['avid-evaluate','--answers','answers.jsonl','--metrics','color_accuracy','--output','scores.json'])
    evaluate_main()
    assert json.loads((tmp_path/'scores.json').read_text())['subjects']['1']['color_accuracy']['score']==1


@pytest.mark.parametrize('template,bank_name,key,custom_path',[
    ('config.yaml','simple_attribute_bank_qwen_v3.json','attribute_bank',None),
    ('config.yaml','tifa_bank_complex.json','tifa_bank',None),
    ('config.mindeye2_full.yaml','tifa_bank_complex.json','tifa_bank',None),
    ('config.yaml','simple_attribute_bank_qwen_v3.json','attribute_bank','relative'),
    ('config.yaml','simple_attribute_bank_qwen_v3.json','attribute_bank','absolute'),
    ('config.yaml','tifa_bank_simple.json','tifa_bank','relative'),
    ('config.yaml','tifa_bank_simple.json','tifa_bank','absolute'),
])
def test_question_bank_paths_through_cli(tmp_path,monkeypatch,template,bank_name,key,custom_path):
    """Read real banks through the CLI/runner, replacing only model inference."""
    from PIL import Image
    from avid_benchmark.cli import evaluate_main,read_config
    from avid_benchmark.evaluation.encoders import EncoderPool
    from avid_benchmark.evaluation._backends import qwen_attribute_metrics,simple_attribute_metrics

    root=Path(__file__).resolve().parents[1]
    bank_path=root/'avid_benchmark/vendor/vmi_metrics'/bank_name
    bank=json.loads(bank_path.read_text(encoding='utf-8'))
    entry=bank['entries'][0]
    config_path=root/template
    if custom_path:
        # Distinct IDs/captions ensure an explicit override cannot silently fall back.
        entry['image_name']='custom.jpg'
        entry['caption']='A custom reference caption.'
        bank['entries']=[entry]
        config_dir=tmp_path/'configuration'
        config_dir.mkdir()
        bank_path=config_dir/'custom_bank.json'
        bank_path.write_text(json.dumps(bank),encoding='utf-8')
        config=read_config(config_path)
        config['evaluation'][key]=bank_path.name if custom_path=='relative' else str(bank_path)
        config_path=config_dir/'custom.json'
        config_path.write_text(json.dumps(config),encoding='utf-8')

    answers={qa.get('prompt',qa['question']):qa['answer'] for qa in entry['qas']}
    calls=[]
    class FakeBackend:
        def __init__(self,*paths):
            pass

        def answer(self,image,question,choices):
            calls.append(question)
            answer=answers[question]
            assert answer in choices
            return answer,answer

    monkeypatch.setattr(qwen_attribute_metrics,'Qwen3VLQuestionBackend',FakeBackend)
    monkeypatch.setattr(simple_attribute_metrics,'TifaQuestionBackend',FakeBackend)
    original_path=EncoderPool.path
    def local_path(self,name):
        if name in ('qwen','tifa_mplug','tifa_sbert'):
            return str(tmp_path)
        return original_path(self,name)
    monkeypatch.setattr(EncoderPool,'path',local_path)

    work_dir=tmp_path/'elsewhere'
    work_dir.mkdir()
    Image.new('RGB',(8,8),'red').save(work_dir/'prediction.png')
    is_attribute=key=='attribute_bank'
    stimulus='simple' if is_attribute or 'simple' in bank_name else 'complex'
    row={'subject_id':1,'stimulus_type':stimulus,'split':'val',
         'image_name':entry.get('image_name','complex.jpg'),
         'reference_caption':entry['caption'],'generated_image_path':'prediction.png'}
    (work_dir/'manifest.jsonl').write_text(json.dumps(row)+'\n',encoding='utf-8')
    metrics=['color_accuracy','position_accuracy','identity_accuracy'] if is_attribute else ['tifa']
    monkeypatch.chdir(work_dir)
    monkeypatch.setattr('sys.argv',['avid-evaluate','--config',str(config_path),
                        '--predictions','manifest.jsonl','--metrics',*metrics,'--output','scores.json'])
    evaluate_main()
    report=json.loads((work_dir/'scores.json').read_text(encoding='utf-8'))
    assert Path(report['evaluation_config'][key])==bank_path.resolve()
    scores=report['subjects'][f'sub-01/{stimulus}/val']['metrics']
    assert all(scores[name]['score']==1 for name in metrics)
    assert calls==[qa.get('prompt',qa['question']) for qa in entry['qas']]
