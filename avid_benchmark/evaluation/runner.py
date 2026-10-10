"""Evaluation orchestration consumes fixed predictions; no model training here."""
from collections import defaultdict
import csv
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
from . import METRICS


def _serialize(value):
    return value.to_dict() if hasattr(value,'to_dict') else value


def _jsonl(path):
    path=Path(path).resolve()
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def _relative(path,manifest):
    path=Path(path)
    return path if path.is_absolute() else Path(manifest).resolve().parent/path


def _pixel_batch(images):
    from .image import resize_images
    return np.concatenate([resize_images(np.asarray(im,dtype=np.float32).transpose(2,0,1)[None]/255,425) for im in images])


def evaluate(args,config):
    selected=args.metrics
    if args.inputs:
        with np.load(args.inputs,allow_pickle=False) as loaded:
            inputs={k:loaded[k] for k in loaded.files}
        output={}
        for name in selected:
            fn=METRICS[name]
            params=inspect.signature(fn).parameters
            missing=[k for k,v in params.items() if v.default is inspect.Parameter.empty and k not in inputs]
            if missing:
                raise ValueError(f'NPZ input for {name} is missing arrays: {missing}')
            output[name]=_serialize(fn(**{k:inputs[k] for k in params if k in inputs}))
        return {'metrics':output,'input_sha256':_sha256(args.inputs)}
    answer_rows=_jsonl(args.answers) if args.answers else None
    if not args.predictions:
        if answer_rows is None:
            raise ValueError('Supply --predictions, --answers, or --inputs.')
        groups=defaultdict(list)
        for row in answer_rows:
            groups[str(row.get('subject_id','unspecified'))].append(row)
        return {'subjects':{sid:{name:_serialize(METRICS[name](rows)) for name in selected} for sid,rows in groups.items()},'answers_sha256':_sha256(args.answers)}

    records=_jsonl(args.predictions)
    if not records:
        raise ValueError('Prediction manifest is empty.')
    groups=defaultdict(list)
    for row in records:
        groups[(row['subject_id'],row.get('stimulus_type',''),row.get('split',''))].append(row)
    metadata={}
    metadata_path=config.get('paths',{}).get('used_stimuli_tsv')
    if metadata_path and Path(metadata_path).is_file():
        with open(metadata_path,encoding='utf-8') as handle:
            metadata={r['image']:r for r in csv.DictReader(handle,delimiter='\t')}
    from .encoders import EncoderPool
    from PIL import Image
    pool=EncoderPool(config.get('evaluation',{}))
    reports={}; extracted_answers=[]
    image_names={'pixcorr','ssim','inception_2way','clip_i_2way','clip_t_2way','siglip_2way','altclip_2way','clip_t_recall_at_1','clip_t_recall_at_5','tifa'}
    vision_names={'alexnet_2_2way','alexnet_5_2way','efficientnet_distance','swav_distance'}
    image_names |= vision_names
    caption_names={'bleu1','bleu4','rouge_l_f1','token_f1','exact_match'}
    for key,rows in sorted(groups.items()):
        ids=[r['image_name'] for r in rows]
        if len(ids)!=len(set(ids)):
            raise ValueError(f'Duplicate stimulus IDs in participant candidate set {key}.')
        if len({r.get('route') for r in rows})>1:
            raise ValueError('One candidate set must use one fixed reconstruction route.')
        captions=[r.get('reference_caption') or r.get('notes',{}).get('caption_reference') or metadata.get(r['image_name'],{}).get('translated_caption','') for r in rows]
        images=[]; references=[]
        if any(name in image_names or 'accuracy' in name or 'chance' in name for name in selected):
            for row in rows:
                with Image.open(_relative(row['generated_image_path'],args.predictions)) as im:
                    images.append(im.convert('RGB'))
        if any(name in {'pixcorr','ssim','inception_2way','clip_i_2way'} | vision_names for name in selected):
            for row in rows:
                with Image.open(_relative(row['gt_image_path'],args.predictions)) as im:
                    references.append(im.convert('RGB'))
        features={}; scores={}; attributes=None
        for name in selected:
            fn=METRICS[name]
            if name in ('pixcorr','ssim'):
                if 'pixels' not in features:
                    features['pixels']=(_pixel_batch(images),_pixel_batch(references))
                value=fn(*features['pixels'],sample_ids=ids)
            elif name in caption_names:
                value=fn([r.get('generated_text') or '' for r in rows],captions,sample_ids=ids)
            elif name=='tifa':
                if not all(captions):
                    raise ValueError('Semantic evaluation requires every reference caption, even for image-only baselines.')
                value=fn(pool.tifa(images,captions),sample_ids=ids)
            elif name in image_names:
                if name in vision_names:
                    encoder='alexnet' if name.startswith('alexnet') else 'efficientnet' if name.startswith('efficientnet') else 'swav'
                    if encoder not in features:
                        features[encoder]=(pool.vision(encoder,images),pool.vision(encoder,references))
                    node='alex2' if name=='alexnet_2_2way' else 'alex5' if name=='alexnet_5_2way' else 'features'
                    pair=tuple(values[node] for values in features[encoder])
                elif name=='inception_2way':
                    if 'inception' not in features:
                        features['inception']=(pool.inception(images),pool.inception(references))
                    pair=features['inception']
                elif name=='clip_i_2way':
                    if 'clip_i' not in features:
                        features['clip_i']=(pool.images('clip',images,True),pool.images('clip',references,True))
                    pair=features['clip_i']
                else:
                    if not all(captions):
                        raise ValueError('Missing reference captions for semantic metrics.')
                    encoder='siglip' if name.startswith('siglip') else 'altclip' if name.startswith('altclip') else 'clip'
                    if encoder not in features:
                        features[encoder]=(pool.images(encoder,images),pool.texts(encoder,captions))
                    pair=features[encoder]
                value=fn(*pair,sample_ids=ids)
            elif 'accuracy' in name or 'chance' in name:
                if attributes is None:
                    if answer_rows is not None:
                        attributes=[r for r in answer_rows if r['image_name'] in ids and int(r.get('subject_id',key[0]))==key[0]]
                    else:
                        from .attributes import extract_attributes
                        from ._backends.qwen_attribute_metrics import Qwen3VLQuestionBackend
                        ecfg=config['evaluation']
                        bank=json.loads(Path(ecfg['attribute_bank']).read_text(encoding='utf-8'))
                        if 'qwen' not in pool.models:
                            pool.models['qwen']=Qwen3VLQuestionBackend(pool.path('qwen'))
                        backend=pool.models['qwen']
                        attributes=extract_attributes(dict(zip(ids,images)),bank,backend)
                    expected={(sid,attribute) for sid in ids for attribute in ('color','position','visual_identity')}
                    actual={(r['image_name'],r['attribute']) for r in attributes}
                    if actual!=expected or len(attributes)!=len(expected):
                        raise ValueError('Attribute answers must cover each requested stimulus/attribute exactly once.')
                    extracted_answers.extend({**row,'subject_id':key[0],'stimulus_type':key[1],'split':key[2]} for row in attributes)
                value=fn(attributes)
            else:
                raise ValueError(f'{name} consumes numerical inputs; use --inputs NPZ or call its function directly.')
            scores[name]=_serialize(value)
        reports[f'sub-{key[0]:02d}/{key[1]}/{key[2]}']={'n':len(ids),'sample_ids':ids,'metrics':scores}
    report={'protocol':'avid-release-v1','prediction_manifest_sha256':_sha256(args.predictions),'subjects':reports,'evaluation_config':config.get('evaluation',{})}
    if getattr(args,'save_answers',None):
        if not extracted_answers:
            raise ValueError('--save-answers requires an attribute metric.')
        path=Path(args.save_answers)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in extracted_answers),encoding='utf-8')
        report['answers_sha256']=_sha256(path)
    elif args.answers:
        report['answers_sha256']=_sha256(args.answers)
    return report


def _sha256(path):
    digest=hashlib.sha256()
    with open(path,'rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()
