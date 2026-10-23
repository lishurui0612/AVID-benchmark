import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from avid_benchmark.evaluation import encoders
from avid_benchmark.evaluation.runner import evaluate


@pytest.fixture
def native_clip(tmp_path,monkeypatch):
    import torch

    checkpoint=tmp_path/'ViT-L-14.pt'
    checkpoint.write_bytes(b'local test checkpoint')
    monkeypatch.setattr(encoders,'CLIP_L14_SHA256',hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    calls={'load':[],'image':[],'text':[],'tokenize':[]}
    class Model:
        dtype=torch.float32

        def eval(self):
            return self

        def encode_image(self,pixels):
            calls['image'].append(pixels.clone())
            return pixels.mean(dim=(2,3))

        def encode_text(self,tokens):
            calls['text'].append(tokens.clone())
            return tokens[:,:3].float()

    model=Model()
    def load(path,device,jit):
        calls['load'].append((path,device,jit))
        return model,lambda image:pytest.fail('Default CLIP center-crop transform must not be used')
    def tokenize(texts,context_length,truncate):
        calls['tokenize'].append((list(texts),context_length,truncate))
        tokens=torch.zeros((len(texts),context_length),dtype=torch.int)
        for i,text in enumerate(texts):
            tokens[i,:3]=torch.tensor([len(text.split()),2,3])
        return tokens
    monkeypatch.setitem(sys.modules,'clip',SimpleNamespace(load=load,tokenize=tokenize))
    return checkpoint,calls


def test_native_clip_matches_paper_preprocessing_and_tokenization(native_clip):
    import torch
    import torch.nn.functional as F
    from PIL import Image

    checkpoint,calls=native_clip
    pool=encoders.EncoderPool({'clip':str(checkpoint),'device':'cpu','batch_size':1,'clip_batch_size':2})
    images=[Image.fromarray(np.arange(9*18*3,dtype=np.uint8).reshape(9,18,3)) for _ in range(3)]
    image_features=pool.images('clip',images,True)
    text_image_features=pool.images('clip',images)
    np.testing.assert_array_equal(image_features,text_image_features)
    # Independent reference: raw RGB tensor, bilinear224/noAA, CLIP normalization.
    x=torch.tensor(np.array(images[0]),dtype=torch.float32).permute(2,0,1)[None]/255
    x=F.interpolate(x,size=(224,224),mode='bilinear',align_corners=False,antialias=False)
    mean=torch.tensor([.48145466,.4578275,.40821073])[None,:,None,None]
    std=torch.tensor([.26862954,.26130258,.27577711])[None,:,None,None]
    torch.testing.assert_close(calls['image'][0][0],((x-mean)/std)[0],rtol=0,atol=0)
    pool.texts('clip',['a cat','a long caption','dog'])
    assert calls['load']==[(str(checkpoint),'cpu',False)]
    assert [len(x) for x in calls['image']]==[2,1,2,1]
    assert calls['tokenize']==[(['a cat','a long caption'],77,True),(['dog'],77,True)]
    assert pool.protocols['clip']['checkpoint_sha256']==hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert pool.protocols['clip']['implementation']=='openai_clip_native'
    assert pool.protocols['clip']['batch_size']==2


@pytest.mark.parametrize('kind',['wrong_checkpoint','hf_directory'])
def test_clip_rejects_noncanonical_weights_before_loading(tmp_path,monkeypatch,kind):
    path=tmp_path/'clip'
    if kind=='hf_directory':
        path.mkdir()
    else:
        path.write_bytes(b'ViT-B/32 or other noncanonical checkpoint')
    monkeypatch.setitem(sys.modules,'clip',SimpleNamespace(load=lambda *a,**kw:pytest.fail('Rejected weights must not load')))
    pool=encoders.EncoderPool({'clip':str(path),'device':'cpu'})
    with pytest.raises(ValueError,match='SHA-256 mismatch|not a Hugging Face directory'):
        pool.texts('clip',['a cat'])


@pytest.mark.parametrize('text_first',[False,True])
def test_all_clip_metrics_share_generated_features_and_record_protocol(tmp_path,native_clip,text_first):
    from PIL import Image

    checkpoint,calls=native_clip
    rows=[]
    for i,color in enumerate(['red','blue']):
        path=tmp_path/f'{i}.png'
        Image.new('RGB',(11,8),color).save(path)
        rows.append(dict(subject_id=1,stimulus_type='simple',split='val',image_name=f'{i}.jpg',
                         generated_image_path=str(path),gt_image_path=str(path),reference_caption=f'a {color} shape'))
    manifest=tmp_path/'predictions.jsonl'
    manifest.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
    metrics=['clip_i_2way','clip_t_2way','clip_t_recall_at_1','clip_t_recall_at_5']
    if text_first:
        metrics=metrics[1:]+metrics[:1]
    args=argparse.Namespace(inputs=None,answers=None,predictions=str(manifest),metrics=metrics)
    report=evaluate(args,{'evaluation':{'clip':str(checkpoint),'device':'cpu','batch_size':1}})
    scores=report['subjects']['sub-01/simple/val']['metrics']
    assert {name:result['score'] for name,result in scores.items()}=={
        'clip_i_2way':1.0,'clip_t_2way':0.0,'clip_t_recall_at_1':0.5,'clip_t_recall_at_5':1.0,
    }
    assert len(calls['image'])==2  # Generated images once, reference images once.
    assert len(calls['text'])==1
    protocol=report['encoder_protocols']['clip']
    assert protocol['model']=='ViT-L/14'
    assert protocol['batch_size']==160
    assert protocol['text_image_preprocessing']=='raw_tensor_bilinear224_clip_normalize'


@pytest.mark.parametrize('name',['config.yaml','config.mindeye2_full.yaml'])
def test_evaluation_templates_use_paper_clip(name):
    from avid_benchmark.cli import read_config

    config=read_config(Path(__file__).resolve().parents[1]/name)
    assert config['evaluation']['clip'].endswith('/ViT-L-14.pt')
    assert config['evaluation']['clip_batch_size']==160
