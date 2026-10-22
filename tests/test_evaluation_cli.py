import argparse
import json
from pathlib import Path
import numpy as np
import pytest
from avid_benchmark.evaluation import compute_clip_t_recall_at_1
from avid_benchmark.evaluation.runner import evaluate


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
