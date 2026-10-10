import argparse
import json
from pathlib import Path
import numpy as np
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
