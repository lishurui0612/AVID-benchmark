import json
import pytest
from avid_benchmark.cli import model_parser, model_config


def config_file(tmp_path):
    path=tmp_path/'paths.json'
    path.write_text(json.dumps({'paths':{name:str(tmp_path/name) for name in ('dataset_root','coco_image_root','simple_image_root','used_stimuli_tsv','train_output_root')}}))
    return path


@pytest.mark.parametrize('model', ['mindeye2','mindbridge'])
def test_simple_recipe_uses_released_seven_subjects(tmp_path,model):
    args=model_parser('train').parse_args(['--model',model,'--stimulus','simple','--config',str(config_file(tmp_path))])
    cfg=model_config(args)
    assert cfg['train']['subjects']==[1,4,5,6,7,8,10]
    assert cfg['train']['prediction_subjects']==[1]


def test_simple_loo_rejects_absent_held_out_subject(tmp_path):
    args=model_parser('train').parse_args(['--model','mindeye2','--subject','2','--stimulus','simple','--config',str(config_file(tmp_path))])
    with pytest.raises(ValueError,match='participant set'):
        model_config(args)


def test_predict_cap_rejects_negative_value(tmp_path):
    args=model_parser('predict').parse_args(['--model','ldm','--max-samples','-1','--config',str(config_file(tmp_path))])
    with pytest.raises(ValueError,match='nonnegative'):
        model_config(args)
