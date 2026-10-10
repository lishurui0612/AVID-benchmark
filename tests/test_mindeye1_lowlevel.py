import pytest
from avid_benchmark.cli import model_parser, model_config
from avid_benchmark.models.lowlevel import validate_response_dimension


def test_lowlevel_recipe_and_epochs_are_independent(tmp_path):
    import json
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'paths': {name: str(tmp_path / name) for name in
        ('dataset_root', 'coco_image_root', 'simple_image_root', 'used_stimuli_tsv', 'train_output_root')}}))
    args = model_parser('train').parse_args(['--model', 'mindeye1', '--component', 'lowlevel',
                                            '--config', str(config), '--epochs', '2'])
    cfg = model_config(args)
    assert cfg['train']['component'] == 'lowlevel'
    assert cfg['lowlevel']['num_epochs'] == 2
    assert cfg['lowlevel']['hidden_dim'] == 4096
    assert cfg['train']['run_name'].endswith('_lowlevel')
    assert cfg['train']['run_validation_prediction_after_train'] is False
    args = model_parser('train').parse_args(['--model', 'ldm', '--component', 'lowlevel', '--config', str(config)])
    with pytest.raises(ValueError, match='only for MindEye1'):
        model_config(args)


def test_input_dimension_cannot_silently_pool_or_truncate():
    import numpy as np
    assert validate_response_dimension(np.zeros((2, 2, 41934)), 41934) == 41934
    with pytest.raises(ValueError, match='expects 41934'):
        validate_response_dimension(np.zeros((2, 2, 40308)), 41934)


def test_chunked_contrastive_loss_matches_full_matrix_and_gradient():
    torch = pytest.importorskip('torch')
    from torch.nn import functional as F
    from avid_benchmark.models.mindeye1.lowlevel_train import lowlevel_loss
    torch.manual_seed(3)
    predicted = torch.randn(2, 4, 8, 8, requires_grad=True)
    target = torch.randn_like(predicted)
    student = torch.randn(2, 3, 5, requires_grad=True)
    teacher, augmented = torch.randn_like(student), torch.randn_like(student)
    student_ref = student.detach().clone().requires_grad_()
    predicted_ref = predicted.detach().clone().requires_grad_()
    actual, _, _ = lowlevel_loss(predicted, target, student, teacher, augmented, chunk_size=2)
    s, t, a = [F.normalize(x.flatten(0, 1), dim=-1) for x in (student_ref, teacher, augmented)]
    contrastive = (-(F.log_softmax(s @ a.T / .075, -1) * F.softmax(t @ a.T / .075, -1)).sum(-1).mean()
                   -(F.log_softmax(a @ s.T / .075, -1) * F.softmax(a @ t.T / .075, -1)).sum(-1).mean()) / 2
    expected = F.l1_loss(predicted_ref, target) / .18215 + .1 * contrastive
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(student.grad, student_ref.grad)
    torch.testing.assert_close(predicted.grad, predicted_ref.grad)


def test_frozen_vae_attention_migration_preserves_all_weights():
    torch = pytest.importorskip('torch')
    diffusers = pytest.importorskip('diffusers')
    from avid_benchmark.models.lowlevel import load_legacy_attention_state
    model = diffusers.AutoencoderKL(block_out_channels=(32,),
                                   down_block_types=('DownEncoderBlock2D',),
                                   up_block_types=('UpDecoderBlock2D',), sample_size=8)
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    old = {}
    for key, value in expected.items():
        legacy = key
        if '.attentions.' in key:
            for before, after in (('.to_q.', '.query.'), ('.to_k.', '.key.'),
                                  ('.to_v.', '.value.'), ('.to_out.0.', '.proj_attn.')):
                legacy = legacy.replace(before, after)
        old[legacy] = value
    migrations = load_legacy_attention_state(model, old)
    assert len(migrations) == 16
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[key])
    conflicting = dict(old)
    conflicting['encoder.mid_block.attentions.0.to_q.weight'] = expected['encoder.mid_block.attentions.0.to_q.weight']
    with pytest.raises(ValueError, match='Conflicting'):
        load_legacy_attention_state(model, conflicting)
