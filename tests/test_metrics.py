import sys
import numpy as np
import pytest
from avid_benchmark.evaluation import (
    compute_clip_i_2way, compute_clip_t_recall_at_1,
    compute_color_balanced_accuracy, compute_color_balanced_chance,
    compute_bleu1, compute_bleu4, compute_token_f1, compute_rouge_l_f1,
)


def test_caption_empty_prediction_counts_and_unicode_preserved():
    assert compute_token_f1(['猫',''],['猫','狗']).score==.5
    assert compute_rouge_l_f1(['猫',''],['猫','狗']).score==.5
    assert compute_bleu1(['a',''],['a','a']).score==.5
    assert compute_bleu4(['a',''],['a','a']).score==.5


def test_historical_bleu_smoothing_is_explicit():
    value=compute_bleu4(['dog'],['cat'])
    assert value.score==pytest.approx(.5**.25)
    assert 'historical-add1' in value.protocol


def test_full_subject_candidates_no_duplicate_stimuli():
    features=np.array([[1.,0.,-1.],[0.,1.,-1.],[-1.,0.,1.]])
    assert compute_clip_i_2way(features,features,['a','b','c']).score==1
    assert compute_clip_t_recall_at_1(features,features,['a','b','c']).score==1
    with pytest.raises(ValueError,match='unique'):
        compute_clip_i_2way(features,features,['a','a','c'])
    with pytest.raises(ValueError,match='Nonfinite'):
        compute_clip_i_2way(features,features*np.nan)


def test_multi_accepted_color_balanced_chance():
    rows=[{'attribute':'color','image_name':str(i),'answer':label,'choices':['red','blue','other'],'acceptable_answers':accepted,'mapped_answer':'red'} for i,(label,accepted) in enumerate([('red',['red']),('red',['red']),('blue',['blue','other'])])]
    assert compute_color_balanced_accuracy(rows).score==.5
    assert compute_color_balanced_chance(rows).score==.5


def test_pixel_metrics_accept_different_source_resolutions():
    from avid_benchmark.evaluation import compute_pixcorr,compute_ssim
    assert compute_pixcorr(np.ones((2,3,8,8)),np.ones((2,3,16,16))).score==0
    assert compute_ssim(np.ones((2,3,8,8)),np.ones((2,3,16,16))).score==pytest.approx(1)


def test_numpy_bilinear_matches_historical_torch_resize():
    torch=pytest.importorskip('torch')
    from avid_benchmark.evaluation.image import resize_images
    values=np.random.default_rng(9).random((2,3,37,51)).astype(np.float32)
    reference=torch.nn.functional.interpolate(torch.from_numpy(values),size=(425,425),mode='bilinear',align_corners=False).numpy()
    np.testing.assert_allclose(resize_images(values,425),reference,rtol=0,atol=5e-6)
