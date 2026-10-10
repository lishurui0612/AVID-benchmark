"""Pixel metrics use the paper's 425px bilinear resize and SSIM settings."""
import numpy as np
from .common import array, paired_arrays, result
from .semantic import compute_clip_i_2way, compute_inception_2way


def _images(predictions, references):
    p, r = array(predictions), array(references)
    if p.ndim != 4 or r.ndim != 4 or p.shape[1] != 3 or r.shape[1]!=3 or len(p)!=len(r) or not len(p):
        raise ValueError('Images must have shape [N,3,H,W], RGB float in [0,1].')
    if min(p.min(), r.min()) < 0 or max(p.max(), r.max()) > 1:
        raise ValueError('Explicit RGB float [0,1] required; value ranges are never guessed.')
    return tuple(resize_images(x,425) for x in (p,r))


def resize_images(images,size):
    """Bilinear half-pixel coordinates, clamped edges: align_corners=False."""
    from scipy.ndimage import map_coordinates
    images=np.asarray(images,dtype=np.float32)
    if images.shape[-2:]==(size,size):
        return images
    height,width=images.shape[-2:]
    y=(np.arange(size,dtype=np.float64)+.5)*height/size-.5
    x=(np.arange(size,dtype=np.float64)+.5)*width/size-.5
    grid=np.meshgrid(y,x,indexing='ij')
    return np.stack([np.stack([map_coordinates(channel,grid,order=1,mode='nearest',prefilter=False) for channel in image]) for image in images])


def compute_pixcorr(predictions, references, sample_ids=None):
    p, r = _images(predictions, references)
    p, r = p.reshape(len(p),-1), r.reshape(len(r),-1)
    p, r = p-p.mean(1,keepdims=True), r-r.mean(1,keepdims=True)
    # Historical epsilon convention returns zero for constant images.
    scores = (p*r).sum(1) / np.maximum(np.sqrt((p*p).sum(1)*(r*r).sum(1)), 1e-8)
    return result('pixcorr', scores, sample_ids, 'avid-pixel-425-v1')


def compute_ssim(predictions, references, sample_ids=None):
    from skimage.color import rgb2gray
    from skimage.metrics import structural_similarity
    p, r = _images(predictions, references)
    scores = [structural_similarity(rgb2gray(a.transpose(1,2,0)), rgb2gray(b.transpose(1,2,0)), gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.) for a,b in zip(p,r)]
    return result('ssim', scores, sample_ids, 'avid-ssim-gray-425-v1')


def _distance(name, predictions, references, sample_ids):
    p,r = paired_arrays(predictions,references)
    p,r = p-p.mean(1,keepdims=True),r-r.mean(1,keepdims=True)
    denom = np.linalg.norm(p,axis=1)*np.linalg.norm(r,axis=1)
    if np.any(denom == 0):
        raise ValueError('Constant feature rows cannot be scored.')
    return result(name, 1-(p*r).sum(1)/denom, sample_ids, 'avid-pearson-distance-v1')


def compute_alexnet_2way(predictions, references, sample_ids=None):
    from .semantic import _two_way
    return _two_way('alexnet_2way',predictions,references,sample_ids,'pearson')


def compute_alexnet_2_2way(predictions, references, sample_ids=None):
    """AlexNet features.4 (layer 2), with the fixed participant candidate set."""
    from .semantic import _two_way
    return _two_way('alexnet_2_2way',predictions,references,sample_ids,'pearson')


def compute_alexnet_5_2way(predictions, references, sample_ids=None):
    """AlexNet features.11 (layer 5), with the fixed participant candidate set."""
    from .semantic import _two_way
    return _two_way('alexnet_5_2way',predictions,references,sample_ids,'pearson')


def compute_efficientnet_distance(predictions, references, sample_ids=None):
    return _distance('efficientnet_distance',predictions,references,sample_ids)


def compute_swav_distance(predictions, references, sample_ids=None):
    return _distance('swav_distance',predictions,references,sample_ids)
