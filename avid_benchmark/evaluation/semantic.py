"""Score complete single-participant candidate sets from reusable embeddings."""
import numpy as np
from .common import array, ids_for, paired_arrays, result


def _similarities(predictions, references, mode):
    p, r = paired_arrays(predictions, references)
    if p.ndim != 2 or len(p) < 2:
        raise ValueError('Identification needs at least two [N,D] feature rows.')
    if mode == 'pearson':
        p, r = p - p.mean(1, keepdims=True), r - r.mean(1, keepdims=True)
    pn, rn = np.linalg.norm(p, axis=1), np.linalg.norm(r, axis=1)
    if np.any(pn == 0) or np.any(rn == 0):
        raise ValueError('Zero-norm/constant embeddings cannot be scored.')
    return (p / pn[:, None]) @ (r / rn[:, None]).T


def _two_way(name, predictions, references, sample_ids=None, mode='cosine'):
    sim = _similarities(predictions, references, mode)
    ids_for(len(sim), sample_ids)
    # Match the original strict comparison: tied distractors count as failures.
    mask = ~np.eye(len(sim), dtype=bool)
    scores = ((sim < np.diag(sim)[:, None]) & mask).sum(1) / (len(sim) - 1)
    return result(name, scores, sample_ids, 'avid-2way-strict-v1', similarity=mode, candidates=len(sim))


def compute_inception_2way(predictions, references, sample_ids=None):
    return _two_way('inception_2way', predictions, references, sample_ids, 'pearson')


def compute_clip_i_2way(predictions, references, sample_ids=None):
    return _two_way('clip_i_2way', predictions, references, sample_ids, 'pearson')


def compute_clip_t_2way(image_embeddings, text_embeddings, sample_ids=None):
    return _two_way('clip_t_2way', image_embeddings, text_embeddings, sample_ids)


def compute_siglip_2way(image_embeddings, text_embeddings, sample_ids=None):
    return _two_way('siglip_2way', image_embeddings, text_embeddings, sample_ids)


def compute_altclip_2way(image_embeddings, text_embeddings, sample_ids=None):
    return _two_way('altclip_2way', image_embeddings, text_embeddings, sample_ids)


def _recall(name, predictions, references, k, sample_ids):
    sim = _similarities(predictions, references, 'cosine')
    # Stable candidate order is explicit for exact ties.
    ranks = np.argsort(-sim, axis=1, kind='stable')[:, :min(k, len(sim))]
    scores = (ranks == np.arange(len(sim))[:, None]).any(1).astype(float)
    return result(name, scores, sample_ids, 'avid-retrieval-stable-v1', requested_k=k, effective_k=min(k, len(sim)), candidates=len(sim))


def compute_clip_t_recall_at_1(image_embeddings, text_embeddings, sample_ids=None):
    return _recall('clip_t_recall_at_1', image_embeddings, text_embeddings, 1, sample_ids)


def compute_clip_t_recall_at_5(image_embeddings, text_embeddings, sample_ids=None):
    return _recall('clip_t_recall_at_5', image_embeddings, text_embeddings, 5, sample_ids)


def compute_tifa(question_scores, sample_ids=None):
    """Each row contains all frozen-question scores for one stimulus, after VQA/SBERT."""
    means = []
    for scores in question_scores:
        scores = array(scores).reshape(-1)
        if not len(scores) or np.any((scores < 0) | (scores > 1)):
            raise ValueError('TIFA needs nonempty question scores in [0,1] for every image.')
        means.append(float(scores.mean()))
    return result('tifa', means, sample_ids, 'avid-tifa-frozen-bank-v1')
