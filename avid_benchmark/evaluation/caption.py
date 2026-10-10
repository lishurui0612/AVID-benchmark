"""Frozen historical caption protocol, including its add-one BLEU smoothing."""
from collections import Counter
import math
from .common import result


def tokenize(text):
    tokens, buffer = [], []
    for char in str(text).strip():
        if char.isascii() and char.isalnum():
            buffer.append(char.lower())
        else:
            if buffer:
                tokens.append(''.join(buffer)); buffer=[]
            if not char.isspace():
                tokens.append(char.lower() if char.isascii() else char)
    if buffer:
        tokens.append(''.join(buffer))
    return tokens


def _pairs(predictions, references):
    if len(predictions) != len(references) or not len(predictions):
        raise ValueError('Nonempty aligned prediction/reference captions required.')
    if any(r is None or not str(r).strip() for r in references):
        raise ValueError('Every caption needs an explicit nonempty reference.')
    return [(tokenize(p or ''),tokenize(r)) for p,r in zip(predictions,references)]


def _bleu(p,r,n):
    if not p or not r:
        return 0.
    precisions=[]
    for order in range(1,n+1):
        pc=Counter(tuple(p[i:i+order]) for i in range(len(p)-order+1))
        rc=Counter(tuple(r[i:i+order]) for i in range(len(r)-order+1))
        total=sum(pc.values())
        precisions.append((sum((pc & rc).values())+1)/(total+1) if total else 1.)
    return min(1.,math.exp(1-len(r)/len(p))) * math.exp(sum(map(math.log,precisions))/n)


def compute_bleu1(predictions, references, sample_ids=None):
    return result('bleu1',[_bleu(p,r,1) for p,r in _pairs(predictions,references)],sample_ids,'avid-caption-historical-add1-v1')


def compute_bleu4(predictions, references, sample_ids=None):
    return result('bleu4',[_bleu(p,r,4) for p,r in _pairs(predictions,references)],sample_ids,'avid-caption-historical-add1-v1')


def _lcs(a,b):
    row=[0]*(len(b)+1)
    for token in a:
        new=[0]
        for i,other in enumerate(b):
            new.append(row[i]+1 if token==other else max(new[-1],row[i+1]))
        row=new
    return row[-1]


def _f1(overlap,p,r):
    return 2*overlap/(len(p)+len(r)) if p and r else 0.


def compute_rouge_l_f1(predictions, references, sample_ids=None):
    return result('rouge_l_f1',[_f1(_lcs(p,r),p,r) for p,r in _pairs(predictions,references)],sample_ids,'avid-caption-historical-v1')


def compute_token_f1(predictions, references, sample_ids=None):
    return result('token_f1',[_f1(sum((Counter(p)&Counter(r)).values()),p,r) for p,r in _pairs(predictions,references)],sample_ids,'avid-caption-historical-v1')


def compute_exact_match(predictions, references, sample_ids=None):
    _pairs(predictions,references)
    return result('exact_match',[float(' '.join(str(p or '').split())==' '.join(str(r).split())) for p,r in zip(predictions,references)],sample_ids,'avid-caption-case-sensitive-v1')
