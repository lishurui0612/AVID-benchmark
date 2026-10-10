"""Small scoring types; importing metrics does not load baseline models."""
from dataclasses import asdict, dataclass, field
import numpy as np


@dataclass(frozen=True)
class MetricResult:
    name: str
    score: float
    sample_ids: list[str]
    values: list[float]
    protocol: str
    details: dict = field(default_factory=dict)
    status: str = 'computed'

    @property
    def n(self):
        return len(self.sample_ids)

    def to_dict(self):
        return {**asdict(self), 'n': self.n}


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite inputs are not allowed; repair or report failed samples explicitly.')
    return result


def ids_for(length, sample_ids=None):
    ids = [str(i) for i in range(length)] if sample_ids is None else list(map(str, sample_ids))
    if not length or len(ids) != length or len(set(ids)) != length:
        raise ValueError('Expected a nonempty set of unique sample IDs, one per scoring unit.')
    return ids


def result(name, values, sample_ids=None, protocol='avid-v1', **details):
    values = array(values).reshape(-1)
    return MetricResult(name, float(values.mean()), ids_for(len(values), sample_ids), values.tolist(), protocol, details)


def paired_arrays(predictions, references):
    p, r = array(predictions), array(references)
    if p.shape != r.shape or not len(p):
        raise ValueError(f'Nonempty paired arrays must have identical shapes: {p.shape}, {r.shape}')
    return p, r
