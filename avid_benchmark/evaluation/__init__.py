"""Public, independent metric functions. No baseline or encoder imports here."""
from .common import MetricResult
from .image import *
from .semantic import *
from .attributes import *
from .caption import *

METRICS = {name.removeprefix('compute_'):function for name,function in list(globals().items()) if name.startswith('compute_')}
__all__ = ['MetricResult','METRICS'] + ['compute_'+name for name in METRICS]
