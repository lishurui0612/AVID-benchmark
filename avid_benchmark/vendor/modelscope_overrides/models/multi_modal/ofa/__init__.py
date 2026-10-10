"""Minimal OFA model exports for the fixed VQA-only runtime."""


class MMSpeechModel:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "This fixed Simple-Attribute-VQA runtime does not include the "
            "optional MMSpeech/fairseq stack."
        )


from .modeling_ofa import OFADecoder, OFAEncoder, OFAModel, OFAPreTrainedModel
from .tokenization_ofa import OFATokenizer, OFATokenizerZH
from .tokenization_ofa_fast import OFATokenizerFast, OFATokenizerZHFast

__all__ = [
    "MMSpeechModel",
    "OFADecoder",
    "OFAEncoder",
    "OFAModel",
    "OFAPreTrainedModel",
    "OFATokenizer",
    "OFATokenizerZH",
    "OFATokenizerFast",
    "OFATokenizerZHFast",
]
