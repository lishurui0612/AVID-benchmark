"""Minimal OFA preprocessor exports for the fixed VQA-only runtime.

ModelScope's upstream package initializer imports every OFA task, including
ASR preprocessors and their optional audio dependencies. Simple-Attribute-VQA
only uses the visual-question-answering preprocessor. Placeholder classes keep
the upstream OfaPreprocessor task map importable without activating unrelated
task stacks.
"""


class _UnavailableOFAEvaluationTask:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "This fixed Simple-Attribute-VQA runtime only supports the "
            "OFA visual-question-answering preprocessor."
        )


OfaASRPreprocessor = _UnavailableOFAEvaluationTask
OfaImageCaptioningPreprocessor = _UnavailableOFAEvaluationTask
OfaImageClassificationPreprocessor = _UnavailableOFAEvaluationTask
OfaOcrRecognitionPreprocessor = _UnavailableOFAEvaluationTask
OfaSudokuPreprocessor = _UnavailableOFAEvaluationTask
OfaSummarizationPreprocessor = _UnavailableOFAEvaluationTask
OfaTextToSqlPreprocessor = _UnavailableOFAEvaluationTask
OfaTextClassificationPreprocessor = _UnavailableOFAEvaluationTask
OfaTextToImageSynthesisPreprocessor = _UnavailableOFAEvaluationTask
OfaVisualEntailmentPreprocessor = _UnavailableOFAEvaluationTask
OfaVisualGroundingPreprocessor = _UnavailableOFAEvaluationTask

from .visual_question_answering import OfaVisualQuestionAnsweringPreprocessor

__all__ = [
    "OfaASRPreprocessor",
    "OfaImageCaptioningPreprocessor",
    "OfaImageClassificationPreprocessor",
    "OfaOcrRecognitionPreprocessor",
    "OfaSudokuPreprocessor",
    "OfaSummarizationPreprocessor",
    "OfaTextToSqlPreprocessor",
    "OfaTextClassificationPreprocessor",
    "OfaTextToImageSynthesisPreprocessor",
    "OfaVisualEntailmentPreprocessor",
    "OfaVisualGroundingPreprocessor",
    "OfaVisualQuestionAnsweringPreprocessor",
]
