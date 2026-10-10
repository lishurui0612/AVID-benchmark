"""Check the real TIFA inference imports without downloading model weights."""
from importlib.metadata import version

from modelscope.pipelines.multi_modal.visual_question_answering_pipeline import (
    VisualQuestionAnsweringPipeline,
)
from avid_benchmark.vendor.tifa_official_8e37050.tifascore import MPLUG, SBERTModel


if __name__ == "__main__":
    assert callable(VisualQuestionAnsweringPipeline)
    assert callable(MPLUG) and callable(SBERTModel)
    for package in ("torch", "torchvision", "transformers", "modelscope", "fairseq"):
        print(f"{package}=={version(package)}")
    print("TIFA inference imports passed (model weights are loaded separately).")
