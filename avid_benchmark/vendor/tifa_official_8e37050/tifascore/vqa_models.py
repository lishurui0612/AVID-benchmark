"""mPLUG wrapper derived from official TIFA (Apache-2.0)."""

from __future__ import annotations

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


class MPLUG:
    def __init__(self, ckpt="damo/mplug_visual-question-answering_coco_large_en"):
        self.pipeline_vqa = pipeline(Tasks.visual_question_answering, model=ckpt)

    def vqa(self, image, question):
        input_vqa = {"image": image, "question": question}
        result = self.pipeline_vqa(input_vqa)
        return result["text"]

