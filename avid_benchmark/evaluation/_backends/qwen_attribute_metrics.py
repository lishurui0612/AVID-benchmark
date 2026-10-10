"""Qwen3-VL backend and strict closed-set prompts for simple attributes."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from avid_benchmark.evaluation._backends.simple_attribute_metrics import (
    ATTRIBUTE_ORDER,
    canonical_json_hash,
    validate_question_bank,
)


QWEN_BANK_VERSION = "qwen_v3"
QWEN_METRIC_NAME = "Simple-Attribute-Qwen3VL"
QWEN_COLOR_CHOICES = ["red", "yellow", "blue", "other or multicolored"]
SEMICIRCLE_LABEL_MAP = {
    "upward-facing semicircle": "semicircle (straight edge at bottom)",
    "downward-facing semicircle": "semicircle (straight edge at top)",
    "left-facing semicircle": "semicircle (straight edge at right)",
    "right-facing semicircle": "semicircle (straight edge at left)",
}

# Manual visible-color audit of the 49 natural-object stimuli. All unlisted
# object images use "other or multicolored". These labels describe only the
# rendered foreground color and do not depend on object identity or family.
OBJECT_PRIMARY_COLOR_LABELS = {
    "simple_356.jpg": "blue",
    "simple_359.jpg": "blue",
    "simple_361.jpg": "blue",
    "simple_366.jpg": "yellow",
    "simple_373.jpg": "yellow",
    "simple_375.jpg": "red",
    "simple_377.jpg": "yellow",
    "simple_378.jpg": "blue",
    "simple_383.jpg": "red",
    "simple_394.jpg": "red",
    "simple_395.jpg": "blue",
}


def _choice_json(choices: Sequence[str]) -> str:
    return json.dumps(list(choices), ensure_ascii=False)


def strict_prompt(attribute: str, choices: Sequence[str]) -> str:
    """Return the frozen prompt for one of the three core attributes."""

    options = _choice_json(choices)
    common_output = f"""Valid options:
{options}

Output requirements:
- Select exactly one option from the list.
- Output only the exact option text.
- Do not output an explanation, label, quotation marks, or punctuation.
- Your entire response must exactly match one valid option."""

    if attribute == "color":
        return f"""Determine the dominant visible color of the main foreground content in the image.

Classification rules:
- Ignore the image background.
- Judge only the foreground's visible color. Do not use its object identity, stimulus family, realism, or meaning.
- Choose "red", "yellow", or "blue" when that color clearly dominates the visible foreground.
- Choose "other or multicolored" when no single one of red, yellow, or blue clearly dominates, or when the dominant color is outside those three colors.
- Treat shading, highlights, outlines, and small details as secondary to the overall visible foreground color.

{common_output}"""

    if attribute == "position":
        return f"""Determine the location of the main foreground content relative to the entire image canvas.

Classification rules:
- Ignore the background.
- Judge the location using the center of the main foreground content as a whole.
- If the content contains multiple connected or repeated elements, use their combined spatial extent.
- Choose the nearest applicable location among the five valid options.

{common_output}"""

    if attribute == "visual_identity":
        return f"""Identify the main foreground content in the image.

Classification rules:
- Select the single option that most specifically identifies the main foreground content.
- Ignore the background and incidental secondary details.
- Determine identity and shape orientation independently of where the foreground is located in the image. Mentally move it to the image center before judging its identity.
- For a letter, identify it regardless of uppercase or lowercase and output its uppercase identity.
- For a digit sequence, preserve the complete digit order exactly.
- For a geometric shape, pay attention to whether it is filled or ring-like.
- Use the following required two-stage procedure for every possible semicircle:
  Stage 1 — identify the shape as a semicircle:
  - First decide whether the foreground shape is a semicircle, before considering any direction.
  - A filled shape bounded by one 180-degree curved arc and one straight diameter edge is a semicircle.
  - If it is a semicircle, the answer must be one of the four options beginning with "semicircle". Do not choose a letter or another shape.
  - In particular, a filled semicircle is not the letter "C". The letter "C" has a concave open gap, whereas a filled semicircle does not.
  Stage 2 — only after identifying a semicircle, locate its straight edge:
  - Ignore where the whole semicircle is positioned on the image canvas.
  - straight edge at the bottom of the semicircle -> "semicircle (straight edge at bottom)"
  - straight edge at the top of the semicircle -> "semicircle (straight edge at top)"
  - straight edge at the right side of the semicircle -> "semicircle (straight edge at right)"
  - straight edge at the left side of the semicircle -> "semicircle (straight edge at left)"
- In the four semicircle options, top, bottom, left, and right refer only to the straight edge within the already-identified semicircle. They never refer to the semicircle's position in the image.
- Classify a rectangle as "vertical rectangle" when its height is greater than its width, and as "horizontal rectangle" when its width is greater than its height. Use "square" when its height and width are equal.

{common_output}"""

    raise ValueError(
        f"Unsupported attribute {attribute!r}; expected one of {ATTRIBUTE_ORDER}"
    )


def build_qwen_question_bank(base_bank: Mapping[str, Any]) -> dict[str, Any]:
    """Convert the split-independent base labels into frozen Qwen v3 prompts."""

    bank = copy.deepcopy(dict(base_bank))
    validate_question_bank(bank)
    bank["metric_name"] = QWEN_METRIC_NAME
    bank["version"] = QWEN_BANK_VERSION
    for entry in bank["entries"]:
        image_stem = Path(str(entry["image_name"])).stem
        for qa in entry["qas"]:
            attribute = str(qa["attribute"])
            if attribute == "color":
                qa["choices"] = list(QWEN_COLOR_CHOICES)
                if entry["family"] == "object":
                    qa["answer"] = OBJECT_PRIMARY_COLOR_LABELS.get(
                        str(entry["image_name"]),
                        "other or multicolored",
                    )
                    qa["acceptable_answers"] = (
                        [qa["answer"], "other or multicolored"]
                        if qa["answer"] != "other or multicolored"
                        else ["other or multicolored"]
                    )
                    qa["subtype"] = "uncontrolled_visible_color"
                elif qa["answer"] == "natural or multicolored":
                    qa["answer"] = "other or multicolored"
                entry["labels"]["color"] = qa["answer"]
            elif attribute == "visual_identity":
                qa["choices"] = [
                    SEMICIRCLE_LABEL_MAP.get(choice, choice)
                    for choice in qa["choices"]
                ]
                qa["answer"] = SEMICIRCLE_LABEL_MAP.get(
                    qa["answer"], qa["answer"]
                )
                entry["labels"]["visual_identity"] = qa["answer"]
            qa["question_id"] = (
                f"{image_stem}.{attribute}.{QWEN_BANK_VERSION}"
            )
            qa["question"] = strict_prompt(attribute, list(qa["choices"]))

    metadata = dict(bank.get("metadata", {}))
    metadata.update(
        {
            "base_bank_content_sha256": base_bank.get("metadata", {}).get(
                "bank_content_sha256"
            ),
            "prompt_policy": "all-options-in-prompt-strict-exact-output",
            "color_policy": "foreground-visible-color-only",
            "color_choices": list(QWEN_COLOR_CHOICES),
            "object_color_annotation": (
                "manual visible-color audit; 11 objects accept a visible "
                "primary color plus other-or-multicolored, and 38 objects "
                "accept only other-or-multicolored"
            ),
            "semicircle_policy": (
                "two-stage-semicircle-first-then-straight-edge-position"
            ),
            "visual_identity_vocabulary_size": 95,
        }
    )
    metadata["bank_content_sha256"] = canonical_json_hash(bank["entries"])
    bank["metadata"] = metadata
    validate_question_bank(bank)
    return bank


class Qwen3VLQuestionBackend:
    """Strict exact-choice backend using a local Qwen3-VL checkpoint."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        max_new_tokens: int = 24,
        attention_implementation: str = "sdpa",
    ):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.model_path = str(Path(model_path).expanduser().resolve())
        if not Path(self.model_path).is_dir():
            raise FileNotFoundError(f"Missing Qwen3-VL model: {self.model_path}")
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen3-VL evaluation requires a visible CUDA GPU")

        self.max_new_tokens = int(max_new_tokens)
        self.attention_implementation = attention_implementation
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=attention_implementation,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    def answer(
        self,
        image: Any,
        question: str,
        choices: Sequence[str],
    ) -> tuple[str, str]:
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        answer = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        # Exact closed-set scoring: no SBERT, fuzzy matching, case correction,
        # quote removal, or punctuation cleanup.
        return answer, answer
