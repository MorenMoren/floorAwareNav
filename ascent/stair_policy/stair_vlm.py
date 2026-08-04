"""Set-of-Marks VLM scoring for upward/downward stair affordances."""

from __future__ import annotations

import json
import re
from typing import Dict, Sequence, Tuple

import numpy as np

from .semantic_frontier_policy import StairDirection


def build_stair_prompt(labels: Sequence[str], direction: StairDirection) -> str:
    if direction == StairDirection.UP:
        motion = "UPWARD along the same staircase toward the next upper landing"
        reject = "descending stairs, side corridors, the current landing"
    else:
        motion = "DOWNWARD along the same staircase toward the next lower landing"
        reject = "ascending stairs, side corridors, the current landing"

    label_text = ", ".join(labels)
    return (
        f"The labeled points {label_text} are reachable visual navigation frontiers. "
        "The robot is already entering or traversing a staircase. "
        f"Its task is to continue safely {motion}. "
        "For every label, estimate the probability from 0 to 1 that moving toward it "
        "follows the requested stair direction, remains on a traversable stair or landing "
        "surface, and makes progress toward the requested floor. "
        "Judge a navigable continuation, not merely whether staircase pixels are visible. "
        f"Do not select railings, walls, obstacles, {reject}, or a visually salient stair "
        "that is not the traversable continuation. "
        "Return only one JSON object. Each label maps to [probability, short reason], for "
        f'example {json.dumps({label: [0.5, "reason"] for label in labels})}.'
    )


def parse_frontier_scores(raw_response: str, labels: Sequence[str]) -> Dict[str, Tuple[float, str]]:
    text = raw_response.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    data = json.loads(text)
    if isinstance(data, list):
        data = data[0] if data else {}
    if not isinstance(data, dict):
        raise ValueError("VLM response must be a JSON object")

    result: Dict[str, Tuple[float, str]] = {}
    for label in labels:
        value = data.get(label)
        if not isinstance(value, (list, tuple)) or not value:
            continue
        try:
            probability = float(value[0])
        except (TypeError, ValueError):
            continue
        reason = str(value[1]) if len(value) > 1 else ""
        result[label] = (float(np.clip(probability, 0.0, 1.0)), reason)
    return result


class QwenStairVLMScorer:
    """Use the same multimodal Qwen instance as Ascent_LLM_Planner."""

    def __init__(self, qwen_vlm) -> None:
        self.qwen_vlm = qwen_vlm
        self.available = callable(getattr(qwen_vlm, "chat_with_image", None))
        self.unavailable_reason = (
            "" if self.available else "shared Ascent Qwen client has no chat_with_image method"
        )

    def score(
        self,
        marked_rgb: np.ndarray,
        labels: Sequence[str],
        direction: StairDirection,
    ) -> Dict[str, Tuple[float, str]]:
        if not self.available:
            return {}
        raw = self.qwen_vlm.chat_with_image(
            build_stair_prompt(labels, direction), marked_rgb
        )
        return parse_frontier_scores(str(raw), labels)


def build_stair_vlm_scorer(shared_qwen_vlm):
    return QwenStairVLMScorer(shared_qwen_vlm)
