"""Shared OpenAI-compatible client for the multimodal Qwen3.6 service."""

from __future__ import annotations

import base64
import io
import logging
import os
from typing import Sequence

import numpy as np
from PIL import Image


LOGGER = logging.getLogger(__name__)


class Qwen3_6Client:
    """Text and image chat client shared by Ascent planners."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.6-27B",
        base_url: str = "http://172.16.33.7:11435/v1",
    ) -> None:
        from openai import OpenAI

        self.model_name = os.getenv("ASCENT_QWEN_MODEL", model_name)
        self.base_url = os.getenv("ASCENT_QWEN_BASE_URL", base_url)
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=os.getenv("ASCENT_QWEN_API_KEY", "EMPTY"),
        )

    @staticmethod
    def _image_data_url(image: np.ndarray) -> str:
        array = np.asarray(image)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="JPEG", quality=90)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _completion(self, messages, max_tokens: int):
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

    def chat(self, txt: str) -> str:
        try:
            response = self._completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an assistant with advanced spatial reasoning. "
                            "Return only the requested valid JSON and a brief reason."
                        ),
                    },
                    {"role": "user", "content": txt},
                ],
                max_tokens=1024,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            LOGGER.warning("Qwen3.6 text request failed: %s", exc)
            return "-1"

    def chat_with_image(self, txt: str, image: np.ndarray) -> str:
        return self.chat_with_images(txt, [image])

    def chat_with_images(self, txt: str, images: Sequence[np.ndarray]) -> str:
        content = [{"type": "text", "text": txt}]
        content.extend(
            {
                "type": "image_url",
                "image_url": {"url": self._image_data_url(image)},
            }
            for image in images
        )
        try:
            response = self._completion(
                messages=[
                    {"role": "system", "content": "Return only the requested valid JSON."},
                    {"role": "user", "content": content},
                ],
                max_tokens=512,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            LOGGER.warning("Qwen3.6 multimodal request failed: %s", exc)
            return "-1"
