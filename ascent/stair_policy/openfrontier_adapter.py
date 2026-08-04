"""Thin runtime adapter for an external official OpenFrontier checkout.

OpenFrontier is intentionally not vendored into Ascent.  Set
``OPENFRONTIER_ROOT`` to an official checkout and ``OPENFRONTIER_WEIGHTS`` to
the FrontierNet RGB-D checkpoint.  The adapter imports the upstream detector
lazily and converts its Frontier objects into the small local representation.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
from typing import Optional

import numpy as np

from .semantic_frontier_policy import (
    FrontierDetection,
    StairFrontierCandidate,
)


LOGGER = logging.getLogger(__name__)


# FrontierNet back-projects depth pixels in the OpenCV camera convention
# [right, down, forward].  ASCENT's camera transforms use the local convention
# [forward, left, up] (the same convention as vlfm.get_point_cloud).  Therefore
# an OpenCV camera point first has to be re-based before camera_to_world is
# applied:
#
#   [x_ascent, y_ascent, z_ascent] = [z_cv, -x_cv, -y_cv]
_OPENCV_TO_ASCENT_CAMERA = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)


class OpenFrontierDetectorAdapter:
    def __init__(
        self,
        *,
        camera_intrinsic: np.ndarray,
        device: str = "cuda",
        image_size: int = 320,
    ) -> None:
        self.camera_intrinsic = np.asarray(camera_intrinsic, dtype=np.float32)
        self.device = "cuda" if str(device).startswith("cuda") else "cpu"
        self.image_size = int(image_size)
        self._detector = None
        self.available = False
        self.unavailable_reason = "not initialized"
        self._initialize()

    def _initialize(self) -> None:
        root_value = os.getenv("OPENFRONTIER_ROOT", "third_party/OpenFrontier")
        root = Path(root_value).expanduser().resolve()
        if not (root / "frontier" / "detector.py").is_file():
            self.unavailable_reason = (
                f"OpenFrontier checkout not found at {root}; set OPENFRONTIER_ROOT"
            )
            return

        weights = self._find_weights(root)
        if weights is None:
            self.unavailable_reason = (
                "FrontierNet weights not found; set OPENFRONTIER_WEIGHTS to rgbd_11cls.pth"
            )
            return

        try:
            root_str = str(root)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
            from frontier.detector import FrontierDetector
            from frontier.model.predict import load_model

            model = load_model(str(weights), num_classes=11, use_depth=True)
            self._detector = FrontierDetector(
                model=model,
                camera_intrinsic=self.camera_intrinsic.copy(),
                use_depth=True,
                img_size_model=(self.image_size, self.image_size),
                device=self.device,
            )
        except Exception as exc:
            self.unavailable_reason = f"failed to initialize FrontierNet: {exc}"
            LOGGER.warning(self.unavailable_reason)
            return

        self.available = True
        self.unavailable_reason = ""
        LOGGER.info("Semantic stair FrontierNet loaded from %s", weights)

    @staticmethod
    def _find_weights(root: Path) -> Optional[Path]:
        explicit = os.getenv("OPENFRONTIER_WEIGHTS")
        candidates = [
            Path(explicit).expanduser() if explicit else None,
            root / "model_weights" / "rgbd_11cls.pth",
            Path("pretrained_weights/frontiernet_rgbd_11cls.pth"),
        ]
        for candidate in candidates:
            if candidate is not None and candidate.is_file():
                return candidate.resolve()
        return None

    def detect(
        self,
        rgb: np.ndarray,
        metric_depth: np.ndarray,
        camera_to_world: np.ndarray,
        stair_mask: Optional[np.ndarray],
    ) -> FrontierDetection:
        if not self.available or self._detector is None:
            return FrontierDetection(candidates=[], marked_rgb=None)

        self._detector.detect(
            rgb=rgb,
            depth=metric_depth,
            df_normalizer=float(os.getenv("ASCENT_FRONTIERNET_DF_NORMALIZER", "10")),
            df_thr=float(os.getenv("ASCENT_FRONTIERNET_DF_THRESHOLD", "0.3")),
        )
        # ``anchor_fts`` expects world -> OpenCV-camera.  Inverting ASCENT's
        # camera transform directly would incorrectly interpret [right, down,
        # forward] as [forward, left, up] and move map goals sideways.
        opencv_camera_to_world = self.opencv_camera_to_world(camera_to_world)
        world_to_camera = np.linalg.inv(opencv_camera_to_world)
        upstream = self._detector.anchor_fts(
            depth=metric_depth,
            extrinsic=world_to_camera,
        )
        if not upstream:
            return FrontierDetection(candidates=[], marked_rgb=None)

        candidates = []
        for index, frontier in enumerate(upstream):
            frontier.label = chr(65 + index) if index < 26 else str(index)
            pixel_xy = self._processed_pixel(frontier.pixel_pos)
            support = self._image_stair_support(stair_mask, pixel_xy)
            gain = frontier.u_gain if frontier.u_gain is not None else frontier.gain
            candidates.append(
                StairFrontierCandidate(
                    label=frontier.label,
                    pixel_xy=pixel_xy,
                    goal_xy=np.asarray(frontier.pos3d, dtype=float)[:2],
                    information_gain=float(gain if gain is not None else 1e-4),
                    image_stair_support=support,
                    goal_xyz=np.asarray(frontier.pos3d, dtype=float)[:3],
                )
            )

        marked_rgb, _, _ = self._detector.get_SoM_img(
            ft_list=upstream,
            radius=int(os.getenv("ASCENT_STAIR_SOM_RADIUS", "20")),
            alpha=0.5,
        )
        return FrontierDetection(candidates=candidates, marked_rgb=np.asarray(marked_rgb))

    @staticmethod
    def opencv_camera_to_world(camera_to_world: np.ndarray) -> np.ndarray:
        """Convert an ASCENT camera pose into an OpenCV-camera pose."""

        camera_to_world = np.asarray(camera_to_world, dtype=float)
        if camera_to_world.shape != (4, 4):
            raise ValueError(
                f"camera_to_world must have shape (4, 4), got {camera_to_world.shape}"
            )
        return camera_to_world @ _OPENCV_TO_ASCENT_CAMERA

    def _processed_pixel(self, normalized_pixel) -> tuple[int, int]:
        u = int(round(float(normalized_pixel[0]) * self.image_size))
        v = int(round(float(normalized_pixel[1]) * self.image_size))
        return (
            int(np.clip(u, 0, self.image_size - 1)),
            int(np.clip(v, 0, self.image_size - 1)),
        )

    def _image_stair_support(
        self,
        stair_mask: Optional[np.ndarray],
        pixel_xy: tuple[int, int],
    ) -> float:
        if stair_mask is None or not np.any(stair_mask):
            return 0.0
        processed = self._resize_center_crop(np.asarray(stair_mask, dtype=np.uint8)) > 0
        u, v = pixel_xy
        radius = int(os.getenv("ASCENT_STAIR_MASK_SUPPORT_RADIUS", "20"))
        y0, y1 = max(0, v - radius), min(processed.shape[0], v + radius + 1)
        x0, x1 = max(0, u - radius), min(processed.shape[1], u + radius + 1)
        return float(np.mean(processed[y0:y1, x0:x1])) if y0 < y1 and x0 < x1 else 0.0

    def _resize_center_crop(self, image: np.ndarray) -> np.ndarray:
        import cv2

        height, width = image.shape[:2]
        scale = max(self.image_size / height, self.image_size / width) + 0.02
        resized = cv2.resize(
            image,
            (int(round(width * scale)), int(round(height * scale))),
            interpolation=cv2.INTER_NEAREST,
        )
        y0 = max(0, (resized.shape[0] - self.image_size) // 2)
        x0 = max(0, (resized.shape[1] - self.image_size) // 2)
        cropped = resized[y0 : y0 + self.image_size, x0 : x0 + self.image_size]
        if cropped.shape == (self.image_size, self.image_size):
            return cropped
        output = np.zeros((self.image_size, self.image_size), dtype=cropped.dtype)
        output[: cropped.shape[0], : cropped.shape[1]] = cropped
        return output
