"""Spatially robust progress tracking for ordinary exploration frontiers."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrontierProgressConfig:
    """Thresholds for declaring an exploration frontier unreachable."""

    association_radius: float = 0.6
    blocked_radius: float = 0.75
    min_progress: float = 0.15
    max_stalled_selections: int = 20

    @classmethod
    def from_env(cls) -> "FrontierProgressConfig":
        def number(name: str, default, cast):
            raw = os.getenv(name)
            if raw is None:
                return default
            try:
                return cast(raw)
            except (TypeError, ValueError):
                LOGGER.warning("Ignoring invalid %s=%r", name, raw)
                return default

        return cls(
            association_radius=max(
                0.0,
                number("ASCENT_FRONTIER_ASSOCIATION_RADIUS", 0.6, float),
            ),
            blocked_radius=max(
                0.0,
                number("ASCENT_FRONTIER_BLOCKED_RADIUS", 0.75, float),
            ),
            min_progress=max(
                0.0, number("ASCENT_FRONTIER_MIN_PROGRESS", 0.15, float)
            ),
            max_stalled_selections=max(
                1,
                number("ASCENT_FRONTIER_MAX_STALLED_SELECTIONS", 20, int),
            ),
        )


@dataclass(frozen=True)
class FrontierProgressDecision:
    unreachable: bool
    target_anchor: np.ndarray
    current_distance: float
    best_distance: float
    stalled_selections: int
    reason: str


@dataclass
class _GoalProgress:
    anchor: np.ndarray
    best_distance: float
    stalled_selections: int = 0


class ExplorationFrontierProgressGuard:
    """Detect unreachable frontiers despite small map-induced goal drift.

    Frontier extraction can move the same boundary point slightly on every map
    update. Exact tuple matching therefore fails to accumulate a stuck count.
    This guard associates nearby goals into a spatial cluster and compares the
    robot's distance against the best distance ever achieved for that cluster.
    """

    def __init__(
        self,
        num_envs: int,
        config: Optional[FrontierProgressConfig] = None,
    ) -> None:
        self.config = config or FrontierProgressConfig.from_env()
        self._num_envs = num_envs
        self._progress: Dict[Tuple[int, int], List[_GoalProgress]] = {}
        self._blocked: Dict[Tuple[int, int], List[np.ndarray]] = {}

    @staticmethod
    def _key(env: int, floor_index: int) -> Tuple[int, int]:
        return int(env), int(floor_index)

    def reset(self, env: int) -> None:
        for storage in (self._progress, self._blocked):
            for key in [key for key in storage if key[0] == env]:
                del storage[key]

    @staticmethod
    def _xy(point: np.ndarray) -> np.ndarray:
        point = np.asarray(point, dtype=float).reshape(-1)
        if point.size < 2 or not np.all(np.isfinite(point[:2])):
            raise ValueError(f"Expected a finite XY frontier, got {point}")
        return point[:2].copy()

    def _nearest_progress(
        self, key: Tuple[int, int], goal: np.ndarray
    ) -> Optional[_GoalProgress]:
        records = self._progress.get(key, [])
        if not records:
            return None
        distances = [np.linalg.norm(record.anchor - goal) for record in records]
        index = int(np.argmin(distances))
        if distances[index] <= self.config.association_radius:
            return records[index]
        return None

    def is_blocked(
        self, *, env: int, floor_index: int, frontier: np.ndarray
    ) -> bool:
        goal = self._xy(frontier)
        centers = self._blocked.get(self._key(env, floor_index), [])
        return any(
            np.linalg.norm(goal - center) <= self.config.blocked_radius
            for center in centers
        )

    def filter_blocked(
        self, *, env: int, floor_index: int, frontiers: np.ndarray
    ) -> np.ndarray:
        points = np.asarray(frontiers)
        if points.size == 0:
            return points.reshape(0, 2)
        points = points.reshape(-1, 2)
        return np.asarray(
            [
                point
                for point in points
                if not self.is_blocked(
                    env=env, floor_index=floor_index, frontier=point
                )
            ],
            dtype=points.dtype,
        ).reshape(-1, 2)

    def update(
        self,
        *,
        env: int,
        floor_index: int,
        robot_xy: np.ndarray,
        frontier: np.ndarray,
    ) -> FrontierProgressDecision:
        goal = self._xy(frontier)
        robot = self._xy(robot_xy)
        key = self._key(env, floor_index)
        current_distance = float(np.linalg.norm(robot - goal))

        record = self._nearest_progress(key, goal)
        if record is None:
            record = _GoalProgress(
                anchor=goal, best_distance=current_distance
            )
            self._progress.setdefault(key, []).append(record)
            return FrontierProgressDecision(
                unreachable=False,
                target_anchor=record.anchor.copy(),
                current_distance=current_distance,
                best_distance=record.best_distance,
                stalled_selections=0,
                reason="tracking a new frontier region",
            )

        # Measure progress against the fixed cluster anchor. This prevents
        # frontier-coordinate jitter itself from looking like robot progress.
        current_distance = float(np.linalg.norm(robot - record.anchor))
        if current_distance <= record.best_distance - self.config.min_progress:
            record.best_distance = current_distance
            record.stalled_selections = 0
            reason = "distance to frontier region improved"
        else:
            record.stalled_selections += 1
            reason = "no new best distance to frontier region"

        unreachable = (
            record.stalled_selections >= self.config.max_stalled_selections
        )
        if unreachable:
            blocked_centers = self._blocked.setdefault(key, [])
            if not any(
                np.linalg.norm(record.anchor - center)
                <= self.config.blocked_radius
                for center in blocked_centers
            ):
                blocked_centers.append(record.anchor.copy())
            self._progress[key] = [
                candidate
                for candidate in self._progress[key]
                if candidate is not record
            ]
            reason = "frontier region made no meaningful navigation progress"

        return FrontierProgressDecision(
            unreachable=unreachable,
            target_anchor=record.anchor.copy(),
            current_distance=current_distance,
            best_distance=record.best_distance,
            stalled_selections=record.stalled_selections,
            reason=reason,
        )
