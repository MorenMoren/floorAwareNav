"""Height-based guard for noisy 2D stair-map exit decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
from typing import List, Optional

import numpy as np


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class StairHeightGuardConfig:
    """Thresholds for confirming that vertical stair motion has ended."""

    window_steps: int = 2
    change_threshold: float = 0.05
    exit_stable_steps: int = 3

    @classmethod
    def from_env(cls) -> "StairHeightGuardConfig":
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
            window_steps=max(
                2, number("ASCENT_STAIR_HEIGHT_WINDOW_STEPS", 5, int)
            ),
            change_threshold=max(
                0.0,
                number("ASCENT_STAIR_HEIGHT_CHANGE_THRESHOLD", 0.03, float),
            ),
            exit_stable_steps=max(
                1, number("ASCENT_STAIR_EXIT_STABLE_STEPS", 6, int)
            ),
        )


@dataclass(frozen=True)
class StairHeightDecision:
    allow_map_exit: bool
    height_available: bool
    directional_change: float
    outside_stable_steps: int
    height_changing: bool


@dataclass
class _HeightState:
    direction: int = 0
    history: List[float] = field(default_factory=list)
    outside_stable_steps: int = 0


class StairHeightGuard:
    """Keep stair mode active while height changes in the climb direction."""

    def __init__(
        self,
        num_envs: int,
        config: Optional[StairHeightGuardConfig] = None,
    ) -> None:
        self.config = config or StairHeightGuardConfig.from_env()
        self._states = [_HeightState() for _ in range(num_envs)]

    def reset(self, env: int) -> None:
        self._states[env] = _HeightState()

    def update(
        self,
        *,
        env: int,
        height: float,
        direction: int,
        in_stair_map: bool,
    ) -> StairHeightDecision:
        """Update height history and decide whether a map exit is credible.

        ``direction`` follows ASCENT's convention: 1 is upstairs and 2 is
        downstairs. Missing height data preserves the legacy map-only exit.
        """

        if not np.isfinite(height) or direction not in (1, 2):
            return StairHeightDecision(True, False, 0.0, 0, False)

        state = self._states[env]
        if state.direction != direction:
            state = _HeightState(direction=direction)
            self._states[env] = state

        state.history.append(float(height))
        if len(state.history) > self.config.window_steps:
            del state.history[: len(state.history) - self.config.window_steps]

        if direction == 1:
            directional_change = float(state.history[-1] - min(state.history))
        else:
            directional_change = float(max(state.history) - state.history[-1])
        directional_change = max(0.0, directional_change)
        height_changing = directional_change >= self.config.change_threshold

        if in_stair_map or height_changing:
            state.outside_stable_steps = 0
        else:
            state.outside_stable_steps += 1

        allow_map_exit = (
            not in_stair_map
            and not height_changing
            and state.outside_stable_steps >= self.config.exit_stable_steps
        )
        return StairHeightDecision(
            allow_map_exit=allow_map_exit,
            height_available=True,
            directional_change=directional_change,
            outside_stable_steps=state.outside_stable_steps,
            height_changing=height_changing,
        )
