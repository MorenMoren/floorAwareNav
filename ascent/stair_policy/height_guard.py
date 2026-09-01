"""Height-based guards for stair exit and legacy-carrot direction decisions."""

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


@dataclass(frozen=True)
class PassiveStairEntryConfig:
    """Evidence required before a 2D stair-map overlap triggers climbing."""

    window_steps: int = 6
    min_map_steps: int = 3
    change_threshold: float = 0.05

    @classmethod
    def from_env(cls) -> "PassiveStairEntryConfig":
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
                2, number("ASCENT_PASSIVE_STAIR_HEIGHT_WINDOW_STEPS", 6, int)
            ),
            min_map_steps=max(
                1, number("ASCENT_PASSIVE_STAIR_MIN_MAP_STEPS", 3, int)
            ),
            change_threshold=max(
                0.0,
                number(
                    "ASCENT_PASSIVE_STAIR_HEIGHT_CHANGE_THRESHOLD", 0.05, float
                ),
            ),
        )


@dataclass(frozen=True)
class PassiveStairEntryDecision:
    confirmed: bool
    height_available: bool
    directional_change: float
    map_steps: int


@dataclass
class _PassiveEntryState:
    history: List[float] = field(default_factory=list)
    map_steps: int = 0


class PassiveStairEntryHeightGuard:
    """Require vertical motion before passively entering stair mode.

    Open stair geometry can project onto the floor below it, making a robot
    standing underneath appear inside the 2D stair map. A map hit is therefore
    treated only as a candidate; upstairs requires rising height and downstairs
    requires falling height over the recent in-map observation window.
    """

    def __init__(
        self,
        num_envs: int,
        config: Optional[PassiveStairEntryConfig] = None,
    ) -> None:
        self.config = config or PassiveStairEntryConfig.from_env()
        self._states = [
            {1: _PassiveEntryState(), 2: _PassiveEntryState()}
            for _ in range(num_envs)
        ]

    def reset(self, env: int, direction: Optional[int] = None) -> None:
        if direction in (1, 2):
            self._states[env][direction] = _PassiveEntryState()
        else:
            self._states[env] = {
                1: _PassiveEntryState(),
                2: _PassiveEntryState(),
            }

    def update(
        self,
        *,
        env: int,
        height: float,
        direction: int,
        in_stair_map: bool,
    ) -> PassiveStairEntryDecision:
        if direction not in (1, 2):
            return PassiveStairEntryDecision(False, False, 0.0, 0)

        if not in_stair_map:
            self.reset(env, direction)
            return PassiveStairEntryDecision(
                False, bool(np.isfinite(height)), 0.0, 0
            )

        if not np.isfinite(height):
            # Passive entry is intentionally disabled without 3D GPS. Falling
            # back to map-only triggering would reintroduce the false positive
            # this guard is designed to prevent.
            self.reset(env, direction)
            return PassiveStairEntryDecision(False, False, 0.0, 0)

        state = self._states[env][direction]
        state.map_steps += 1
        state.history.append(float(height))
        if len(state.history) > self.config.window_steps:
            del state.history[: len(state.history) - self.config.window_steps]

        if direction == 1:
            directional_change = float(state.history[-1] - min(state.history))
        else:
            directional_change = float(max(state.history) - state.history[-1])
        directional_change = max(0.0, directional_change)
        confirmed = (
            state.map_steps >= self.config.min_map_steps
            and directional_change >= self.config.change_threshold
        )
        return PassiveStairEntryDecision(
            confirmed=confirmed,
            height_available=True,
            directional_change=directional_change,
            map_steps=state.map_steps,
        )


@dataclass(frozen=True)
class CarrotHeightDirectionConfig:
    """Controls height validation for the legacy stair-carrot fallback."""

    probe_forward_steps: int = 4
    change_threshold: float = 0.03
    rotate_steps: int = 3

    @classmethod
    def from_env(cls) -> "CarrotHeightDirectionConfig":
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
            probe_forward_steps=max(
                1,
                number("ASCENT_CARROT_HEIGHT_PROBE_FORWARD_STEPS", 4, int),
            ),
            change_threshold=max(
                0.0,
                number("ASCENT_CARROT_HEIGHT_CHANGE_THRESHOLD", 0.03, float),
            ),
            rotate_steps=max(
                1, number("ASCENT_CARROT_DIRECTION_ROTATE_STEPS", 3, int)
            ),
        )


@dataclass(frozen=True)
class CarrotHeightDirectionDecision:
    should_rotate: bool
    restart_carrot: bool
    height_available: bool
    direction_confirmed: bool
    signed_height_change: float
    probe_forward_steps: int
    completed_rotate_steps: int
    reason: str


@dataclass
class _CarrotHeightDirectionState:
    direction: int = 0
    probe_start_height: float = float("nan")
    probe_forward_steps: int = 0
    searching: bool = False
    completed_rotate_steps: int = 0


class CarrotHeightDirectionGuard:
    """Validate a legacy carrot using the robot's resulting height change.

    A carrot cannot be validated from height before the robot moves.  The
    guard therefore lets PointNav execute a short forward probe.  If that
    probe moves vertically in the requested direction, navigation continues.
    If it moves in the opposite direction or remains vertically flat, the
    guard requests a consecutive turn sweep and then starts a fresh probe with
    a newly generated carrot.
    """

    def __init__(
        self,
        num_envs: int,
        config: Optional[CarrotHeightDirectionConfig] = None,
    ) -> None:
        self.config = config or CarrotHeightDirectionConfig.from_env()
        self._states = [
            _CarrotHeightDirectionState() for _ in range(num_envs)
        ]

    def reset(self, env: int) -> None:
        self._states[env] = _CarrotHeightDirectionState()

    @staticmethod
    def _decision(
        state: _CarrotHeightDirectionState,
        *,
        should_rotate: bool,
        restart_carrot: bool,
        height_available: bool,
        direction_confirmed: bool,
        signed_height_change: float,
        reason: str,
    ) -> CarrotHeightDirectionDecision:
        return CarrotHeightDirectionDecision(
            should_rotate=should_rotate,
            restart_carrot=restart_carrot,
            height_available=height_available,
            direction_confirmed=direction_confirmed,
            signed_height_change=signed_height_change,
            probe_forward_steps=state.probe_forward_steps,
            completed_rotate_steps=state.completed_rotate_steps,
            reason=reason,
        )

    def update(
        self,
        *,
        env: int,
        height: float,
        direction: int,
        previous_action_was_forward: bool,
        previous_action_was_search_turn: bool,
    ) -> CarrotHeightDirectionDecision:
        """Return whether the caller must override PointNav with a turn.

        ``direction`` follows ASCENT: 1 means upstairs and 2 downstairs.
        Positive ``signed_height_change`` always means progress in the desired
        direction, regardless of whether the robot is climbing or descending.
        """

        if not np.isfinite(height) or direction not in (1, 2):
            self.reset(env)
            state = self._states[env]
            return self._decision(
                state,
                should_rotate=False,
                restart_carrot=False,
                height_available=False,
                direction_confirmed=False,
                signed_height_change=0.0,
                reason="height unavailable; preserving legacy carrot behaviour",
            )

        state = self._states[env]
        if state.direction != direction:
            state = _CarrotHeightDirectionState(
                direction=direction, probe_start_height=float(height)
            )
            self._states[env] = state

        direction_sign = 1.0 if direction == 1 else -1.0
        signed_height_change = direction_sign * (
            float(height) - state.probe_start_height
        )

        if state.searching:
            if previous_action_was_search_turn:
                state.completed_rotate_steps += 1

            if state.completed_rotate_steps < self.config.rotate_steps:
                return self._decision(
                    state,
                    should_rotate=True,
                    restart_carrot=True,
                    height_available=True,
                    direction_confirmed=False,
                    signed_height_change=signed_height_change,
                    reason="searching for a new stair direction",
                )

            # A turn sweep has exposed a new view.  Discard the old carrot and
            # use the current height as the next forward-probe baseline.
            state.searching = False
            state.completed_rotate_steps = 0
            state.probe_forward_steps = 0
            state.probe_start_height = float(height)
            return self._decision(
                state,
                should_rotate=False,
                restart_carrot=True,
                height_available=True,
                direction_confirmed=False,
                signed_height_change=0.0,
                reason="turn sweep complete; probing a newly generated carrot",
            )

        if not previous_action_was_forward:
            return self._decision(
                state,
                should_rotate=False,
                restart_carrot=False,
                height_available=True,
                direction_confirmed=False,
                signed_height_change=signed_height_change,
                reason="waiting for a forward probe",
            )

        state.probe_forward_steps += 1
        threshold = self.config.change_threshold
        if signed_height_change >= threshold:
            state.probe_start_height = float(height)
            state.probe_forward_steps = 0
            return self._decision(
                state,
                should_rotate=False,
                restart_carrot=False,
                height_available=True,
                direction_confirmed=True,
                signed_height_change=signed_height_change,
                reason="height changed in the requested stair direction",
            )

        moved_opposite = signed_height_change <= -threshold
        probe_expired = (
            state.probe_forward_steps >= self.config.probe_forward_steps
        )
        if moved_opposite or probe_expired:
            state.searching = True
            state.completed_rotate_steps = 0
            reason = (
                "height changed opposite to the requested stair direction"
                if moved_opposite
                else "forward probe produced no requested height change"
            )
            return self._decision(
                state,
                should_rotate=True,
                restart_carrot=True,
                height_available=True,
                direction_confirmed=False,
                signed_height_change=signed_height_change,
                reason=reason,
            )

        return self._decision(
            state,
            should_rotate=False,
            restart_carrot=False,
            height_available=True,
            direction_confirmed=False,
            signed_height_change=signed_height_change,
            reason="collecting forward-probe height samples",
        )
