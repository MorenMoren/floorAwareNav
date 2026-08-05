"""A compact OpenFrontier-style semantic policy for stair traversal.

The public policy deliberately returns a short-range XY subgoal instead of a
Habitat action.  Ascent's existing PointNav policy remains responsible for
turning that subgoal into discrete actions, and Map_Controller remains the
single owner of stair-completion and floor-switching state.

The implementation uses dependency injection for the visual-frontier detector
and VLM scorer.  This keeps the policy testable and lets the OpenFrontier
adapter fail closed when the external checkout, weights, or VLM credentials
are unavailable.  In that case Ascent keeps using its legacy carrot target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import logging
import math
import os
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np


LOGGER = logging.getLogger(__name__)


class StairDirection(IntEnum):
    """Values intentionally match Map_Controller._climb_stair_flag."""

    UP = 1
    DOWN = 2


@dataclass
class StairFrontierCandidate:
    """One image-space frontier anchored into Ascent's episodic XY frame."""

    label: str
    pixel_xy: Tuple[int, int]
    goal_xy: np.ndarray
    information_gain: float
    image_stair_support: float = 0.0
    semantic_probability: float = 0.5
    semantic_reason: str = "Not scored"
    world_stair_support: float = 0.0
    utility: float = float("-inf")
    # Preserve the episodic 3D anchor for top-down-map debug rendering.
    # Navigation continues to consume only ``goal_xy``.
    goal_xyz: Optional[np.ndarray] = None
    # Populated by geometric/semantic filtering and exposed in debug output.
    filter_reason: str = "not evaluated"


@dataclass
class FrontierDetection:
    candidates: List[StairFrontierCandidate]
    marked_rgb: Optional[np.ndarray]


@dataclass(frozen=True)
class StairFrontierConfig:
    enabled: bool = True
    replan_interval: int = 4
    min_subgoal_distance: float = 0.3
    max_subgoal_distance: float = 2.5
    min_semantic_probability: float = 0.15
    semantic_power: float = 2.0
    gain_power: float = 1.0
    distance_power: float = 1.0
    turn_penalty: float = 0.15
    max_relative_heading_deg: float = 125.0
    stair_support_floor: float = 0.35
    world_stair_radius_px: int = 12
    hysteresis_ratio: float = 1.2
    # A selected OpenFrontier subgoal is locked until the robot enters this
    # radius.  PointNav commonly needs more than a few discrete actions to
    # cover an open staircase, so replanning on a short step interval makes
    # the goal drift with the camera instead of remaining a world-space target.
    goal_reached_distance: float = 0.8
    stuck_distance_epsilon: float = 0.05
    stuck_steps: int = 8

    @classmethod
    def from_env(cls) -> "StairFrontierConfig":
        """Build a config without expanding the structured Habitat schema."""

        def flag(name: str, default: bool) -> bool:
            raw = os.getenv(name)
            if raw is None:
                return default
            return raw.strip().lower() not in {"0", "false", "no", "off"}

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
            enabled=flag("ASCENT_SEMANTIC_STAIR_ENABLED", True),
            replan_interval=number("ASCENT_STAIR_REPLAN_INTERVAL", 4, int),
            min_subgoal_distance=number("ASCENT_STAIR_MIN_SUBGOAL_DISTANCE", 0.3, float),
            max_subgoal_distance=number("ASCENT_STAIR_MAX_SUBGOAL_DISTANCE", 2.5, float),
            min_semantic_probability=number(
                "ASCENT_STAIR_MIN_SEMANTIC_PROBABILITY", 0.15, float
            ),
            semantic_power=number("ASCENT_STAIR_SEMANTIC_POWER", 2.0, float),
            gain_power=number("ASCENT_STAIR_GAIN_POWER", 1.0, float),
            distance_power=number("ASCENT_STAIR_DISTANCE_POWER", 1.0, float),
            turn_penalty=number("ASCENT_STAIR_TURN_PENALTY", 0.15, float),
            max_relative_heading_deg=number("ASCENT_STAIR_MAX_HEADING_DEG", 125.0, float),
            stair_support_floor=number("ASCENT_STAIR_SUPPORT_FLOOR", 0.35, float),
            hysteresis_ratio=number("ASCENT_STAIR_HYSTERESIS_RATIO", 1.2, float),
            goal_reached_distance=number(
                "ASCENT_STAIR_GOAL_REACHED_DISTANCE", 0.8, float
            ),
            stuck_steps=number("ASCENT_STAIR_STUCK_STEPS", 8, int),
        )


@dataclass
class StairSubgoal:
    xy: np.ndarray
    label: str
    utility: float
    semantic_probability: float
    reason: str
    used_vlm: bool


@dataclass
class _EnvironmentState:
    active: Optional[StairSubgoal] = None
    last_replan_step: int = -10**9
    last_distance: float = float("inf")
    stagnant_steps: int = 0
    blocked_goals: List[np.ndarray] = field(default_factory=list)
    debug_candidates: List[StairFrontierCandidate] = field(default_factory=list)
    debug_valid_labels: set[str] = field(default_factory=set)
    debug_used_vlm: bool = False
    debug_marked_rgb: Optional[np.ndarray] = None
    debug_detection_step: int = -1


class FrontierCandidateDetector(Protocol):
    available: bool
    unavailable_reason: str

    def detect(
        self,
        rgb: np.ndarray,
        metric_depth: np.ndarray,
        camera_to_world: np.ndarray,
        stair_mask: Optional[np.ndarray],
    ) -> FrontierDetection:
        ...


class FrontierSemanticScorer(Protocol):
    available: bool
    unavailable_reason: str

    def score(
        self,
        marked_rgb: np.ndarray,
        labels: Sequence[str],
        direction: StairDirection,
    ) -> Dict[str, Tuple[float, str]]:
        ...


class SemanticStairFrontierPolicy:
    """Select a semantically grounded, short-range stair subgoal."""

    def __init__(
        self,
        detector: FrontierCandidateDetector,
        scorer: Optional[FrontierSemanticScorer],
        num_envs: int,
        config: Optional[StairFrontierConfig] = None,
    ) -> None:
        self.detector = detector
        self.scorer = scorer
        self.config = config or StairFrontierConfig.from_env()
        self._states = [_EnvironmentState() for _ in range(num_envs)]
        self._reported_unavailable = False

    @property
    def available(self) -> bool:
        return bool(self.config.enabled and getattr(self.detector, "available", False))

    def reset(self, env: int) -> None:
        self._states[env] = _EnvironmentState()

    def has_active_subgoal(self, env: int) -> bool:
        """Whether PointNav is currently following a locked semantic goal."""

        return self._states[env].active is not None

    def active_subgoal_distance(
        self, env: int, robot_xy: np.ndarray
    ) -> Optional[float]:
        """Return distance to the locked goal without running the detector."""

        active = self._states[env].active
        if active is None:
            return None
        robot_xy = np.asarray(robot_xy, dtype=float).reshape(2)
        return float(np.linalg.norm(active.xy - robot_xy))

    def complete_active_subgoal(self, env: int) -> None:
        """Release a reached goal while retaining its last debug detection."""

        state = self._states[env]
        state.active = None
        # Reaching a goal is an explicit request for the next local segment;
        # do not apply a previous failed-detection cooldown to it.
        state.last_replan_step = -10**9
        state.last_distance = float("inf")
        state.stagnant_steps = 0

    def debug_snapshot(self, env: int) -> Dict[str, object]:
        """Return the latest visual-frontier state without affecting navigation."""

        state = self._states[env]
        points_xyz = []
        image_pixels = []
        labels = []
        valid = []
        probabilities = []
        utilities = []
        reasons = []
        filter_reasons = []
        for candidate in state.debug_candidates:
            if candidate.goal_xyz is None:
                goal_xy = np.asarray(candidate.goal_xy, dtype=float).reshape(2)
                points_xyz.append([goal_xy[0], goal_xy[1], 0.0])
            else:
                points_xyz.append(
                    np.asarray(candidate.goal_xyz, dtype=float).reshape(3).tolist()
                )
            image_pixels.append(
                np.asarray(candidate.pixel_xy, dtype=float).reshape(2).tolist()
            )
            labels.append(candidate.label)
            valid.append(candidate.label in state.debug_valid_labels)
            probabilities.append(candidate.semantic_probability)
            utilities.append(candidate.utility)
            reasons.append(candidate.semantic_reason)
            filter_reasons.append(candidate.filter_reason)

        selected_label = state.active.label if state.active is not None else None
        selected_index = None
        if state.active is not None:
            for index, candidate in enumerate(state.debug_candidates):
                if np.linalg.norm(
                    np.asarray(candidate.goal_xy, dtype=float) - state.active.xy
                ) < 0.15:
                    selected_index = index
                    break
            if selected_index is None:
                # Hysteresis may retain a goal from the previous detection,
                # whose A/B/C labels have since been reassigned.
                points_xyz.append([state.active.xy[0], state.active.xy[1], 0.0])
                image_pixels.append([-1.0, -1.0])
                labels.append(state.active.label)
                valid.append(True)
                probabilities.append(state.active.semantic_probability)
                utilities.append(state.active.utility)
                reasons.append(state.active.reason)
                filter_reasons.append("cached active goal")
                selected_index = len(points_xyz) - 1

        return {
            "points_xyz": np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3),
            "image_pixels": np.asarray(image_pixels, dtype=np.float32).reshape(-1, 2),
            "labels": labels,
            "valid": np.asarray(valid, dtype=bool),
            "probabilities": np.asarray(probabilities, dtype=np.float32),
            "utilities": np.asarray(utilities, dtype=np.float32),
            "reasons": reasons,
            "filter_reasons": filter_reasons,
            "selected_label": selected_label,
            "selected_index": selected_index,
            "used_vlm": state.debug_used_vlm,
            "detection_step": state.debug_detection_step,
            "marked_rgb": (
                None
                if state.debug_marked_rgb is None
                else state.debug_marked_rgb.copy()
            ),
        }

    def propose_subgoal(
        self,
        *,
        env: int,
        step: int,
        rgb: np.ndarray,
        normalized_depth: np.ndarray,
        min_depth: float,
        max_depth: float,
        camera_to_world: np.ndarray,
        robot_xy: np.ndarray,
        heading: float,
        direction: int,
        stair_mask: Optional[np.ndarray] = None,
        world_stair_map: Optional[np.ndarray] = None,
        xy_to_px: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> Optional[StairSubgoal]:
        """Return a cached or newly selected subgoal; return None for legacy fallback."""

        if not self.available:
            if not self._reported_unavailable and self.config.enabled:
                LOGGER.warning(
                    "Semantic stair frontier disabled at runtime: %s",
                    getattr(self.detector, "unavailable_reason", "detector unavailable"),
                )
                self._reported_unavailable = True
            return None

        try:
            stair_direction = StairDirection(direction)
        except ValueError:
            return None

        state = self._states[env]
        robot_xy = np.asarray(robot_xy, dtype=float).reshape(2)

        if state.active is not None:
            active_distance = float(np.linalg.norm(state.active.xy - robot_xy))
            if state.last_distance - active_distance > self.config.stuck_distance_epsilon:
                state.stagnant_steps = 0
            else:
                state.stagnant_steps += 1
            state.last_distance = active_distance

            if active_distance > self.config.goal_reached_distance:
                # Keep the selected world-space OpenFrontier point as the
                # PointNav goal for the whole local segment.  In particular,
                # do not invoke FrontierNet/VLM merely because the old
                # replan interval elapsed or progress was temporarily slow.
                return state.active

            # Reaching the locked point completes this local stair segment.
            # Detection below may now select the next segment from the fresh
            # RGB-D observation.
            state.active = None
            state.last_replan_step = -10**9
            state.last_distance = float("inf")
            state.stagnant_steps = 0

        # Failed/empty detections are also rate limited. Without this guard an
        # empty FrontierNet result is recomputed every step and the caller's
        # legacy fallback never gets a chance to accumulate recovery state.
        if step - state.last_replan_step < self.config.replan_interval:
            return None

        depth = np.asarray(normalized_depth, dtype=np.float32).squeeze()
        if depth.ndim != 2 or depth.size == 0:
            return state.active
        metric_depth = depth * float(max_depth - min_depth) + float(min_depth)

        try:
            detection = self.detector.detect(
                rgb=np.asarray(rgb),
                metric_depth=metric_depth,
                camera_to_world=np.asarray(camera_to_world, dtype=float),
                stair_mask=None if stair_mask is None else np.asarray(stair_mask),
            )
        except Exception as exc:  # an external detector must never take down navigation
            LOGGER.warning("Semantic stair frontier detection failed: %s", exc)
            state.last_replan_step = step
            return state.active

        detected_candidates = detection.candidates
        state.debug_candidates = list(detected_candidates)
        state.debug_valid_labels = set()
        state.debug_used_vlm = False
        state.debug_marked_rgb = (
            None
            if detection.marked_rgb is None
            else np.asarray(detection.marked_rgb).copy()
        )
        state.debug_detection_step = step
        state.last_replan_step = step
        if not detected_candidates:
            return state.active

        used_vlm = False
        labels = [candidate.label for candidate in detected_candidates]
        if (
            len(detected_candidates) > 1
            and detection.marked_rgb is not None
            and self.scorer is not None
            and getattr(self.scorer, "available", False)
        ):
            try:
                scores = self.scorer.score(detection.marked_rgb, labels, stair_direction)
                used_vlm = bool(scores)
            except Exception as exc:
                LOGGER.warning("Semantic stair VLM scoring failed: %s", exc)
                scores = {}
        else:
            scores = {}
        state.debug_used_vlm = used_vlm

        # Score every mark shown in the image. Filtering first would leave
        # extra, unexplained labels visible to the VLM.
        for candidate in detected_candidates:
            if candidate.label in scores:
                probability, reason = scores[candidate.label]
                candidate.semantic_probability = float(np.clip(probability, 0.0, 1.0))
                candidate.semantic_reason = str(reason)

        candidates = self._filter_candidates(
            detected_candidates,
            robot_xy=robot_xy,
            heading=float(heading),
            blocked_goals=state.blocked_goals,
            world_stair_map=world_stair_map,
            xy_to_px=xy_to_px,
            scored_labels=set(scores),
        )
        state.debug_valid_labels = {candidate.label for candidate in candidates}
        if not candidates:
            # A fresh observation explicitly rejected every visible candidate.
            # Do not keep steering toward a stale (possibly semantically wrong)
            # frontier; let the caller use the legacy carrot for this step.
            state.active = None
            return None

        for candidate in candidates:
            candidate.utility = self._utility(candidate, robot_xy, float(heading))

        best = max(candidates, key=lambda item: item.utility)
        proposed = StairSubgoal(
            xy=best.goal_xy.copy(),
            label=best.label,
            utility=float(best.utility),
            semantic_probability=float(best.semantic_probability),
            reason=best.semantic_reason,
            used_vlm=used_vlm,
        )

        if state.active is not None and not self._should_replace(state.active, proposed):
            state.last_replan_step = step
            return state.active

        state.active = proposed
        state.last_replan_step = step
        state.last_distance = float(np.linalg.norm(proposed.xy - robot_xy))
        state.stagnant_steps = 0
        return proposed

    def _filter_candidates(
        self,
        candidates: Sequence[StairFrontierCandidate],
        *,
        robot_xy: np.ndarray,
        heading: float,
        blocked_goals: Sequence[np.ndarray],
        world_stair_map: Optional[np.ndarray],
        xy_to_px: Optional[Callable[[np.ndarray], np.ndarray]],
        scored_labels: set[str],
    ) -> List[StairFrontierCandidate]:
        valid: List[StairFrontierCandidate] = []
        max_heading = math.radians(self.config.max_relative_heading_deg)

        for candidate in candidates:
            candidate.filter_reason = "not evaluated"
            candidate.goal_xy = np.asarray(candidate.goal_xy, dtype=float).reshape(2)
            if not np.all(np.isfinite(candidate.goal_xy)):
                candidate.filter_reason = "non-finite 3D anchor"
                continue
            if (
                candidate.label in scored_labels
                and candidate.semantic_probability
                < self.config.min_semantic_probability
            ):
                candidate.filter_reason = (
                    "semantic probability below threshold "
                    f"({candidate.semantic_probability:.2f} < "
                    f"{self.config.min_semantic_probability:.2f})"
                )
                continue
            distance = float(np.linalg.norm(candidate.goal_xy - robot_xy))
            min_navigation_distance = max(
                self.config.min_subgoal_distance,
                self.config.goal_reached_distance,
            )
            if not min_navigation_distance < distance <= self.config.max_subgoal_distance:
                candidate.filter_reason = (
                    f"distance {distance:.2f}m outside "
                    f"({min_navigation_distance:.2f}, "
                    f"{self.config.max_subgoal_distance:.2f}]m"
                )
                continue
            if any(np.linalg.norm(candidate.goal_xy - blocked) < 0.25 for blocked in blocked_goals):
                candidate.filter_reason = "near a previously blocked goal"
                continue
            relative_heading = abs(_wrap_angle(math.atan2(
                candidate.goal_xy[1] - robot_xy[1],
                candidate.goal_xy[0] - robot_xy[0],
            ) - heading))
            if relative_heading > max_heading:
                candidate.filter_reason = (
                    f"relative heading {math.degrees(relative_heading):.1f}deg exceeds "
                    f"{self.config.max_relative_heading_deg:.1f}deg"
                )
                continue

            candidate.world_stair_support = self._world_stair_support(
                candidate.goal_xy, world_stair_map, xy_to_px
            )
            candidate.filter_reason = "valid"
            valid.append(candidate)
        return valid

    def _world_stair_support(
        self,
        goal_xy: np.ndarray,
        world_stair_map: Optional[np.ndarray],
        xy_to_px: Optional[Callable[[np.ndarray], np.ndarray]],
    ) -> float:
        if world_stair_map is None or xy_to_px is None or not np.any(world_stair_map):
            return 0.0
        try:
            px = np.asarray(xy_to_px(np.atleast_2d(goal_xy)))[0]
            x, y = int(round(float(px[0]))), int(round(float(px[1])))
        except Exception:
            return 0.0
        radius = self.config.world_stair_radius_px
        y0, y1 = max(0, y - radius), min(world_stair_map.shape[0], y + radius + 1)
        x0, x1 = max(0, x - radius), min(world_stair_map.shape[1], x + radius + 1)
        if y0 >= y1 or x0 >= x1:
            return 0.0
        return float(np.mean(np.asarray(world_stair_map[y0:y1, x0:x1]) > 0))

    def _utility(
        self,
        candidate: StairFrontierCandidate,
        robot_xy: np.ndarray,
        heading: float,
    ) -> float:
        distance = max(float(np.linalg.norm(candidate.goal_xy - robot_xy)), 1e-3)
        goal_heading = math.atan2(
            candidate.goal_xy[1] - robot_xy[1], candidate.goal_xy[0] - robot_xy[0]
        )
        relative_heading = abs(_wrap_angle(goal_heading - heading))
        gain = max(float(candidate.information_gain), 1e-4)
        probability = float(np.clip(candidate.semantic_probability, 0.0, 1.0))
        support = max(
            float(candidate.image_stair_support),
            float(candidate.world_stair_support),
            self.config.stair_support_floor,
        )
        score = (
            probability ** self.config.semantic_power
            * gain ** self.config.gain_power
            * support
            / distance ** self.config.distance_power
        )
        return float(score - self.config.turn_penalty * relative_heading / math.pi)

    def _should_replace(self, active: StairSubgoal, proposed: StairSubgoal) -> bool:
        if np.linalg.norm(active.xy - proposed.xy) < 0.15:
            return True
        # Multiplying a negative utility by a ratio inverts the intended
        # hysteresis (for example -0.09 * 1.2 == -0.108).  Use an additive
        # improvement margin so replacement always requires a better score.
        improvement_margin = max(
            abs(active.utility) * (self.config.hysteresis_ratio - 1.0),
            1e-3,
        )
        return proposed.utility >= active.utility + improvement_margin


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def build_semantic_stair_frontier_policy(
    *,
    num_envs: int,
    camera_intrinsic: np.ndarray,
    device: str,
    shared_qwen_vlm,
) -> SemanticStairFrontierPolicy:
    """Construct the optional OpenFrontier and VLM adapters lazily."""

    from .openfrontier_adapter import OpenFrontierDetectorAdapter
    from .stair_vlm import build_stair_vlm_scorer

    config = StairFrontierConfig.from_env()
    detector = OpenFrontierDetectorAdapter(
        camera_intrinsic=np.asarray(camera_intrinsic, dtype=np.float32),
        device=device,
    )
    scorer = build_stair_vlm_scorer(shared_qwen_vlm)
    return SemanticStairFrontierPolicy(
        detector=detector,
        scorer=scorer,
        num_envs=num_envs,
        config=config,
    )
