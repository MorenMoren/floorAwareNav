import json
import unittest

import numpy as np

from ascent.stair_policy.semantic_frontier_policy import (
    FrontierDetection,
    SemanticStairFrontierPolicy,
    StairDirection,
    StairFrontierCandidate,
    StairFrontierConfig,
    StairSubgoal,
)
from ascent.stair_policy.openfrontier_adapter import OpenFrontierDetectorAdapter
from ascent.stair_policy.stair_vlm import (
    QwenStairVLMScorer,
    build_stair_prompt,
    parse_frontier_scores,
)


class FakeDetector:
    available = True
    unavailable_reason = ""

    def __init__(self, candidates):
        self.candidates = candidates
        self.calls = 0

    def detect(self, rgb, metric_depth, camera_to_world, stair_mask):
        self.calls += 1
        return FrontierDetection(
            candidates=[
                StairFrontierCandidate(
                    label=c.label,
                    pixel_xy=c.pixel_xy,
                    goal_xy=c.goal_xy.copy(),
                    information_gain=c.information_gain,
                    image_stair_support=c.image_stair_support,
                    goal_xyz=None if c.goal_xyz is None else c.goal_xyz.copy(),
                )
                for c in self.candidates
            ],
            marked_rgb=np.zeros((64, 64, 3), dtype=np.uint8),
        )


class FakeScorer:
    available = True
    unavailable_reason = ""

    def __init__(self, scores):
        self.scores = scores
        self.calls = 0
        self.last_direction = None

    def score(self, marked_rgb, labels, direction):
        self.calls += 1
        self.last_direction = direction
        return self.scores


class UnavailableDetector:
    available = False
    unavailable_reason = "test detector unavailable"


class FakeQwenVLM:
    def __init__(self):
        self.calls = []

    def chat_with_image(self, prompt, image):
        self.calls.append((prompt, image))
        return json.dumps({"A": [0.2, "side"], "B": [0.8, "continues down"]})


class SemanticStairFrontierPolicyTest(unittest.TestCase):
    def make_inputs(self):
        return dict(
            env=0,
            step=10,
            rgb=np.zeros((32, 48, 3), dtype=np.uint8),
            normalized_depth=np.full((32, 48), 0.5, dtype=np.float32),
            min_depth=0.0,
            max_depth=10.0,
            camera_to_world=np.eye(4),
            robot_xy=np.array([0.0, 0.0]),
            heading=0.0,
            direction=1,
            stair_mask=np.ones((32, 48), dtype=bool),
        )

    def test_vlm_semantics_selects_upward_candidate(self):
        detector = FakeDetector(
            [
                StairFrontierCandidate("A", (10, 10), np.array([1.0, 0.0]), 1.0, 1.0),
                StairFrontierCandidate("B", (20, 10), np.array([1.2, 0.2]), 1.0, 1.0),
            ]
        )
        scorer = FakeScorer(
            {"A": (0.1, "side landing"), "B": (0.9, "continues upward")}
        )
        policy = SemanticStairFrontierPolicy(
            detector,
            scorer,
            num_envs=1,
            config=StairFrontierConfig(replan_interval=4),
        )

        result = policy.propose_subgoal(**self.make_inputs())

        self.assertIsNotNone(result)
        self.assertEqual(result.label, "B")
        self.assertTrue(result.used_vlm)
        self.assertEqual(scorer.last_direction, StairDirection.UP)

    def test_selected_goal_stays_locked_until_robot_is_within_point_eight_metres(self):
        detector = FakeDetector(
            [StairFrontierCandidate("A", (10, 10), np.array([2.0, 0.0]), 1.0, 1.0)]
        )
        policy = SemanticStairFrontierPolicy(
            detector,
            scorer=None,
            num_envs=1,
            config=StairFrontierConfig(replan_interval=4),
        )
        inputs = self.make_inputs()
        first = policy.propose_subgoal(**inputs)
        self.assertTrue(policy.has_active_subgoal(0))
        self.assertAlmostEqual(
            policy.active_subgoal_distance(0, np.array([0.0, 0.0])),
            2.0,
        )

        # Exceed both the old four-step replan interval and the old eight-step
        # stagnation threshold.  The selected world point must remain locked.
        for step in range(11, 31):
            inputs["step"] = step
            inputs["robot_xy"] = np.array([0.0, 0.0])
            locked = policy.propose_subgoal(**inputs)
            np.testing.assert_allclose(first.xy, locked.xy)

        inputs["step"] = 31
        inputs["robot_xy"] = np.array([1.19, 0.0])  # 0.81m from the goal
        still_locked = policy.propose_subgoal(**inputs)

        self.assertEqual(detector.calls, 1)
        np.testing.assert_allclose(first.xy, still_locked.xy)

        inputs["step"] = 32
        inputs["robot_xy"] = np.array([1.21, 0.0])  # 0.79m from the goal
        policy.propose_subgoal(**inputs)

        self.assertEqual(detector.calls, 2)
        self.assertFalse(policy.has_active_subgoal(0))

    def test_unavailable_detector_returns_none_for_legacy_fallback(self):
        policy = SemanticStairFrontierPolicy(
            UnavailableDetector(), scorer=None, num_envs=1
        )
        self.assertIsNone(policy.propose_subgoal(**self.make_inputs()))

    def test_empty_detection_is_rate_limited_during_legacy_fallback(self):
        detector = FakeDetector([])
        policy = SemanticStairFrontierPolicy(
            detector,
            scorer=None,
            num_envs=1,
            config=StairFrontierConfig(replan_interval=4),
        )
        inputs = self.make_inputs()

        for step in range(10, 14):
            inputs["step"] = step
            self.assertIsNone(policy.propose_subgoal(**inputs))
        self.assertEqual(detector.calls, 1)

        inputs["step"] = 14
        self.assertIsNone(policy.propose_subgoal(**inputs))
        self.assertEqual(detector.calls, 2)

    def test_filters_behind_and_out_of_range_candidates(self):
        detector = FakeDetector(
            [
                StairFrontierCandidate("A", (10, 10), np.array([-1.0, 0.0]), 10.0, 1.0),
                StairFrontierCandidate("B", (20, 10), np.array([5.0, 0.0]), 10.0, 1.0),
                StairFrontierCandidate("C", (30, 10), np.array([1.0, 0.0]), 1.0, 1.0),
            ]
        )
        policy = SemanticStairFrontierPolicy(detector, None, num_envs=1)
        result = policy.propose_subgoal(**self.make_inputs())
        self.assertEqual(result.label, "C")

    def test_debug_snapshot_contains_all_candidates_and_selected_point(self):
        detector = FakeDetector(
            [
                StairFrontierCandidate(
                    "A",
                    (10, 10),
                    np.array([-1.0, 0.0]),
                    1.0,
                    1.0,
                    goal_xyz=np.array([-1.0, 0.0, 0.2]),
                ),
                StairFrontierCandidate(
                    "B",
                    (20, 10),
                    np.array([1.0, 0.0]),
                    1.0,
                    1.0,
                    goal_xyz=np.array([1.0, 0.0, 0.4]),
                ),
            ]
        )
        policy = SemanticStairFrontierPolicy(detector, None, num_envs=1)

        policy.propose_subgoal(**self.make_inputs())
        debug = policy.debug_snapshot(0)

        self.assertEqual(debug["labels"], ["A", "B"])
        self.assertEqual(debug["valid"].tolist(), [False, True])
        self.assertEqual(debug["selected_label"], "B")
        self.assertEqual(debug["selected_index"], 1)
        self.assertEqual(debug["marked_rgb"].shape, (64, 64, 3))
        np.testing.assert_allclose(debug["image_pixels"], [[10, 10], [20, 10]])
        np.testing.assert_allclose(
            debug["points_xyz"],
            [[-1.0, 0.0, 0.2], [1.0, 0.0, 0.4]],
        )
        self.assertIn("relative heading", debug["filter_reasons"][0])
        self.assertEqual(debug["filter_reasons"][1], "valid")
        self.assertEqual(debug["detection_step"], 10)

    def test_rejects_zero_probability_vlm_candidates_for_legacy_fallback(self):
        detector = FakeDetector(
            [
                StairFrontierCandidate("A", (10, 10), np.array([1.0, 0.0]), 1.0),
                StairFrontierCandidate("B", (20, 10), np.array([1.0, 0.2]), 1.0),
            ]
        )
        scorer = FakeScorer(
            {
                "A": (0.0, "wall above landing"),
                "B": (0.0, "not a traversable stair path"),
            }
        )
        policy = SemanticStairFrontierPolicy(
            detector,
            scorer,
            num_envs=1,
            config=StairFrontierConfig(min_semantic_probability=0.15),
        )

        result = policy.propose_subgoal(**self.make_inputs())
        debug = policy.debug_snapshot(0)

        self.assertIsNone(result)
        self.assertEqual(debug["valid"].tolist(), [False, False])
        self.assertTrue(
            all("semantic probability below threshold" in reason
                for reason in debug["filter_reasons"])
        )

    def test_negative_utility_hysteresis_requires_real_improvement(self):
        detector = FakeDetector([])
        policy = SemanticStairFrontierPolicy(detector, None, num_envs=1)
        active = StairSubgoal(
            xy=np.array([1.0, 0.0]),
            label="A",
            utility=-0.09,
            semantic_probability=0.2,
            reason="active",
            used_vlm=True,
        )
        worse = StairSubgoal(
            xy=np.array([1.0, 1.0]),
            label="B",
            utility=-0.10,
            semantic_probability=0.2,
            reason="worse",
            used_vlm=True,
        )

        self.assertFalse(policy._should_replace(active, worse))


class OpenFrontierCoordinateConventionTest(unittest.TestCase):
    def test_opencv_camera_axes_are_rebased_to_ascent_axes(self):
        camera_to_world = np.eye(4)
        camera_to_world[:3, 3] = [10.0, 20.0, 30.0]
        cv_camera_to_world = OpenFrontierDetectorAdapter.opencv_camera_to_world(
            camera_to_world
        )

        cv_forward = cv_camera_to_world @ np.array([0.0, 0.0, 2.0, 1.0])
        cv_right = cv_camera_to_world @ np.array([1.0, 0.0, 0.0, 1.0])
        cv_down = cv_camera_to_world @ np.array([0.0, 1.0, 0.0, 1.0])

        np.testing.assert_allclose(cv_forward[:3], [12.0, 20.0, 30.0])
        np.testing.assert_allclose(cv_right[:3], [10.0, 19.0, 30.0])
        np.testing.assert_allclose(cv_down[:3], [10.0, 20.0, 29.0])


class StairVLMTest(unittest.TestCase):
    def test_prompt_describes_affordance_not_object_detection(self):
        up = build_stair_prompt(["A", "B"], StairDirection.UP)
        down = build_stair_prompt(["A", "B"], StairDirection.DOWN)
        self.assertIn("UPWARD", up)
        self.assertIn("DOWNWARD", down)
        self.assertIn("not merely whether staircase pixels are visible", up)

    def test_parse_scores_accepts_fenced_json_and_clamps_probability(self):
        raw = "```json\n" + json.dumps(
            {"A": [1.2, "good"], "B": [-0.1, "bad"]}
        ) + "\n```"
        scores = parse_frontier_scores(raw, ["A", "B", "C"])
        self.assertEqual(scores["A"], (1.0, "good"))
        self.assertEqual(scores["B"], (0.0, "bad"))
        self.assertNotIn("C", scores)

    def test_qwen_scorer_reuses_multimodal_client(self):
        qwen = FakeQwenVLM()
        scorer = QwenStairVLMScorer(qwen)
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        scores = scorer.score(image, ["A", "B"], StairDirection.DOWN)

        self.assertEqual(len(qwen.calls), 1)
        self.assertIn("DOWNWARD", qwen.calls[0][0])
        self.assertIs(qwen.calls[0][1], image)
        self.assertEqual(scores["B"], (0.8, "continues down"))


if __name__ == "__main__":
    unittest.main()
