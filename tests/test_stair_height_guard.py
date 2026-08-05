import unittest

import numpy as np

from ascent.stair_policy.height_guard import (
    StairHeightGuard,
    StairHeightGuardConfig,
)


class StairHeightGuardTest(unittest.TestCase):
    def make_guard(self):
        return StairHeightGuard(
            num_envs=1,
            config=StairHeightGuardConfig(
                window_steps=3,
                change_threshold=0.05,
                exit_stable_steps=2,
            ),
        )

    def test_upward_height_change_keeps_climb_active_outside_map(self):
        guard = self.make_guard()

        decisions = [
            guard.update(env=0, height=h, direction=1, in_stair_map=False)
            for h in (0.0, 0.06, 0.12)
        ]

        self.assertFalse(decisions[-1].allow_map_exit)
        self.assertTrue(decisions[-1].height_changing)
        self.assertAlmostEqual(decisions[-1].directional_change, 0.12)

    def test_exit_is_confirmed_after_upward_height_stabilizes(self):
        guard = self.make_guard()
        decisions = [
            guard.update(env=0, height=h, direction=1, in_stair_map=False)
            for h in (0.0, 0.06, 0.12, 0.12, 0.12, 0.12)
        ]

        self.assertFalse(decisions[-2].allow_map_exit)
        self.assertTrue(decisions[-1].allow_map_exit)
        self.assertFalse(decisions[-1].height_changing)

    def test_downward_height_change_keeps_climb_active(self):
        guard = self.make_guard()
        guard.update(env=0, height=0.0, direction=2, in_stair_map=True)
        decision = guard.update(
            env=0, height=-0.08, direction=2, in_stair_map=False
        )

        self.assertFalse(decision.allow_map_exit)
        self.assertTrue(decision.height_changing)
        self.assertAlmostEqual(decision.directional_change, 0.08)

    def test_returning_inside_map_resets_outside_confirmation(self):
        guard = self.make_guard()
        first = guard.update(env=0, height=0.0, direction=1, in_stair_map=False)
        inside = guard.update(env=0, height=0.0, direction=1, in_stair_map=True)
        outside = guard.update(env=0, height=0.0, direction=1, in_stair_map=False)

        self.assertEqual(first.outside_stable_steps, 1)
        self.assertEqual(inside.outside_stable_steps, 0)
        self.assertEqual(outside.outside_stable_steps, 1)
        self.assertFalse(outside.allow_map_exit)

    def test_missing_height_preserves_legacy_map_exit(self):
        guard = self.make_guard()
        decision = guard.update(
            env=0, height=np.nan, direction=1, in_stair_map=False
        )

        self.assertTrue(decision.allow_map_exit)
        self.assertFalse(decision.height_available)


if __name__ == "__main__":
    unittest.main()
