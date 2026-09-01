import unittest

import numpy as np

from ascent.stair_policy.height_guard import (
    CarrotHeightDirectionConfig,
    CarrotHeightDirectionGuard,
    PassiveStairEntryConfig,
    PassiveStairEntryHeightGuard,
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


class PassiveStairEntryHeightGuardTest(unittest.TestCase):
    def make_guard(self):
        return PassiveStairEntryHeightGuard(
            num_envs=1,
            config=PassiveStairEntryConfig(
                window_steps=5,
                min_map_steps=3,
                change_threshold=0.05,
            ),
        )

    @staticmethod
    def update(guard, height, direction=1, in_map=True):
        return guard.update(
            env=0,
            height=height,
            direction=direction,
            in_stair_map=in_map,
        )

    def test_stable_height_under_open_stairs_does_not_trigger(self):
        guard = self.make_guard()
        decisions = [self.update(guard, 0.0) for _ in range(10)]

        self.assertFalse(any(decision.confirmed for decision in decisions))
        self.assertEqual(decisions[-1].directional_change, 0.0)

    def test_upstairs_requires_map_dwell_and_rising_height(self):
        guard = self.make_guard()
        first = self.update(guard, 0.00, direction=1)
        second = self.update(guard, 0.03, direction=1)
        third = self.update(guard, 0.07, direction=1)

        self.assertFalse(first.confirmed)
        self.assertFalse(second.confirmed)
        self.assertTrue(third.confirmed)
        self.assertAlmostEqual(third.directional_change, 0.07)

    def test_downstairs_requires_falling_height(self):
        guard = self.make_guard()
        self.update(guard, 1.00, direction=2)
        self.update(guard, 0.97, direction=2)
        decision = self.update(guard, 0.93, direction=2)

        self.assertTrue(decision.confirmed)
        self.assertAlmostEqual(decision.directional_change, 0.07)

    def test_wrong_height_direction_does_not_trigger(self):
        guard = self.make_guard()
        decisions = [
            self.update(guard, height, direction=1)
            for height in (1.0, 0.96, 0.92, 0.88)
        ]

        self.assertFalse(any(decision.confirmed for decision in decisions))

    def test_leaving_map_resets_height_baseline(self):
        guard = self.make_guard()
        self.update(guard, 0.00)
        self.update(guard, 0.04)
        outside = self.update(guard, 0.04, in_map=False)
        reentered = self.update(guard, 0.04)

        self.assertEqual(outside.map_steps, 0)
        self.assertEqual(reentered.map_steps, 1)
        self.assertEqual(reentered.directional_change, 0.0)

    def test_missing_height_disables_passive_trigger(self):
        guard = self.make_guard()
        decision = self.update(guard, np.nan)

        self.assertFalse(decision.height_available)
        self.assertFalse(decision.confirmed)
        self.assertEqual(decision.map_steps, 0)


class CarrotHeightDirectionGuardTest(unittest.TestCase):
    def make_guard(self):
        return CarrotHeightDirectionGuard(
            num_envs=1,
            config=CarrotHeightDirectionConfig(
                probe_forward_steps=3,
                change_threshold=0.05,
                rotate_steps=2,
            ),
        )

    @staticmethod
    def update(
        guard,
        height,
        direction=1,
        forward=False,
        search_turn=False,
    ):
        return guard.update(
            env=0,
            height=height,
            direction=direction,
            previous_action_was_forward=forward,
            previous_action_was_search_turn=search_turn,
        )

    def test_requested_upward_height_change_confirms_carrot(self):
        guard = self.make_guard()
        self.update(guard, 0.0)
        collecting = self.update(guard, 0.02, forward=True)
        confirmed = self.update(guard, 0.06, forward=True)

        self.assertFalse(collecting.should_rotate)
        self.assertTrue(confirmed.direction_confirmed)
        self.assertFalse(confirmed.should_rotate)
        self.assertAlmostEqual(confirmed.signed_height_change, 0.06)

    def test_requested_downward_height_change_confirms_carrot(self):
        guard = self.make_guard()
        self.update(guard, 1.0, direction=2)
        confirmed = self.update(
            guard, 0.94, direction=2, forward=True
        )

        self.assertTrue(confirmed.direction_confirmed)
        self.assertAlmostEqual(confirmed.signed_height_change, 0.06)

    def test_flat_forward_probe_starts_turn_sweep_then_retries(self):
        guard = self.make_guard()
        self.update(guard, 0.0)
        self.update(guard, 0.0, forward=True)
        self.update(guard, 0.0, forward=True)
        rejected = self.update(guard, 0.0, forward=True)

        self.assertTrue(rejected.should_rotate)
        self.assertTrue(rejected.restart_carrot)
        self.assertIn("no requested height change", rejected.reason)

        rotating = self.update(guard, 0.0, search_turn=True)
        retry = self.update(guard, 0.0, search_turn=True)

        self.assertTrue(rotating.should_rotate)
        self.assertFalse(retry.should_rotate)
        self.assertTrue(retry.restart_carrot)
        self.assertEqual(retry.probe_forward_steps, 0)

    def test_opposite_height_change_rejects_carrot_immediately(self):
        guard = self.make_guard()
        self.update(guard, 0.5, direction=1)
        rejected = self.update(
            guard, 0.43, direction=1, forward=True
        )

        self.assertTrue(rejected.should_rotate)
        self.assertIn("opposite", rejected.reason)

    def test_missing_height_preserves_legacy_carrot(self):
        guard = self.make_guard()
        decision = self.update(guard, np.nan, forward=True)

        self.assertFalse(decision.height_available)
        self.assertFalse(decision.should_rotate)


if __name__ == "__main__":
    unittest.main()
