import unittest

import numpy as np

from ascent.frontier_progress_guard import (
    ExplorationFrontierProgressGuard,
    FrontierProgressConfig,
)


class ExplorationFrontierProgressGuardTest(unittest.TestCase):
    def make_guard(self):
        return ExplorationFrontierProgressGuard(
            num_envs=1,
            config=FrontierProgressConfig(
                association_radius=0.6,
                blocked_radius=0.75,
                min_progress=0.15,
                max_stalled_selections=3,
            ),
        )

    @staticmethod
    def update(guard, robot_xy, frontier, floor_index=0):
        return guard.update(
            env=0,
            floor_index=floor_index,
            robot_xy=np.asarray(robot_xy, dtype=float),
            frontier=np.asarray(frontier, dtype=float),
        )

    def test_meaningful_progress_resets_stalled_count(self):
        guard = self.make_guard()
        self.update(guard, [0.0, 0.0], [3.0, 0.0])
        stalled = self.update(guard, [0.05, 0.0], [3.0, 0.0])
        progressed = self.update(guard, [0.25, 0.0], [3.0, 0.0])

        self.assertEqual(stalled.stalled_selections, 1)
        self.assertEqual(progressed.stalled_selections, 0)
        self.assertAlmostEqual(progressed.best_distance, 2.75)

    def test_static_unreachable_frontier_is_blocked(self):
        guard = self.make_guard()
        decisions = [self.update(guard, [0.0, 0.0], [3.0, 0.0])]
        decisions.extend(
            self.update(guard, [0.0, 0.0], [3.0, 0.0])
            for _ in range(3)
        )

        self.assertTrue(decisions[-1].unreachable)
        self.assertTrue(
            guard.is_blocked(env=0, floor_index=0, frontier=[3.0, 0.0])
        )

    def test_small_frontier_drift_is_one_progress_region(self):
        guard = self.make_guard()
        goals = ([3.0, 0.0], [3.1, 0.05], [2.9, -0.05], [3.2, 0.0])
        decisions = [
            self.update(guard, [0.0, 0.0], goal) for goal in goals
        ]

        self.assertTrue(decisions[-1].unreachable)
        self.assertTrue(
            guard.is_blocked(env=0, floor_index=0, frontier=[3.4, 0.0])
        )

    def test_block_filter_removes_region_but_keeps_other_frontiers(self):
        guard = self.make_guard()
        for _ in range(4):
            self.update(guard, [0.0, 0.0], [3.0, 0.0])

        filtered = guard.filter_blocked(
            env=0,
            floor_index=0,
            frontiers=np.array([[3.1, 0.1], [5.0, 0.0]]),
        )

        np.testing.assert_allclose(filtered, [[5.0, 0.0]])

    def test_blocked_regions_are_isolated_per_floor(self):
        guard = self.make_guard()
        for _ in range(4):
            self.update(guard, [0.0, 0.0], [3.0, 0.0], floor_index=0)

        self.assertTrue(
            guard.is_blocked(env=0, floor_index=0, frontier=[3.0, 0.0])
        )
        self.assertFalse(
            guard.is_blocked(env=0, floor_index=1, frontier=[3.0, 0.0])
        )


if __name__ == "__main__":
    unittest.main()
