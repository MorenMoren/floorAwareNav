import sys
import unittest

import numpy as np

# ascent.run installs these project-local dependencies on sys.path before it
# imports the trainer.  Mirror that setup for a standalone unit-test process.
sys.path[:0] = [
    ".",
    "third_party/frontier_exploration",
    "third_party/depth_camera_filtering",
    "third_party/vlfm",
]

from ascent.ascent_trainer import extract_scalars_from_info


class ExtractScalarsFromInfoTest(unittest.TestCase):
    def test_ignores_nested_none_and_lists(self):
        info = {
            "success": 1,
            "custom_metric": {
                "score": np.float32(0.75),
                "optional_value": None,
                "debug_points": [[1, 2], [3, 4]],
            },
        }

        self.assertEqual(
            extract_scalars_from_info(info),
            {"success": 1.0, "custom_metric.score": 0.75},
        )

    def test_preserves_habitat_non_scalar_filtering(self):
        info = {
            "top_down_map": {"map": np.zeros((2, 2), dtype=np.uint8)},
            "collisions": {"count": 2, "is_collision": False},
        }

        self.assertEqual(
            extract_scalars_from_info(info), {"collisions.count": 2.0}
        )


if __name__ == "__main__":
    unittest.main()
