import unittest
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for relative_path in (
    "third_party/frontier_exploration",
    "third_party/depth_camera_filtering",
    "third_party/vlfm",
):
    sys.path.insert(0, str(PROJECT_ROOT / relative_path))

from ascent.habitat_visualizer import (
    draw_semantic_stair_frontiers_on_map,
    draw_semantic_stair_frontiers_on_rgb,
)


class SemanticStairFrontierMapTest(unittest.TestCase):
    def test_selected_frontier_is_drawn_on_top_down_map(self):
        rendered = np.zeros((100, 100, 3), dtype=np.uint8)
        info = {
            "top_down_map": {
                "map": np.zeros((100, 100), dtype=np.uint8),
                "upper_bound": np.array([10.0, 10.0]),
                "lower_bound": np.array([-10.0, -10.0]),
                "grid_resolution": np.array([100, 100]),
                "tf_episodic_to_global": np.eye(4),
            }
        }
        policy_info = {
            "semantic_stair_frontiers": {
                "points_xyz": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                "labels": ["A"],
                "valid": np.array([True]),
                "selected_label": "A",
                "selected_index": 0,
            }
        }

        result = draw_semantic_stair_frontiers_on_map(
            rendered, info, policy_info
        )

        self.assertEqual(result[50, 50].tolist(), [255, 0, 255])
        self.assertFalse(np.shares_memory(result, rendered))

    def test_selected_frontier_is_drawn_on_full_resolution_rgb(self):
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        policy_info = {
            "semantic_stair_frontiers": {
                "image_pixels": np.array([[160.0, 160.0]], dtype=np.float32),
                "marked_rgb": np.zeros((320, 320, 3), dtype=np.uint8),
                "labels": ["A"],
                "valid": np.array([True]),
                "selected_index": 0,
            }
        }

        result = draw_semantic_stair_frontiers_on_rgb(rgb, policy_info)

        # The inverse center crop places processed-image center near RGB center.
        self.assertTrue(np.all(result[240, 320] == np.array([255, 0, 255])))
        self.assertFalse(np.shares_memory(result, rgb))


if __name__ == "__main__":
    unittest.main()
