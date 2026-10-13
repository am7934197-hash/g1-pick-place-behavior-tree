import unittest

import numpy as np

from robot_io import expand_model_action, gripper_command_position, linear_interpolate


class RobotExecuteTest(unittest.TestCase):
    def test_expand_keeps_leg_and_head_from_state(self):
        state = np.arange(23, dtype=np.float32)
        action = np.ones(16, dtype=np.float32)
        expanded = expand_model_action(action, state, 16)
        np.testing.assert_array_equal(expanded[:16], action)
        np.testing.assert_array_equal(expanded[16:], state[16:])

    def test_expand_rejects_wrong_action_dim(self):
        with self.assertRaisesRegex(ValueError, "expected 16"):
            expand_model_action(np.zeros(23), np.zeros(23), 16)

    def test_linear_interpolate_spreads_one_period(self):
        waypoints = np.array([[0.0] * 23, [0.03] * 23], dtype=np.float32)
        frames = linear_interpolate(waypoints, 1.0 / 30.0, 1.0 / 250.0)
        self.assertEqual(len(frames), 10)
        self.assertAlmostEqual(float(frames[0, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(frames[-1, 0]), 0.03, places=6)

    def test_gripper_command_scales_model_units_to_meters(self):
        self.assertAlmostEqual(gripper_command_position(100.0, 0.0012, 0.0, 0.12), 0.12)
        self.assertAlmostEqual(gripper_command_position(-1.0, 0.0012, 0.0, 0.12), 0.0)


if __name__ == "__main__":
    unittest.main()
