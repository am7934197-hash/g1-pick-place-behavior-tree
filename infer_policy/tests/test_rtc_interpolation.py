import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import robot_controller as robot_controller_module
from action_filter import BaseFilter
from joint_interpolator import LinearInterpolator
from robot_controller import RobotController


class _JointCommand:
    def __init__(self):
        self.position = 0.0
        self.velocity = 0.0
        self.effort = 0.0


class _Robot:
    def __init__(self):
        self.commands = []

    def set_joint_commands(self, commands, joint_groups, unused, time_from_start):
        del joint_groups, unused, time_from_start
        self.commands.append([command.position for command in commands])
        return "success"


class RTCInterpolationTest(unittest.TestCase):
    def test_model_action_dimension_must_match_config_exactly(self):
        controller = object.__new__(RobotController)
        controller.cfg = {"vla_action_dim": 16}
        state = np.arange(23, dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "expected exactly 16"):
            controller._expand_model_action(np.zeros(23, dtype=np.float32), state)

        expanded = controller._expand_model_action(
            np.ones(16, dtype=np.float32), state
        )
        np.testing.assert_array_equal(expanded[:16], np.ones(16, dtype=np.float32))
        np.testing.assert_array_equal(expanded[16:], state[16:])

    def test_single_rtc_action_is_spread_over_one_model_period(self):
        controller = object.__new__(RobotController)
        controller.cfg = {
            "vla_action_dim": 16,
            "control": {
                "interpolation": {
                    "enabled": True,
                    "method": "linear",
                    "input_hz": 30,
                    "output_hz": 250,
                },
                "max_joint_delta_per_step": 0.08,
                "gripper_max_delta_per_step": 20.0,
                "gripper_action_scale": 0.0012,
            },
        }
        controller.robot = _Robot()
        controller.action_filter = BaseFilter(dim=23)
        controller.interpolator = LinearInterpolator(
            dim=23, input_dt=1.0 / 30.0, output_dt=1.0 / 250.0,
        )
        controller.get_joint_state = lambda: np.zeros(23, dtype=np.float32)
        controller._build_execute_group_spec = lambda: (
            ["arms"], list(range(16)) + [-1] * 7, {7, 15}, 16,
        )
        controller._clip_to_joint_limits = lambda action: action
        controller._set_joint_time_from_start = lambda: 0.0
        controller._gripper_command_position = lambda value, scale: value * scale

        sleeps = []
        sdk = SimpleNamespace(ControlStatus=SimpleNamespace(SUCCESS="success", FAULT="fault"))
        target = np.full(16, 0.03, dtype=np.float32)

        with patch.object(robot_controller_module, "JointCommand", _JointCommand), \
                patch.object(robot_controller_module, "gm", sdk), \
                patch.object(
                    robot_controller_module.time, "sleep", side_effect=sleeps.append
                ):
            ok = controller.execute_action_chunk(
                [target], dt=1.0 / 30.0, skip_interp=False, sleep=True,
            )

        self.assertTrue(ok)
        self.assertEqual(len(controller.robot.commands), 10)
        self.assertEqual(len(sleeps), 9)
        self.assertAlmostEqual(sum(sleeps), 1.0 / 30.0, places=7)
        self.assertAlmostEqual(controller.robot.commands[0][0], 0.0, places=7)
        self.assertAlmostEqual(controller.robot.commands[-1][0], 0.03, places=7)


if __name__ == "__main__":
    unittest.main()
