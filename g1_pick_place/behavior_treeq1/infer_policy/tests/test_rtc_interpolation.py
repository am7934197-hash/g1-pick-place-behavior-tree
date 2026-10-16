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
    def test_runtime_profile_can_switch_per_joint_limits(self):
        controller = object.__new__(RobotController)
        controller.cfg = {
            "control": {
                "max_joint_delta_per_step": 0.05,
            },
        }
        pick = {
            "default": 0.05,
            "right_arm": [0.07, 0.035, 0.075, 0.065, 0.07, 0.095, 0.115],
            "left_arm": [0.035, 0.03, 0.045, 0.06, 0.08, 0.085, 0.12],
        }
        place = {
            "default": 0.05,
            "right_arm": [0.03, 0.02, 0.04, 0.04, 0.055, 0.065, 0.05],
            "left_arm": [0.03, 0.03, 0.04, 0.04, 0.065, 0.07, 0.055],
        }

        controller.set_joint_max_delta_per_step(pick, profile_name="model_at_a")
        np.testing.assert_allclose(
            controller._joint_max_delta_per_step(0.3)[0:7], pick["right_arm"]
        )
        controller.set_joint_max_delta_per_step(place, profile_name="model_at_b")
        np.testing.assert_allclose(
            controller._joint_max_delta_per_step(0.3)[8:15], place["left_arm"]
        )
        controller.set_joint_max_delta_per_step(None)
        np.testing.assert_allclose(
            controller._joint_max_delta_per_step(0.3), np.full(23, 0.05)
        )

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

    def test_per_joint_limits_project_the_same_safe_rtc_prefix(self):
        controller = object.__new__(RobotController)
        controller.cfg = {
            "vla_action_dim": 16,
            "control": {
                "max_joint_delta_per_step": {
                    "default": 0.05,
                    "right_arm": [0.06, 0.03, 0.06, 0.06, 0.06, 0.08, 0.10],
                    "left_arm": [0.03, 0.03, 0.04, 0.05, 0.07, 0.07, 0.10],
                },
                "gripper_max_delta_per_step": 15.0,
            },
        }
        controller.action_filter = BaseFilter(dim=23)
        controller._clip_to_joint_limits = lambda action: action

        target = np.full(16, 100.0, dtype=np.float32)
        projected = controller.project_action_chunk(
            [target], np.zeros(23, dtype=np.float32)
        )[0]

        expected = np.zeros(23, dtype=np.float32)
        expected[:16] = [
            0.06, 0.03, 0.06, 0.06, 0.06, 0.08, 0.10, 15.0,
            0.03, 0.03, 0.04, 0.05, 0.07, 0.07, 0.10, 15.0,
        ]
        np.testing.assert_allclose(projected, expected, atol=1e-7)

    def test_rtc_model_projection_returns_only_policy_dimensions(self):
        controller = object.__new__(RobotController)
        controller.cfg = {
            "vla_action_dim": 16,
            "control": {
                "max_joint_delta_per_step": 0.05,
                "gripper_max_delta_per_step": 15.0,
            },
        }
        controller.action_filter = BaseFilter(dim=23)
        controller._clip_to_joint_limits = lambda action: action

        projected = controller.project_model_action_chunk(
            [np.full(16, 1.0, dtype=np.float32)],
            np.zeros(23, dtype=np.float32),
        )

        self.assertEqual(len(projected), 1)
        self.assertEqual(projected[0].shape, (16,))
        np.testing.assert_allclose(projected[0], [
            0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 1.0,
            0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 1.0,
        ], atol=1e-7)

    def test_per_joint_limits_are_used_for_sdk_dispatch(self):
        controller = object.__new__(RobotController)
        controller.cfg = {
            "vla_action_dim": 16,
            "control": {
                "interpolation": {"enabled": False},
                "max_joint_delta_per_step": {
                    "default": 0.05,
                    "right_arm": [0.06, 0.03, 0.06, 0.06, 0.06, 0.08, 0.10],
                    "left_arm": [0.03, 0.03, 0.04, 0.05, 0.07, 0.07, 0.10],
                },
                "gripper_max_delta_per_step": 15.0,
                "gripper_action_scale": 1.0,
            },
        }
        controller.robot = _Robot()
        controller.action_filter = BaseFilter(dim=23)
        controller.get_joint_state = lambda: np.zeros(23, dtype=np.float32)
        controller._build_execute_group_spec = lambda: (
            ["arms"], list(range(16)) + [-1] * 7, {7, 15}, 16,
        )
        controller._clip_to_joint_limits = lambda action: action
        controller._set_joint_time_from_start = lambda: 0.0
        controller._gripper_command_position = lambda value, scale: value * scale
        sdk = SimpleNamespace(ControlStatus=SimpleNamespace(SUCCESS="success", FAULT="fault"))

        with patch.object(robot_controller_module, "JointCommand", _JointCommand), \
                patch.object(robot_controller_module, "gm", sdk):
            ok = controller.execute_action_chunk(
                [np.full(16, 100.0, dtype=np.float32)],
                dt=1.0 / 30.0,
                skip_interp=True,
                sleep=False,
            )

        self.assertTrue(ok)
        np.testing.assert_allclose(controller.robot.commands[-1], [
            0.06, 0.03, 0.06, 0.06, 0.06, 0.08, 0.10, 15.0,
            0.03, 0.03, 0.04, 0.05, 0.07, 0.07, 0.10, 15.0,
        ], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
