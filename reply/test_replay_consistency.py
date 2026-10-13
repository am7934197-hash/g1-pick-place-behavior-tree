"""Unit tests for replay elbow-descent and tracking RMSE helpers."""

import math
import unittest

from replay_dataset import (
    LEFT_DESCENT_LIMIT,
    RIGHT_DESCENT_LIMIT,
    commanded_elbow_reports,
    elbow_report,
    flatten_joint_positions,
    gripper_state_to_vla,
    status_kind,
    tracking_rmse,
)


class DummyState:
    def __init__(self, position):
        self.position = position


class ElbowReportTest(unittest.TestCase):
    def test_right_descends_toward_limit(self):
        report = elbow_report([1.48, 1.40, 1.20, 1.18], right=True)
        self.assertTrue(report["descended"])
        self.assertTrue(report["reached_training_descent_limit"])
        self.assertAlmostEqual(report["descent_from_start"], 0.30, places=2)
        self.assertEqual(report["training_descent_limit"], RIGHT_DESCENT_LIMIT)

    def test_right_hover_does_not_count_as_descent(self):
        report = elbow_report([1.415, 1.45, 1.70, 1.69], right=True)
        self.assertFalse(report["descended"])
        self.assertFalse(report["reached_training_descent_limit"])

    def test_left_descends_toward_limit(self):
        report = elbow_report([-1.62, -1.50, -1.30, -1.25], right=False)
        self.assertTrue(report["descended"])
        self.assertTrue(report["reached_training_descent_limit"])
        self.assertEqual(report["training_descent_limit"], LEFT_DESCENT_LIMIT)


class TrackingRmseTest(unittest.TestCase):
    def test_rmse_zero_when_perfect(self):
        sample = [0.0] * 16
        result = tracking_rmse([sample, sample])
        self.assertEqual(result["n_samples"], 2)
        self.assertAlmostEqual(result["rmse_mean"], 0.0)
        self.assertAlmostEqual(result["rmse_max"], 0.0)

    def test_rmse_on_one_joint(self):
        err = [0.0] * 16
        err[3] = 0.2
        result = tracking_rmse([err])
        self.assertAlmostEqual(result["by_joint"]["right_arm_joint4"], 0.2)
        self.assertAlmostEqual(result["rmse_mean"], math.sqrt((0.2 ** 2) / 16))


class GripperAndStateTest(unittest.TestCase):
    def test_gripper_meters_to_vla(self):
        self.assertAlmostEqual(gripper_state_to_vla(0.0), 0.0)
        self.assertAlmostEqual(gripper_state_to_vla(0.12), 100.0)
        self.assertAlmostEqual(gripper_state_to_vla(50.0), 50.0)

    def test_flatten_16_dim_states(self):
        states = [
            DummyState([0.1] * 7),
            DummyState(0.06),
            DummyState([0.2] * 7),
            DummyState(0.0),
        ]
        values = flatten_joint_positions(states)
        self.assertEqual(len(values), 16)
        self.assertAlmostEqual(values[7], 50.0)
        self.assertAlmostEqual(values[15], 0.0)

    def test_status_kind_fault(self):
        class ControlStatus:
            SUCCESS = "SUCCESS"
            FAULT = "FAULT"

        self.assertEqual(status_kind(ControlStatus.SUCCESS, ControlStatus), "SUCCESS")
        self.assertEqual(status_kind(ControlStatus.FAULT, ControlStatus), "FAULT")


class CommandedElbowTest(unittest.TestCase):
    def test_from_clipped_rows(self):
        rows = [[0.0] * 23, [0.0] * 23]
        rows[0][3] = 1.48
        rows[1][3] = 1.18
        rows[0][11] = -1.62
        rows[1][11] = -1.25
        report = commanded_elbow_reports(rows)
        self.assertTrue(report["right_arm_joint4"]["descended"])
        self.assertTrue(report["left_arm_joint4"]["descended"])


if __name__ == "__main__":
    unittest.main()
