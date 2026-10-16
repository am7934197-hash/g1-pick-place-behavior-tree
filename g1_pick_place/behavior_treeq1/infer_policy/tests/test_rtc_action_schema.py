import unittest

import numpy as np

from vla_client import _validate_rtc_previous_actions


class RTCActionSchemaTest(unittest.TestCase):
    def test_accepts_empty_and_16d_guidance(self):
        _validate_rtc_previous_actions([], 16)
        _validate_rtc_previous_actions(np.zeros((12, 16), dtype=np.float32), 16)

    def test_rejects_full_robot_targets_for_16d_policy(self):
        with self.assertRaisesRegex(
            ValueError, r"got \(12, 23\), expected \[T, 16\]"
        ):
            _validate_rtc_previous_actions(
                np.zeros((12, 23), dtype=np.float32), 16
            )


if __name__ == "__main__":
    unittest.main()
