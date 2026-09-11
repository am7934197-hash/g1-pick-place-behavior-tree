import unittest
from unittest.mock import patch

import head_capture_switch as hcs


class EnsureModeTest(unittest.TestCase):
    def test_skips_restart_when_already_on_data_collection(self):
        with patch.object(hcs, "current_head_mode", return_value="data_collection"), \
             patch.object(hcs, "_pids", return_value=[42]), \
             patch.object(hcs, "ensure_arm_captures") as arms, \
             patch.object(hcs, "switch_to") as switch:
            pid = hcs.ensure_mode("data_collection")
        self.assertEqual(pid, 42)
        arms.assert_called_once()
        switch.assert_not_called()

    def test_switches_when_currently_working(self):
        with patch.object(hcs, "current_head_mode", return_value="working"), \
             patch.object(hcs, "_pids", return_value=[7]), \
             patch.object(hcs, "switch_to", return_value=99) as switch, \
             patch.object(hcs, "time") as fake_time:
            pid = hcs.ensure_mode("data_collection", settle_s=1.5)
        self.assertEqual(pid, 99)
        switch.assert_called_once_with("data_collection")
        fake_time.sleep.assert_called_once_with(1.5)
