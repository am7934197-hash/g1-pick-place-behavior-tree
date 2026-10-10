import unittest

import numpy as np

from rtc_controller import RTCController


def actions(count, offset=0.0, dim=2):
    return [np.full(dim, offset + i, dtype=np.float32) for i in range(count)]


class RTCControllerTest(unittest.TestCase):
    def test_threshold_triggers_then_only_n_old_actions_are_executed(self):
        rtc = RTCController({"max_steps_per_observation": 10, "pre_infer_threshold": 45})
        rtc.add_chunk(actions(50))
        for _ in range(5):
            self.assertFalse(rtc.should_trigger_inference())
            self.assertIsNotNone(rtc.get_next_action())
        self.assertTrue(rtc.should_trigger_inference())

        rtc.mark_inference_started()
        capture = rtc.mark_observation_captured(rtc.session_id)
        self.assertEqual(capture["obs_capture_trajectory_step"], 5)
        self.assertEqual(len(capture["previous_actions"]), 45)

        for _ in range(10):
            self.assertIsNotNone(rtc.get_next_action())
        for _ in range(25):
            self.assertIsNone(rtc.get_next_action())
            rtc.record_hold_frame()
            self.assertFalse(rtc.should_trigger_inference())

        rtc.mark_inference_done(actions(50), capture)
        report, _ = rtc.consume_new_chunk()
        self.assertEqual(report["skipped"], 10)
        self.assertEqual(report["kept"], 40)

    def test_fast_result_releases_old_action_limit_immediately(self):
        rtc = RTCController({"max_steps_per_observation": 10, "pre_infer_threshold": 45})
        rtc.add_chunk(actions(50))
        for _ in range(5):
            rtc.get_next_action()
        rtc.mark_inference_started()
        capture = rtc.mark_observation_captured(rtc.session_id)
        rtc.get_next_action()
        rtc.get_next_action()
        rtc.mark_inference_done(actions(50, 100.0), capture)
        report, _ = rtc.consume_new_chunk()
        self.assertEqual(report["skipped"], 2)
        self.assertIsNotNone(rtc.get_next_action())

    def test_zero_limit_preserves_continuous_threshold_rtc(self):
        rtc = RTCController({"max_steps_per_observation": 0, "pre_infer_threshold": 45})
        rtc.add_chunk(actions(50))
        for _ in range(5):
            rtc.get_next_action()
        self.assertTrue(rtc.should_trigger_inference())
        rtc.mark_inference_started()
        rtc.mark_observation_captured(rtc.session_id)
        for _ in range(20):
            self.assertIsNotNone(rtc.get_next_action())

    def test_bounded_short_chunk_triggers_before_budget(self):
        rtc = RTCController({"max_steps_per_observation": 5})
        rtc.add_chunk(actions(2))
        rtc.get_next_action()
        rtc.get_next_action()
        self.assertTrue(rtc.should_trigger_inference())

    def setUp(self):
        self.rtc = RTCController({
            "enabled": True,
            "pre_infer_threshold": 3,
            "execution_horizon": 4,
            "max_guidance_weight": 10.0,
            "prefix_attention_schedule": "LINEAR",
        })

    def test_initial_chunk_is_played_in_order(self):
        report = self.rtc.add_chunk(actions(5))
        self.assertEqual(report["mode"], "replace_rtc")
        np.testing.assert_array_equal(self.rtc.get_next_action(), [0.0, 0.0])
        self.assertEqual(self.rtc.buffer_remaining(), 4)

    def test_threshold_triggers_only_one_pending_inference(self):
        self.rtc.add_chunk(actions(5))
        self.rtc.get_next_action()
        self.rtc.get_next_action()
        self.assertTrue(self.rtc.should_trigger_inference())
        self.rtc.mark_inference_started()
        self.assertFalse(self.rtc.should_trigger_inference())
        self.rtc.mark_inference_done(actions(5, 100.0))
        self.assertFalse(self.rtc.should_trigger_inference())

    def test_inference_latency_skips_elapsed_new_frames(self):
        self.rtc.add_chunk(actions(10))
        for _ in range(7):
            self.rtc.get_next_action()
        self.rtc.mark_inference_started()
        capture = self.rtc.mark_observation_captured(self.rtc.session_id)
        for _ in range(2):
            self.rtc.get_next_action()
        self.rtc.mark_inference_done(actions(10, 100.0), capture)
        report, _ = self.rtc.consume_new_chunk()
        self.assertEqual(report["skipped"], 2)
        self.assertEqual(report["kept"], 8)
        self.assertEqual(self.rtc.buffer_remaining(), 8)

    def test_new_chunk_replaces_old_tail_without_posthoc_interpolation(self):
        old = [np.zeros(2, dtype=np.float32) for _ in range(8)]
        new = [np.full(2, 100.0, dtype=np.float32) for _ in range(8)]
        self.rtc.add_chunk(old)
        self.rtc.get_next_action()
        self.rtc.mark_inference_started()
        self.rtc.mark_inference_done(new)
        self.rtc.consume_new_chunk()
        values = [float(self.rtc.get_next_action()[0]) for _ in range(4)]
        np.testing.assert_allclose(values, [100.0, 100.0, 100.0, 100.0])

    def test_observation_snapshot_contains_unexecuted_prefix(self):
        self.rtc.add_chunk(actions(6))
        self.rtc.get_next_action()
        self.rtc.get_next_action()
        self.rtc.mark_inference_started()
        snapshot = self.rtc.mark_observation_captured(self.rtc.session_id)
        self.assertEqual(snapshot["previous_action_start_index"], 2)
        self.assertEqual(len(snapshot["previous_actions"]), 4)
        np.testing.assert_array_equal(snapshot["previous_actions"][0], [2.0, 2.0])

    def test_empty_result_keeps_old_tail(self):
        self.rtc.add_chunk(actions(6))
        self.rtc.get_next_action()
        self.rtc.mark_inference_started()
        self.rtc.mark_inference_done([])
        report, _ = self.rtc.consume_new_chunk()
        self.assertEqual(report["mode"], "empty")
        self.assertEqual(self.rtc.buffer_remaining(), 5)

    def test_short_new_chunk_is_safe_to_install(self):
        self.rtc.add_chunk(actions(10))
        self.rtc.get_next_action()
        self.rtc.mark_inference_started()
        self.rtc.mark_inference_done(actions(2, 100.0))
        report, _ = self.rtc.consume_new_chunk()
        self.assertEqual(report["kept"], 2)
        self.assertEqual(self.rtc.buffer_remaining(), 2)

    def test_inconsistent_frame_dimensions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "action dim mismatch"):
            self.rtc.add_chunk([np.zeros(2), np.zeros(3)])

    def test_reset_clears_state(self):
        self.rtc.add_chunk(actions(5))
        self.rtc.get_next_action()
        self.rtc.reset()
        self.assertEqual(self.rtc.buffer_remaining(), 0)
        self.assertFalse(self.rtc.infer_in_progress)


if __name__ == "__main__":
    unittest.main()
