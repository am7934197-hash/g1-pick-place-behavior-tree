import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

import server


def _chunk():
    return [{"action": np.zeros(16, dtype=np.float32)} for _ in range(50)]


class _DelayedVLA:
    def __init__(self, delay_seconds):
        self.delay_seconds = delay_seconds
        self.calls = 0
        self.timesteps = []
        self.contexts = []
        self.lock = threading.Lock()

    def infer(self, obs, instruction, timestep, rtc_context=None):
        del obs, instruction
        with self.lock:
            self.calls += 1
            self.timesteps.append(timestep)
            self.contexts.append(rtc_context)
            call = self.calls
        if call > 1:
            time.sleep(self.delay_seconds)
        return _chunk()


class _Robot:
    def __init__(self):
        self.execute_calls = 0
        self.project_calls = 0

    def reset_action_filter(self):
        pass

    def execute_action_chunk(self, actions, **kwargs):
        del actions, kwargs
        self.execute_calls += 1
        return True

    def project_model_action_chunk(self, actions, current_state):
        del current_state
        self.project_calls += 1
        # Match RobotController's real behavior: safety projection happens in
        # the full 23-D robot space, but RTC guidance sent to the model is the
        # 16-D policy prefix.
        projected = []
        for action in actions:
            full_action = np.zeros(23, dtype=np.float32)
            full_action[:16] = np.asarray(action, dtype=np.float32) + 0.25
            projected.append(full_action[:16].copy())
        return projected


class RTCUnderflowTest(unittest.TestCase):
    def _run(self, infer_delay, max_steps=55, max_steps_per_observation=0,
             pre_infer_threshold=10):
        fake_vla = _DelayedVLA(infer_delay)
        fake_robot = _Robot()
        rtc_cfg = {
            "enabled": True,
            "pre_infer_threshold": pre_infer_threshold,
            "execution_horizon": 10,
            "max_guidance_weight": 10.0,
            "prefix_attention_schedule": "LINEAR",
            "max_steps_per_observation": max_steps_per_observation,
        }
        control_cfg = {
            "fps": 100,
            "dt": 0.01,
            "interpolation": {"output_hz": 250},
        }
        server._stop_event.clear()
        with patch.object(server, "vla", fake_vla), \
                patch.object(server, "robot", fake_robot), \
                patch.dict(server.ROBOT_CFG, {"control": control_cfg}, clear=True), \
                patch.object(
                    server,
                    "_capture_and_preprocess",
                    return_value=(
                        {
                            "head_right": object(),
                            **{name: 0.0 for name in server.STATE_NAMES},
                        },
                        {"obs_capture_ts": time.time()},
                    ),
                ), \
                patch.object(server, "validate_observation", return_value=True), \
                patch.object(server.inference_logger, "start_run"), \
                patch.object(server.inference_logger, "end_run"), \
                patch.object(server.inference_logger, "log_obs"), \
                patch.object(server, "log_inferred_actions"):
            result = server._run_vla_rtc(
                "place object", max_steps=max_steps, rtc_cfg=rtc_cfg
            )
        return result, fake_robot, fake_vla

    def test_bounded_rtc_prefetches_then_holds_after_five_old_actions(self):
        result, robot, vla = self._run(
            0.1,
            max_steps=16,
            max_steps_per_observation=5,
            pre_infer_threshold=45,
        )
        self.assertTrue(result["success"])
        self.assertEqual(vla.timesteps[:2], [0, 5])
        self.assertTrue(all(
            later - earlier <= 6
            for earlier, later in zip(vla.timesteps[1:], vla.timesteps[2:])
        ))
        self.assertGreater(robot.execute_calls, 16)

    def test_chunk_arriving_just_after_empty_recovers_while_holding(self):
        # Ten buffered frames provide 100 ms; the result arrives about 50 ms
        # after exhaustion and must be installed instead of failing the task.
        result, robot, vla = self._run(0.15)

        self.assertTrue(result["success"])
        self.assertEqual(result["details"]["steps"], 55)
        self.assertGreater(robot.execute_calls, 55)  # includes hold cycles
        self.assertGreaterEqual(vla.calls, 2)

    def test_async_guidance_uses_locally_projected_previous_actions(self):
        result, robot, vla = self._run(0.01, max_steps=16, pre_infer_threshold=45)

        self.assertTrue(result["success"])
        self.assertGreater(robot.project_calls, 0)
        guided_contexts = [
            context for context in vla.contexts[1:]
            if context and context.get("previous_actions")
        ]
        self.assertTrue(guided_contexts)
        self.assertTrue(all(
            np.asarray(action).shape == (16,)
            for context in guided_contexts
            for action in context["previous_actions"]
        ))
        np.testing.assert_allclose(
            guided_contexts[0]["previous_actions"][0],
            np.full(16, 0.25, dtype=np.float32),
        )

    def test_inference_past_underflow_timeout_fails_safely(self):
        result, robot, _ = self._run(0.5)

        self.assertFalse(result["success"])
        self.assertEqual(result["message"], "RTC action buffer underflow timeout")
        self.assertGreater(robot.execute_calls, 50)  # last target was held


if __name__ == "__main__":
    unittest.main()
