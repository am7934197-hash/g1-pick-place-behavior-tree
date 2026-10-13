import unittest
import pickle
import time

import numpy as np

from alignment_schema import make_observation_features
from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import send_bytes_in_chunks
from vla_client import (
    VLAClient,
    _get_actions,
    _send_observations,
    summarize_infer_rpc_samples,
)


class _FakeTensor:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self.values, dtype=np.float32)


class _FakeCall:
    def trailing_metadata(self):
        return (
            ("x-lerobot-trace-id", "trace-1"),
            ("x-lerobot-server-queue-ms", "3.5"),
            ("x-lerobot-server-model-ms", "291.0"),
            ("x-lerobot-server-get-actions-ms", "300.0"),
        )


class _FakeGetActions:
    def with_call(self, request, timeout=None, metadata=None):
        del request, timeout, metadata
        from lerobot.async_inference.helpers import TimedAction

        payload = pickle.dumps([TimedAction(1.0, 2, _FakeTensor([1.0, 2.0]))])
        return services_pb2.Actions(data=payload), _FakeCall()


class _FakeRTCGetActions:
    def with_call(self, request, timeout=None, metadata=None):
        del request, timeout, metadata
        from lerobot.async_inference.helpers import TimedAction

        payload = pickle.dumps({
            "trace_id": "trace-1",
            "observation_timestep": 2,
            "inference_delay_frames": 9,
            "prefix_frames_used": 10,
            "actions": [TimedAction(1.0, 2, _FakeTensor([1.0, 2.0]))],
        })
        return services_pb2.Actions(data=payload), _FakeCall()


class _FakeStub:
    def __init__(self):
        self.GetActions = _FakeGetActions()

    def SendObservations(self, request_iterator, metadata=None):
        self.metadata = metadata
        self.messages = list(request_iterator)
        time.sleep(0.01)
        return services_pb2.Empty()


class InferRpcTimingSplitTest(unittest.TestCase):
    def test_observation_rpc_splits_iterator_from_ack_wait(self):
        stub = _FakeStub()
        observation = TimedObservation(1.0, 2, {"joint": 0.5}, True)
        detail = _send_observations(stub, observation, "trace-1")

        self.assertEqual(stub.metadata, (("x-lerobot-trace-id", "trace-1"),))
        self.assertEqual(len(stub.messages), 1)
        self.assertGreaterEqual(detail["send_rpc_ms"], 9.0)
        self.assertGreaterEqual(detail["send_ack_wait_ms"], 9.0)
        self.assertLess(detail["request_iterator_ms"], detail["send_rpc_ms"])

    def test_get_actions_reads_optional_server_timing_metadata(self):
        actions, detail = _get_actions(_FakeStub(), timeout=1.0, trace_id="trace-1")
        self.assertEqual(actions[0]["action"], [1.0, 2.0])
        self.assertEqual(detail["server_timing"]["server_queue_ms"], 3.5)
        self.assertEqual(detail["server_timing"]["server_model_infer_ms"], 291.0)
        self.assertEqual(detail["server_timing"]["server_get_actions_ms"], 300.0)

    def test_get_actions_decodes_rtc_response_envelope(self):
        stub = _FakeStub()
        stub.GetActions = _FakeRTCGetActions()
        actions, detail = _get_actions(stub, timeout=1.0, trace_id="trace-1")
        self.assertEqual(actions[0]["action"], [1.0, 2.0])
        self.assertEqual(detail["response_meta"]["trace_id"], "trace-1")
        self.assertEqual(detail["response_meta"]["observation_timestep"], 2)
        self.assertEqual(detail["response_meta"]["inference_delay_frames"], 9)
        self.assertEqual(detail["response_meta"]["prefix_frames_used"], 10)

    def test_uses_lerobot_wire_types_and_generated_stub(self):
        observation = TimedObservation(1.0, 2, {"joint": 0.5}, True)
        restored = pickle.loads(pickle.dumps(observation))
        self.assertEqual(type(restored).__module__, "lerobot.async_inference.helpers")
        self.assertEqual(restored.get_timestep(), 2)

        messages = list(send_bytes_in_chunks(b"payload", services_pb2.Observation))
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].transfer_state, services_pb2.TRANSFER_END)
        self.assertEqual(messages[0].data, b"payload")
        self.assertTrue(hasattr(services_pb2_grpc, "AsyncInferenceStub"))

        config = RemotePolicyConfig("pi0", "/checkpoint", {}, 50, "cuda")
        self.assertEqual(config.policy_type, "pi0")

    def test_spikes_are_counted_as_network_not_compute(self):
        samples = [{"send_ms": 130.0, "wait_ms": 200.0, "total_ms": 340.0}] * 20
        samples.append({"send_ms": 140.0, "wait_ms": 8000.0, "total_ms": 8200.0})
        summary = summarize_infer_rpc_samples(samples, rtt_ms=20.0)

        self.assertEqual(summary["count"], 21)
        self.assertAlmostEqual(summary["median_wait_ms"], 200.0)
        # Typical GPU wait minus RTT, including the spiked call capped at 3x median.
        self.assertAlmostEqual(summary["compute_ms"], 21 * (200.0 - 20.0), places=1)
        self.assertGreater(summary["network_ms"], 8000.0 - 200.0)
        self.assertGreater(summary["wait_spike_ms"], 7000.0)

    def test_uniform_fast_waits_have_no_spike(self):
        samples = [{"send_ms": 120.0, "wait_ms": 190.0, "total_ms": 320.0}] * 10
        summary = summarize_infer_rpc_samples(samples, rtt_ms=15.0)
        self.assertEqual(summary["wait_spike_ms"], 0.0)
        self.assertAlmostEqual(summary["compute_ms"], 10 * (190.0 - 15.0), places=1)
        self.assertAlmostEqual(summary["network_ms"], 10 * 120.0 + 10 * 15.0, places=1)

    def test_rpc_summary_keeps_detailed_client_and_server_totals(self):
        samples = [
            {
                "send_ms": 120.0,
                "wait_ms": 190.0,
                "send_rpc_ms": 120.0,
                "serialize_ms": 12.0,
                "send_ack_wait_ms": 80.0,
                "get_actions_rpc_ms": 190.0,
                "response_decode_ms": 3.0,
                "payload_bytes": 1024,
                "server_timing": {
                    "server_queue_ms": 4.0,
                    "server_model_infer_ms": 150.0,
                },
            },
            {
                "send_ms": 130.0,
                "wait_ms": 210.0,
                "send_rpc_ms": 130.0,
                "serialize_ms": 13.0,
                "send_ack_wait_ms": 90.0,
                "get_actions_rpc_ms": 210.0,
                "response_decode_ms": 4.0,
                "payload_bytes": 2048,
                "server_timing": {
                    "server_queue_ms": 5.0,
                    "server_model_infer_ms": 160.0,
                },
            },
        ]
        detail = summarize_infer_rpc_samples(samples, rtt_ms=15.0)["rpc_detail"]
        self.assertEqual(detail["payload_bytes"], 3072)
        self.assertEqual(detail["send_rpc_ms"], 250.0)
        self.assertEqual(detail["send_ack_wait_ms"], 170.0)
        self.assertEqual(detail["get_actions_rpc_ms"], 400.0)
        self.assertEqual(detail["server_queue_ms"], 9.0)
        self.assertEqual(detail["server_model_infer_ms"], 310.0)

    def test_transport_drops_camera_not_in_training_schema(self):
        client = VLAClient({
            "observation_features": make_observation_features(
                ["head_right", "left_arm", "right_arm"]
            ),
            "observation_transport": {"image_encoding": "png", "png_compress_level": 1},
        })
        image = np.zeros((8, 8, 3), dtype=np.uint8)

        packed = client._encode_images_for_transport({
            "head_left": image,
            "head_right": image,
            "left_arm": image,
            "right_arm": image,
            "right_arm_joint1": 0.0,
        })

        self.assertNotIn("head_left", packed)
        self.assertEqual(
            {name for name in ("head_right", "left_arm", "right_arm") if name in packed},
            {"head_right", "left_arm", "right_arm"},
        )
        self.assertEqual(packed["head_right"]["__lerobot_image_encoding__"], "png")


if __name__ == "__main__":
    unittest.main()
