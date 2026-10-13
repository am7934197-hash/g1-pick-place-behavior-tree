import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

# The lightweight CI environment used for these unit tests does not install the
# HTTP runtime.  Stub only the decorators/types needed while importing the module.
try:
    import fastapi  # noqa: F401
except ModuleNotFoundError:
    fastapi_stub = types.ModuleType("fastapi")

    class FakeFastAPI:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, *args, **kwargs):
            return lambda function: function

        def post(self, *args, **kwargs):
            return lambda function: function

    fastapi_stub.FastAPI = FakeFastAPI
    fastapi_stub.HTTPException = RuntimeError
    sys.modules["fastapi"] = fastapi_stub

try:
    import pydantic  # noqa: F401
except ModuleNotFoundError:
    pydantic_stub = types.ModuleType("pydantic")
    pydantic_stub.BaseModel = object
    pydantic_stub.Field = lambda default=None, **kwargs: default
    sys.modules["pydantic"] = pydantic_stub

import infer_server


class FakeVLAClient:
    instances = []

    def __init__(self, config):
        self.config = config
        self.connected = False
        self._ready = False
        self.connect_calls = 0
        self.setup_calls = 0
        self.disconnect_calls = 0
        self.__class__.instances.append(self)

    @property
    def policy_ready(self):
        return self.connected and self._ready

    def connect(self):
        self.connect_calls += 1
        self.connected = True
        return True

    def setup_policy(self):
        self.setup_calls += 1
        self._ready = True
        return True

    def disconnect(self):
        self.disconnect_calls += 1
        self.connected = False
        self._ready = False


class InferServerPreloadTest(unittest.TestCase):
    def setUp(self):
        FakeVLAClient.instances = []
        behavior_dir = Path(infer_server.__file__).resolve().parent
        infer_server.BASE_VLA_CONFIG = infer_server.load_yaml(behavior_dir / "config.yaml")
        infer_server.SERVER_CONFIG = infer_server.load_yaml(behavior_dir / "config.yaml")
        infer_server.SERVER_CONFIG["vla_profiles"]["model_at_b"] = {
            **infer_server.SERVER_CONFIG["vla_profiles"]["model_at_a"],
            "cameras": ["head_left", "head_right", "left_arm", "right_arm"],
        }
        infer_server.SERVER_CONFIG["vla_runtime"] = {"mode": "multi_model"}
        infer_server.SERVER_CONFIG["vla_profiles"]["model_at_a"]["policy_server_address"] = "localhost:1"
        infer_server.SERVER_CONFIG["vla_profiles"]["model_at_b"]["policy_server_address"] = "localhost:2"
        infer_server.REFERENCE = SimpleNamespace(
            VLAClient=FakeVLAClient,
            ImagePreprocessor=lambda config: ("preprocessor", config),
            vla=None,
            CFG=None,
            VLA_CFG=None,
            image_preprocessor=None,
            robot=SimpleNamespace(open_grippers=lambda: True),
        )
        infer_server.PROFILE_RUNTIMES = {}
        infer_server.WARMED_RUNTIME_PROFILES = set()
        infer_server.ACTIVE_PROFILE = None

    def tearDown(self):
        for runtime in infer_server.PROFILE_RUNTIMES.values():
            runtime.client.disconnect()
        infer_server.PROFILE_RUNTIMES = {}
        infer_server.WARMED_RUNTIME_PROFILES = set()
        infer_server.ACTIVE_PROFILE = None

    def test_preloads_each_profile_once_and_switch_does_not_reload(self):
        infer_server.preload_profiles()

        self.assertEqual(set(infer_server.PROFILE_RUNTIMES), {"model_at_a", "model_at_b"})
        self.assertEqual(len(FakeVLAClient.instances), 2)
        self.assertTrue(all(client.connect_calls == 1 for client in FakeVLAClient.instances))
        self.assertTrue(all(client.setup_calls == 1 for client in FakeVLAClient.instances))

        infer_server.activate_profile("model_at_a")
        infer_server.activate_profile("model_at_b")
        infer_server.activate_profile("model_at_a")

        self.assertEqual(sum(client.setup_calls for client in FakeVLAClient.instances), 2)
        self.assertTrue(all(client.disconnect_calls == 0 for client in FakeVLAClient.instances))
        self.assertEqual(infer_server.ACTIVE_PROFILE, "model_at_a")
        for runtime in infer_server.PROFILE_RUNTIMES.values():
            self.assertGreaterEqual(runtime.load_ms, 0.0)

    def test_rejects_010000_profile(self):
        infer_server.SERVER_CONFIG["vla_profiles"]["model_at_a"]["pretrained_name_or_path"] = (
            "/home/zhangyuqi/zhangyuqi/G1-20260821/g1/pickandplace-top1/"
            "stage2_batch64/checkpoints/010000/pretrained_model"
        )
        with self.assertRaisesRegex(ValueError, "010000"):
            infer_server.preload_profiles()

    def test_rejects_profiles_sharing_one_policy_server(self):
        infer_server.SERVER_CONFIG["vla_profiles"]["model_at_b"]["policy_server_address"] = "localhost:1"
        with self.assertRaisesRegex(ValueError, "distinct port"):
            infer_server.preload_profiles()

    def test_single_model_loads_shared_profile_once_and_routes_b_to_it(self):
        infer_server.SERVER_CONFIG["vla_runtime"] = {
            "mode": "single_model",
            "shared_profile": "model_at_a",
        }
        infer_server.SERVER_CONFIG["vla_profiles"]["model_at_b"]["policy_server_address"] = (
            "localhost:1"
        )

        infer_server.preload_profiles()
        infer_server.activate_profile("model_at_a")
        infer_server.activate_profile("model_at_b")

        self.assertEqual(set(infer_server.PROFILE_RUNTIMES), {"model_at_a"})
        self.assertEqual(len(FakeVLAClient.instances), 1)
        self.assertEqual(FakeVLAClient.instances[0].setup_calls, 1)
        self.assertIs(infer_server.REFERENCE.vla, FakeVLAClient.instances[0])
        self.assertEqual(infer_server.ACTIVE_PROFILE, "model_at_b")

    def test_single_model_executes_distinct_tasks_on_the_shared_client(self):
        infer_server.SERVER_CONFIG["vla_runtime"] = {
            "mode": "single_model",
            "shared_profile": "model_at_a",
        }
        tasks = []
        infer_server.REFERENCE.run_vla = lambda instruction, max_steps: (
            tasks.append((instruction, max_steps, infer_server.REFERENCE.CFG["rtc"]["warmup_inferences"]))
            or {
                "success": True,
                "details": {"warmup": {"success": True, "warmup_inferences": 1}},
            }
        )
        infer_server.preload_profiles()

        pick_result = infer_server.execute(SimpleNamespace(
            profile="model_at_a", instruction="pick task", max_steps=12,
        ))
        place_result = infer_server.execute(SimpleNamespace(
            profile="model_at_b", instruction="place task", max_steps=34,
        ))

        self.assertEqual(len(FakeVLAClient.instances), 1)
        self.assertEqual(tasks, [("pick task", 12, 1), ("place task", 34, 0)])
        self.assertEqual(infer_server.WARMED_RUNTIME_PROFILES, {"model_at_a"})
        self.assertEqual(pick_result["timing"]["runtime_profile"], "model_at_a")
        self.assertEqual(place_result["timing"]["runtime_profile"], "model_at_a")

    def test_only_pick_opens_both_grippers_after_max_steps(self):
        opened = []
        infer_server.SERVER_CONFIG["only_pick"] = True
        infer_server.SERVER_CONFIG["vla_runtime"] = {
            "mode": "single_model",
            "shared_profile": "model_at_a",
        }
        infer_server.REFERENCE.robot = SimpleNamespace(
            open_grippers=lambda: opened.append(True) or True,
        )
        infer_server.REFERENCE.run_vla = lambda instruction, max_steps: {
            "success": True,
            "details": {"steps": max_steps},
        }
        infer_server.preload_profiles()

        result = infer_server.execute(SimpleNamespace(
            profile="model_at_a", instruction="pick task", max_steps=800,
        ))

        self.assertTrue(result["success"])
        self.assertEqual(opened, [True])
        self.assertTrue(result["details"]["only_pick_grippers_opened"])

    def test_full_pick_place_keeps_grippers_closed_after_pick(self):
        opened = []
        infer_server.SERVER_CONFIG["only_pick"] = False
        infer_server.SERVER_CONFIG["vla_runtime"] = {
            "mode": "single_model",
            "shared_profile": "model_at_a",
        }
        infer_server.REFERENCE.robot = SimpleNamespace(
            open_grippers=lambda: opened.append(True) or True,
        )
        infer_server.REFERENCE.run_vla = lambda instruction, max_steps: {
            "success": True,
            "details": {"steps": max_steps},
        }
        infer_server.preload_profiles()

        result = infer_server.execute(SimpleNamespace(
            profile="model_at_a", instruction="pick task", max_steps=800,
        ))

        self.assertTrue(result["success"])
        self.assertEqual(opened, [])
        self.assertNotIn("only_pick_grippers_opened", result["details"])

    def test_only_pick_does_not_open_grippers_after_early_stop(self):
        opened = []
        infer_server.SERVER_CONFIG["only_pick"] = True
        infer_server.SERVER_CONFIG["vla_runtime"] = {
            "mode": "single_model",
            "shared_profile": "model_at_a",
        }
        infer_server.REFERENCE.robot = SimpleNamespace(
            open_grippers=lambda: opened.append(True) or True,
        )
        infer_server.REFERENCE.run_vla = lambda instruction, max_steps: {
            "success": True,
            "details": {"steps": max_steps - 1},
        }
        infer_server.preload_profiles()

        result = infer_server.execute(SimpleNamespace(
            profile="model_at_a", instruction="pick task", max_steps=800,
        ))

        self.assertTrue(result["success"])
        self.assertEqual(opened, [])

    def test_single_model_switches_complete_rtc_config_per_logical_profile(self):
        infer_server.SERVER_CONFIG["vla_runtime"] = {
            "mode": "single_model",
            "shared_profile": "model_at_a",
        }
        pick_rtc = {
            "enabled": False,
            "execution_horizon": 3,
            "max_guidance_weight": 2,
            "prefix_attention_schedule": "LINEAR",
            "pre_infer_threshold": 8,
            "underflow_wait_seconds": 1,
            "warmup_inferences": 2,
            "warmup_once_per_model": False,
        }
        place_rtc = {
            "enabled": True,
            "execution_horizon": 6,
            "max_guidance_weight": 7,
            "prefix_attention_schedule": "EXP",
            "pre_infer_threshold": 14,
            "underflow_wait_seconds": 3,
            "warmup_inferences": 1,
            "warmup_once_per_model": True,
        }
        infer_server.SERVER_CONFIG["vla_profiles"]["model_at_a"]["rtc"] = pick_rtc
        infer_server.SERVER_CONFIG["vla_profiles"]["model_at_b"]["rtc"] = place_rtc
        infer_server.preload_profiles()

        infer_server.activate_profile("model_at_a")
        self.assertEqual(infer_server.REFERENCE.CFG["rtc"], pick_rtc)

        infer_server.activate_profile("model_at_b")
        self.assertEqual(infer_server.REFERENCE.CFG["rtc"], place_rtc)
        self.assertIs(infer_server.REFERENCE.vla, FakeVLAClient.instances[0])

        infer_server.activate_profile("model_at_a")
        self.assertEqual(infer_server.REFERENCE.CFG["rtc"], pick_rtc)
        self.assertEqual(len(FakeVLAClient.instances), 1)

    def test_single_model_requires_valid_shared_profile(self):
        infer_server.SERVER_CONFIG["vla_runtime"] = {
            "mode": "single_model",
            "shared_profile": "missing",
        }
        with self.assertRaisesRegex(ValueError, "unknown profile"):
            infer_server.preload_profiles()

    def test_single_model_allows_logical_task_without_second_physical_profile(self):
        infer_server.SERVER_CONFIG["vla_runtime"] = {
            "mode": "single_model",
            "shared_profile": "model_at_a",
        }
        del infer_server.SERVER_CONFIG["vla_profiles"]["model_at_b"]

        infer_server.preload_profiles()
        infer_server.activate_profile("model_at_b")

        self.assertEqual(set(infer_server.PROFILE_RUNTIMES), {"model_at_a"})
        self.assertEqual(infer_server.ACTIVE_PROFILE, "model_at_b")
        self.assertIs(infer_server.REFERENCE.vla, FakeVLAClient.instances[0])

    def test_profiles_match_checkpoint_shapes(self):
        configs = infer_server._configured_profiles()
        self.assertNotIn(
            "observation.images.head_left", configs["model_at_a"]["observation_features"]
        )
        self.assertEqual(
            configs["model_at_a"]["observation_features"]["observation.images.head_right"]["shape"],
            [224, 224, 3],
        )
        self.assertEqual(
            configs["model_at_b"]["observation_features"]["observation.images.head_left"]["shape"],
            [224, 224, 3],
        )
        self.assertEqual(configs["model_at_a"]["robot"]["vla_action_dim"], 16)
        self.assertEqual(configs["model_at_b"]["robot"]["vla_action_dim"], 16)

    def test_switch_refreshes_active_observation_schema(self):
        infer_server.preload_profiles()
        infer_server.REFERENCE.REQUIRED_CAMERAS = []
        infer_server.REFERENCE.REQUIRED_STATE_NAMES = []

        infer_server.activate_profile("model_at_a")
        self.assertEqual(
            set(infer_server.REFERENCE.REQUIRED_CAMERAS),
            {"head_right", "left_arm", "right_arm"},
        )
        self.assertEqual(len(infer_server.REFERENCE.REQUIRED_STATE_NAMES), 23)

        infer_server.activate_profile("model_at_b")
        self.assertEqual(
            set(infer_server.REFERENCE.REQUIRED_CAMERAS),
            {"head_left", "head_right", "left_arm", "right_arm"},
        )

    def test_restore_initial_pose_reuses_owned_robot(self):
        robot = SimpleNamespace(set_init_pose=lambda: True)
        infer_server.REFERENCE.robot = robot
        infer_server.REFERENCE.ROBOT_CFG = {"init_joint_positions": {"head": [0.0, 0.0]}}

        result = infer_server.restore_initial_pose()

        self.assertTrue(result["success"])
        self.assertEqual(result["message"], "Initial joint pose restored")

    def test_head_camera_mode_switches_and_reacquires(self):
        switched = []
        infer_server.REFERENCE.robot = SimpleNamespace(
            reacquire_cameras_after_capture_restart=lambda expected_head_wh: switched.append(
                ("reacquire", tuple(expected_head_wh))
            ),
        )
        original = infer_server.switch_to

        def fake_switch(mode):
            switched.append(("switch", mode))
            return 1

        infer_server.switch_to = fake_switch
        try:
            result = infer_server.head_camera_mode(SimpleNamespace(mode="data_collection"))
        finally:
            infer_server.switch_to = original

        self.assertTrue(result["success"])
        self.assertEqual(result["mode"], "data_collection")
        self.assertEqual(switched, [
            ("switch", "data_collection"),
            ("reacquire", (640, 480)),
        ])

    def test_head_camera_mode_requires_robot(self):
        infer_server.REFERENCE.robot = None
        result = infer_server.head_camera_mode(SimpleNamespace(mode="working"))
        self.assertFalse(result["success"])
        self.assertIn("robot not initialized", result["message"])

    def test_prepare_head_camera_defaults_to_skip(self):
        called = []
        original = infer_server.ensure_mode
        infer_server.SERVER_CONFIG["head_camera"] = {}

        def fake_ensure(mode):
            called.append(mode)
            return 1

        infer_server.ensure_mode = fake_ensure
        try:
            infer_server.prepare_head_camera_before_robot_init()
        finally:
            infer_server.ensure_mode = original
        self.assertEqual(called, [])

    def test_prepare_head_camera_can_be_skipped(self):
        called = []
        original = infer_server.ensure_mode
        infer_server.SERVER_CONFIG["head_camera"] = {"startup_mode": "skip"}
        infer_server.ensure_mode = lambda mode: called.append(mode)
        try:
            infer_server.prepare_head_camera_before_robot_init()
        finally:
            infer_server.ensure_mode = original
        self.assertEqual(called, [])

    def test_health_reports_profile_load_ms(self):
        infer_server.preload_profiles()
        infer_server.REFERENCE.robot = SimpleNamespace(is_available=lambda: True)
        result = infer_server.health()
        self.assertIn("load_ms", result["profile_status"]["model_at_a"])
        self.assertGreaterEqual(result["profile_status"]["model_at_a"]["load_ms"], 0.0)
        self.assertGreaterEqual(result["profile_status"]["model_at_b"]["load_ms"], 0.0)

    def test_attach_execute_timing_splits_load_and_inference(self):
        infer_server.preload_profiles()
        result = {
            "success": True,
            "details": {
                "warmup": {"wall_ms": 2900.0},
                "inference_wall_ms": 18000.0,
            },
        }
        out = infer_server._attach_execute_timing(result, "model_at_a", 12.0, infer_server.time.perf_counter())
        self.assertEqual(out["timing"]["activate_ms"], 12.0)
        self.assertEqual(out["timing"]["warmup_ms"], 2900.0)
        self.assertEqual(out["timing"]["load_ms"], 2912.0)
        self.assertEqual(out["timing"]["inference_ms"], 18000.0)
        self.assertGreaterEqual(out["timing"]["preload_ms"], 0.0)
        self.assertEqual(out["timing"]["profile"], "model_at_a")

    def test_attach_execute_timing_prefers_compute_over_wall(self):
        infer_server.preload_profiles()
        result = {
            "success": True,
            "details": {
                "warmup": {
                    "wall_ms": 6900.0,
                    "timing_breakdown": {"count": 1, "compute_ms": 3200.0, "network_ms": 150.0},
                },
                "inference_wall_ms": 126000.0,
                "timing_breakdown": {
                    "count": 27,
                    "compute_ms": 5400.0,
                    "network_ms": 82000.0,
                    "robot_exec_ms": 13300.0,
                    "send_ms": 2700.0,
                    "rtt_ms": 10.0,
                    "wait_spike_ms": 79030.0,
                    "rpc_detail": {"send_rpc_ms": 2700.0, "payload_bytes": 27000},
                },
            },
        }
        out = infer_server._attach_execute_timing(result, "model_at_a", 10.0, infer_server.time.perf_counter())
        self.assertEqual(out["timing"]["load_ms"], 3210.0)
        self.assertEqual(out["timing"]["inference_ms"], 5400.0)
        self.assertEqual(out["timing"]["load_wall_ms"], 6910.0)
        self.assertEqual(out["timing"]["inference_wall_ms"], 126000.0)
        self.assertEqual(out["timing"]["inference_network_ms"], 82000.0)
        self.assertEqual(out["timing"]["robot_exec_ms"], 13300.0)
        self.assertEqual(out["timing"]["execute_other_ms"], 25300.0)
        self.assertEqual(out["timing"]["execute_overlap_ms"], 0.0)
        self.assertEqual(out["timing"]["network_attribution_ms"]["send_rpc_ms"], 2700.0)
        self.assertEqual(out["timing"]["network_attribution_ms"]["baseline_rtt_total_ms"], 270.0)
        self.assertEqual(out["timing"]["rpc_detail"]["payload_bytes"], 27000)
        self.assertGreaterEqual(out["timing"]["preload_ms"], 0.0)

    def test_attach_execute_timing_records_pipeline_overlap(self):
        infer_server.preload_profiles()
        result = {
            "success": True,
            "details": {
                "warmup": {"wall_ms": 2890.0},
                "inference_wall_ms": 16880.0,
                "timing_breakdown": {
                    "count": 13,
                    "compute_ms": 4590.0,
                    "network_ms": 2820.0,
                    "robot_exec_ms": 15710.0,
                },
            },
        }
        out = infer_server._attach_execute_timing(result, "model_at_b", 10.0, infer_server.time.perf_counter())
        self.assertEqual(out["timing"]["inference_ms"], 4590.0)
        self.assertEqual(out["timing"]["inference_network_ms"], 2820.0)
        self.assertEqual(out["timing"]["robot_exec_ms"], 15710.0)
        self.assertEqual(out["timing"]["execute_other_ms"], 0.0)
        self.assertEqual(out["timing"]["execute_overlap_ms"], 6240.0)

    def test_attach_execute_timing_uses_rtc_control_wall_for_robot_timeline(self):
        infer_server.preload_profiles()
        result = {
            "success": True,
            "details": {
                "warmup": {"wall_ms": 0.0},
                "inference_wall_ms": 16870.0,
                "rtc_initial_phase_ms": 620.0,
                "rtc_control_loop_wall_ms": 16000.0,
                "timing_breakdown": {
                    "count": 22,
                    "compute_ms": 9020.0,
                    "network_ms": 5580.0,
                    "robot_exec_ms": 980.0,
                },
            },
        }
        out = infer_server._attach_execute_timing(
            result, "model_at_a", 0.0, infer_server.time.perf_counter()
        )["timing"]
        self.assertEqual(out["robot_exec_ms"], 16000.0)
        self.assertEqual(out["robot_sdk_command_ms"], 980.0)
        self.assertEqual(out["execute_other_ms"], 250.0)
        self.assertEqual(out["execute_overlap_ms"], 13980.0)


if __name__ == "__main__":
    unittest.main()
