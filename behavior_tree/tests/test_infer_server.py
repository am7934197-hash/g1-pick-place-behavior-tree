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
        infer_server.BASE_VLA_CONFIG = infer_server.load_yaml(
            behavior_dir.parent / "infer_policy" / "vla_config.yaml"
        )
        infer_server.SERVER_CONFIG = infer_server.load_yaml(behavior_dir / "infer_server.yaml")
        infer_server.SERVER_CONFIG["vla_profiles"]["model_at_a"]["policy_server_address"] = "localhost:1"
        infer_server.SERVER_CONFIG["vla_profiles"]["model_at_b"]["policy_server_address"] = "localhost:2"
        infer_server.REFERENCE = SimpleNamespace(
            VLAClient=FakeVLAClient,
            ImagePreprocessor=lambda config: ("preprocessor", config),
            vla=None,
            CFG=None,
            VLA_CFG=None,
            image_preprocessor=None,
        )
        infer_server.PROFILE_RUNTIMES = {}
        infer_server.ACTIVE_PROFILE = None

    def tearDown(self):
        for runtime in infer_server.PROFILE_RUNTIMES.values():
            runtime.client.disconnect()
        infer_server.PROFILE_RUNTIMES = {}
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


if __name__ == "__main__":
    unittest.main()
