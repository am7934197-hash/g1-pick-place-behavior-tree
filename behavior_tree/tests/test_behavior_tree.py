import json
import tempfile
import unittest
from pathlib import Path

from behavior_tree import Context, InferServerClient, Status, build_tree, load_config


NAV_POINTS = "/home/galbot/project_yuqiz/g1_move/slam/nav_points.json"


def nav_cfg(**overrides):
    cfg = {
        "points_file": NAV_POINTS,
        "pick": "pick",
        "place": "place",
        "timeout_seconds": 56.0,
        "retries": 2,
    }
    cfg.update(overrides)
    return cfg


class FakeClient(InferServerClient):
    def __init__(self):
        self.calls = []

    def health(self):
        self.calls.append(("health",))
        return {"success": True}

    def execute(self, profile, instruction, max_steps):
        self.calls.append(("execute", profile, instruction, max_steps))
        return {"success": True}

    def togoal(self, goal_name, points_file, timeout_seconds, retries):
        self.calls.append(("togoal", goal_name, points_file, timeout_seconds, retries))
        return {"success": True}

    def restore_initial_pose(self):
        self.calls.append(("restore_initial_pose",))
        return {"success": True}

    def set_head_camera_mode(self, mode):
        self.calls.append(("set_head_camera_mode", mode))
        return {"success": True}


class BehaviourTreeTest(unittest.TestCase):
    def test_executes_pick_then_nav_to_place_then_place_then_back(self):
        config = {
            "infer_server": {},
            "navigation": nav_cfg(),
            "point_a": {"profile": "a", "instruction": "do A", "max_steps": 12},
            "point_b": {"profile": "b", "instruction": "do B", "max_steps": 34},
        }
        client = FakeClient()
        context = Context(config=config, client=client)

        self.assertEqual(build_tree(context).run(context), Status.SUCCESS)
        self.assertEqual(client.calls, [
            ("health",),
            ("restore_initial_pose",),
            ("execute", "a", "do A", 12),
            ("togoal", "place", NAV_POINTS, 56.0, 2),
            ("execute", "b", "do B", 34),
            ("togoal", "pick", NAV_POINTS, 56.0, 2),
            ("restore_initial_pose",),
        ])
        self.assertNotIn("set_head_camera_mode", [call[0] for call in client.calls])

    def test_vla_failure_at_a_stops_before_chassis_move(self):
        class FailingVLAClient(FakeClient):
            def execute(self, profile, instruction, max_steps):
                self.calls.append(("execute", profile, instruction, max_steps))
                raise RuntimeError("Robot rejected VLA action at RTC step 0")

        config = {
            "infer_server": {},
            "navigation": nav_cfg(),
            "point_a": {"profile": "a", "instruction": "do A", "max_steps": 12},
            "point_b": {"profile": "b", "instruction": "do B", "max_steps": 34},
        }
        client = FailingVLAClient()
        context = Context(config=config, client=client)

        self.assertEqual(build_tree(context).run(context), Status.FAILURE)
        self.assertIn("Robot rejected VLA action", context.error)
        self.assertEqual(client.calls, [
            ("health",),
            ("restore_initial_pose",),
            ("execute", "a", "do A", 12),
        ])
        self.assertNotIn("set_head_camera_mode", [call[0] for call in client.calls])

    def test_initial_pose_failure_stops_before_vla(self):
        class FailingRestoreClient(FakeClient):
            def restore_initial_pose(self):
                self.calls.append(("restore_initial_pose",))
                raise RuntimeError("Initial joint pose restore failed")

        config = {
            "infer_server": {},
            "navigation": nav_cfg(),
            "point_a": {"profile": "a", "instruction": "do A"},
            "point_b": {"profile": "b", "instruction": "do B"},
        }
        client = FailingRestoreClient()
        context = Context(config=config, client=client)

        self.assertEqual(build_tree(context).run(context), Status.FAILURE)
        self.assertIn("Initial joint pose restore failed", context.error)
        self.assertEqual(client.calls, [("health",), ("restore_initial_pose",)])

    def test_rejects_missing_navigation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.yaml"
            path.write_text(
                "infer_server: {}\npoint_a: {profile: a, instruction: A}\n"
                "point_b: {profile: b, instruction: B}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "navigation"):
                load_config(path)

    def test_rejects_unknown_place_name(self):
        with tempfile.TemporaryDirectory() as directory:
            points = Path(directory) / "nav_points.json"
            points.write_text(
                json.dumps({"points": {"pick": {"pose": [0, 0, 0, 0, 0, 0, 1]}}}),
                encoding="utf-8",
            )
            path = Path(directory) / "task.yaml"
            path.write_text(
                "infer_server: {}\n"
                f"navigation: {{points_file: {points}, pick: pick, place: missing}}\n"
                "point_a: {profile: a, instruction: A}\n"
                "point_b: {profile: b, instruction: B}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing"):
                load_config(path)

    def test_load_config_accepts_named_goals(self):
        config = load_config(Path(__file__).resolve().parents[1] / "task.yaml")
        self.assertEqual(config["navigation"]["pick"], "pick")
        self.assertEqual(config["navigation"]["place"], "place")
        self.assertTrue(Path(config["navigation"]["points_file"]).is_file())

    def test_never_switches_head_camera_mid_episode(self):
        config = {
            "infer_server": {},
            "head_camera": {"switch_for_vla": True},
            "navigation": nav_cfg(),
            "point_a": {"profile": "a", "instruction": "do A", "max_steps": 12},
            "point_b": {"profile": "b", "instruction": "do B", "max_steps": 34},
        }
        client = FakeClient()
        context = Context(config=config, client=client)

        self.assertEqual(build_tree(context).run(context), Status.SUCCESS)
        self.assertNotIn("set_head_camera_mode", [call[0] for call in client.calls])
        self.assertEqual(
            [call for call in client.calls if call[0] == "execute"],
            [("execute", "a", "do A", 12), ("execute", "b", "do B", 34)],
        )
