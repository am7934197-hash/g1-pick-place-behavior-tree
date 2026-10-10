import json
import tempfile
import unittest
from pathlib import Path

from behavior_tree import (
    Context,
    InferServerClient,
    RemoteCallError,
    Status,
    _record_vla_timing,
    build_timing_report,
    build_tree,
    load_config,
)


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

    def restore_intermediate_arm_pose(self):
        self.calls.append(("restore_intermediate_arm_pose",))
        return {"success": True}


class BehaviourTreeTest(unittest.TestCase):
    def test_only_pick_runs_a_without_navigation_or_b_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.yaml"
            path.write_text(
                "only_pick: true\ninfer_server: {}\n"
                "point_a: {profile: a, instruction: A, max_steps: 12}\n",
                encoding="utf-8",
            )
            config = load_config(path)
        client = FakeClient()
        context = Context(config=config, client=client)
        self.assertEqual(build_tree(context).run(context), Status.SUCCESS)
        self.assertEqual(client.calls, [
            ("health",), ("restore_initial_pose",), ("execute", "a", "A", 12),
        ])
        report = build_timing_report(context)
        for stage in ("model_b_exec", "a_to_b_move", "b_to_a_move"):
            self.assertEqual(report["stages_s"][stage], 0.0)

    def test_only_pick_stops_on_failure(self):
        for failed_call in ("restore_initial_pose", "execute"):
            with self.subTest(failed_call=failed_call):
                client = FakeClient()
                def fail(*args):
                    client.calls.append((failed_call, *args))
                    raise RuntimeError("test failure")
                setattr(client, failed_call, fail)
                context = Context(config={
                    "only_pick": True,
                    "point_a": {"profile": "a", "instruction": "A", "max_steps": 12},
                }, client=client)
                self.assertEqual(build_tree(context).run(context), Status.FAILURE)
                expected = [("health",), ("restore_initial_pose",)]
                if failed_call == "execute":
                    expected.append(("execute", "a", "A", 12))
                self.assertEqual(client.calls, expected)
                self.assertIn("test failure", context.error)

    def test_rejects_non_boolean_only_pick(self):
        for value in ('"false"', '"true"', "0", "1", "null", "[]", "{}"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "task.yaml"
                path.write_text(f"only_pick: {value}\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "only_pick.*boolean"):
                    load_config(path)
                with self.assertRaisesRegex(ValueError, "only_pick.*boolean"):
                    build_tree(Context(config={"only_pick": value}, client=FakeClient()))

    def test_explicit_false_keeps_full_flow(self):
        config = {
            "navigation": nav_cfg(),
            "point_a": {"profile": "a", "instruction": "A"},
            "point_b": {"profile": "b", "instruction": "B"},
        }
        calls = []
        for overrides in ({}, {"only_pick": False}):
            client = FakeClient()
            context = Context(config={**config, **overrides}, client=client)
            self.assertEqual(build_tree(context).run(context), Status.SUCCESS)
            calls.append(client.calls)
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(len(calls[1]), 8)

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
            ("restore_intermediate_arm_pose",),
            ("togoal", "place", NAV_POINTS, 56.0, 2),
            ("execute", "b", "do B", 34),
            ("togoal", "pick", NAV_POINTS, 56.0, 2),
            ("restore_initial_pose",),
        ])

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

    def test_initial_pose_remote_rejection_warns_and_continues_to_vla(self):
        class RejectingRestoreClient(FakeClient):
            def restore_initial_pose(self):
                self.calls.append(("restore_initial_pose",))
                raise RemoteCallError(
                    "Initial joint pose restore failed",
                    {"success": False, "verification": None},
                )

        config = {
            "only_pick": True,
            "point_a": {"profile": "a", "instruction": "do A", "max_steps": 12},
        }
        client = RejectingRestoreClient()
        context = Context(config=config, client=client)

        with self.assertLogs("g1_ab_tree", level="WARNING") as captured:
            status = build_tree(context).run(context)

        self.assertEqual(status, Status.SUCCESS)
        self.assertEqual(context.error, "")
        self.assertEqual(client.calls, [
            ("health",),
            ("restore_initial_pose",),
            ("execute", "a", "do A", 12),
        ])
        self.assertIn("continuing without restored pose", "\n".join(captured.output))

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
        config = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
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
        self.assertEqual(
            [call for call in client.calls if call[0] == "execute"],
            [("execute", "a", "do A", 12), ("execute", "b", "do B", 34)],
        )

    def test_records_model_load_infer_and_move_times(self):
        class TimingClient(FakeClient):
            def health(self):
                self.calls.append(("health",))
                return {
                    "success": True,
                    "profile_status": {
                        "a": {"ready": True, "load_ms": 45100.0},
                        "b": {"ready": True, "load_ms": 48200.0},
                    },
                }

            def execute(self, profile, instruction, max_steps):
                self.calls.append(("execute", profile, instruction, max_steps))
                if profile == "a":
                    return {
                        "success": True,
                        "timing": {
                            "load_ms": 4540.0,
                            "load_wall_ms": 6810.0,
                            "load_network_ms": 190.0,
                            "inference_ms": 10790.0,
                            "inference_wall_ms": 39480.0,
                            "inference_network_ms": 5480.0,
                            "robot_exec_ms": 18970.0,
                            "execute_other_ms": 4240.0,
                            "preload_ms": 45100.0,
                        },
                    }
                return {
                    "success": True,
                    "timing": {
                        "load_ms": 380.0,
                        "load_wall_ms": 2590.0,
                        "load_network_ms": 160.0,
                        "inference_ms": 11150.0,
                        "inference_wall_ms": 38880.0,
                        "inference_network_ms": 4440.0,
                        "robot_exec_ms": 18990.0,
                        "execute_other_ms": 4300.0,
                        "preload_ms": 48200.0,
                    },
                }

            def togoal(self, goal_name, points_file, timeout_seconds, retries):
                self.calls.append(("togoal", goal_name, points_file, timeout_seconds, retries))
                elapsed = 12050.0 if goal_name == "place" else 11920.0
                return {"success": True, "elapsed_ms": elapsed}

        config = {
            "infer_server": {},
            "navigation": nav_cfg(),
            "point_a": {"profile": "a", "instruction": "do A", "max_steps": 12},
            "point_b": {"profile": "b", "instruction": "do B", "max_steps": 34},
        }
        client = TimingClient()
        context = Context(config=config, client=client)

        self.assertEqual(build_tree(context).run(context), Status.SUCCESS)
        self.assertAlmostEqual(context.stage_times["model_a_load"], 6.81)
        self.assertAlmostEqual(context.stage_times["model_a_exec"], 39.48)
        self.assertAlmostEqual(context.stage_meta["model_a_exec"]["compute_s"], 10.79)
        self.assertAlmostEqual(context.stage_meta["model_a_exec"]["network_s"], 5.48)
        self.assertAlmostEqual(context.stage_meta["model_a_exec"]["robot_s"], 18.97)
        self.assertAlmostEqual(context.stage_meta["model_a_exec"]["other_s"], 4.24)
        self.assertAlmostEqual(context.stage_meta["model_a_exec"]["overlap_s"], 0.0)
        self.assertAlmostEqual(
            context.stage_meta["model_a_exec"]["compute_s"]
            + context.stage_meta["model_a_exec"]["network_s"]
            + context.stage_meta["model_a_exec"]["robot_s"]
            + context.stage_meta["model_a_exec"]["other_s"]
            - context.stage_meta["model_a_exec"]["overlap_s"],
            context.stage_times["model_a_exec"],
        )
        self.assertEqual(context.stage_times["a_to_b_move"], 12.05)
        self.assertAlmostEqual(context.stage_times["model_b_load"], 2.59)
        self.assertAlmostEqual(context.stage_times["model_b_exec"], 38.88)
        self.assertEqual(context.stage_times["b_to_a_move"], 11.92)
        self.assertEqual(context.preload_times["a"], 45.1)
        self.assertEqual(context.preload_times["b"], 48.2)
        report = build_timing_report(context)
        self.assertAlmostEqual(report["stages_s"]["model_a_exec"], 39.48)
        self.assertGreater(report["e2e_s"], 0.0)
        self.assertGreater(report["stages_s"]["restore_pose"], 0.0)
        self.assertEqual(report["cold_start"]["a"], 45.1)
        self.assertEqual(
            [item["name"] for item in context.action_times],
            [
                "Check infer_server",
                "Restore the training pose before point A",
                "Run VLA model at pick",
                "Restore both arms after point A",
                "Navigate from pick to place",
                "Run VLA model at place",
                "Navigate back to pick",
                "Restore the configured initial joint pose",
            ],
        )

    def test_timing_report_closes_end_to_end(self):
        context = Context(config={}, client=FakeClient())
        context.stage_times.update({
            "tree_wall": 116.43,
            "model_a_load": 6.81,
            "model_a_exec": 39.48,
            "a_to_b_move": 6.51,
            "model_b_load": 2.59,
            "model_b_exec": 38.88,
            "b_to_a_move": 6.01,
        })
        context.stage_meta.update({
            "model_a_load": {"compute_s": 4.54, "network_s": 0.19},
            "model_a_exec": {
                "compute_s": 10.79, "network_s": 5.48, "robot_s": 18.97, "other_s": 4.24,
            },
            "model_b_load": {"compute_s": 0.38, "network_s": 0.16},
            "model_b_exec": {
                "compute_s": 11.15, "network_s": 4.44, "robot_s": 18.99, "other_s": 4.30,
            },
        })
        context.action_times = [
            {"name": "Restore the training pose before point A", "elapsed_s": 8.02, "status": "SUCCESS"},
            {"name": "Restore the configured initial joint pose", "elapsed_s": 8.01, "status": "SUCCESS"},
        ]
        report = build_timing_report(context)
        a = report["exec_breakdown"]["model_a"]
        self.assertAlmostEqual(
            a["compute_s"] + a["network_s"] + a["robot_s"] + a["other_s"] - a["overlap_s"],
            report["stages_s"]["model_a_exec"],
        )
        classified = sum(report["stages_s"].values())
        self.assertAlmostEqual(classified + report["unclassified_s"], report["e2e_s"], places=5)
        category = report["category"]
        self.assertAlmostEqual(
            category["compute_s"]
            + category["network_s"]
            + category["robot_s"]
            + category["move_s"]
            + category["restore_s"]
            + category["other_system_s"]
            - category["overlap_s"],
            report["e2e_s"],
            places=5,
        )
        self.assertNotIn(52.92, report["stages_s"].values())
        self.assertEqual(report["cold_start"], {})

    def test_timing_report_accounts_for_rtc_overlap(self):
        context = Context(config={}, client=FakeClient())
        context.stage_times.update({
            "tree_wall": 63.99,
            "model_a_load": 7.17,
            "model_a_exec": 17.20,
            "a_to_b_move": 6.51,
            "model_b_load": 2.89,
            "model_b_exec": 16.88,
            "b_to_a_move": 6.51,
        })
        context.stage_meta.update({
            "model_a_load": {"compute_s": 4.81, "network_s": 0.31},
            "model_a_exec": {
                "compute_s": 4.52, "network_s": 2.77, "robot_s": 15.85, "other_s": 0.0,
            },
            "model_b_load": {"compute_s": 0.65, "network_s": 0.21},
            "model_b_exec": {
                "compute_s": 4.59, "network_s": 2.82, "robot_s": 15.71, "other_s": 0.0,
            },
        })
        context.action_times = [
            {"name": "Restore the training pose before point A", "elapsed_s": 1.73, "status": "SUCCESS"},
            {"name": "Restore the configured initial joint pose", "elapsed_s": 1.73, "status": "SUCCESS"},
        ]
        report = build_timing_report(context)
        a = report["exec_breakdown"]["model_a"]
        b = report["exec_breakdown"]["model_b"]
        self.assertAlmostEqual(a["overlap_s"], 4.52 + 2.77 + 15.85 - 17.20, places=5)
        self.assertAlmostEqual(a["other_s"], 0.0)
        self.assertAlmostEqual(
            a["compute_s"] + a["network_s"] + a["robot_s"] + a["other_s"] - a["overlap_s"],
            report["stages_s"]["model_a_exec"],
            places=5,
        )
        self.assertAlmostEqual(b["overlap_s"], 4.59 + 2.82 + 15.71 - 16.88, places=5)
        self.assertAlmostEqual(
            b["compute_s"] + b["network_s"] + b["robot_s"] + b["other_s"] - b["overlap_s"],
            report["stages_s"]["model_b_exec"],
            places=5,
        )
        category = report["category"]
        self.assertGreater(category["overlap_s"], 0.0)
        self.assertAlmostEqual(
            category["compute_s"]
            + category["network_s"]
            + category["robot_s"]
            + category["move_s"]
            + category["restore_s"]
            + category["other_system_s"]
            - category["overlap_s"],
            report["e2e_s"],
            places=5,
        )

    def test_record_vla_timing_computes_pipeline_overlap(self):
        context = Context(
            config={"point_a": {"profile": "a"}, "point_b": {"profile": "b"}},
            client=FakeClient(),
        )
        _record_vla_timing(
            context,
            "b",
            {
                "timing": {
                    "load_ms": 650.0,
                    "load_wall_ms": 2890.0,
                    "load_network_ms": 210.0,
                    "inference_ms": 4590.0,
                    "inference_wall_ms": 16880.0,
                    "inference_network_ms": 2820.0,
                    "robot_exec_ms": 15710.0,
                    "infer_count": 13,
                },
            },
            19.82,
        )
        meta = context.stage_meta["model_b_exec"]
        self.assertAlmostEqual(context.stage_times["model_b_exec"], 16.88)
        self.assertAlmostEqual(meta["overlap_s"], 6.24)
        self.assertAlmostEqual(meta["other_s"], 0.0)
        self.assertAlmostEqual(
            meta["compute_s"] + meta["network_s"] + meta["robot_s"] + meta["other_s"] - meta["overlap_s"],
            context.stage_times["model_b_exec"],
        )

    def test_record_vla_timing_uses_serial_rtc_phases_for_other(self):
        context = Context(
            config={"point_a": {"profile": "a"}, "point_b": {"profile": "b"}},
            client=FakeClient(),
        )
        _record_vla_timing(
            context,
            "a",
            {
                "timing": {
                    "load_ms": 0.0,
                    "load_wall_ms": 0.0,
                    "inference_ms": 9020.0,
                    "inference_wall_ms": 16870.0,
                    "inference_network_ms": 5580.0,
                    "robot_exec_ms": 16000.0,
                    "robot_sdk_command_ms": 980.0,
                    "rtc_initial_phase_ms": 620.0,
                    "rtc_control_loop_wall_ms": 16000.0,
                    "infer_count": 22,
                },
            },
            16.87,
        )
        meta = context.stage_meta["model_a_exec"]
        self.assertAlmostEqual(meta["robot_s"], 16.0)
        self.assertAlmostEqual(meta["robot_sdk_s"], 0.98)
        self.assertAlmostEqual(meta["other_s"], 0.25)
        self.assertAlmostEqual(meta["overlap_s"], 13.98)
        self.assertAlmostEqual(
            meta["compute_s"] + meta["network_s"] + meta["robot_s"]
            + meta["other_s"] - meta["overlap_s"],
            context.stage_times["model_a_exec"],
        )
