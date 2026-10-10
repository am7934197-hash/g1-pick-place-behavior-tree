#!/usr/bin/env python3
"""A -> B G1 task behaviour tree.

The tree deliberately has only one robot owner: ``infer_server.py``.  This
process sends commands to that server over HTTP, so two VLA stages cannot
compete for the G1 SDK connection.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml


LOG = logging.getLogger("g1_ab_tree")

# Top-level End-to-End stages. Values are wall-clock seconds.
STAGE_ORDER: tuple[tuple[str, str], ...] = (
    ("model_a_load", "模型A任务准备阶段"),
    ("model_a_exec", "模型A执行阶段"),
    ("a_to_b_move", "A→B移动"),
    ("model_b_load", "模型B任务准备阶段"),
    ("model_b_exec", "模型B执行阶段"),
    ("b_to_a_move", "B→A移动"),
    ("restore_pose", "恢复初始姿态"),
)

RESTORE_ACTION_NAMES: tuple[str, ...] = (
    "Restore the training pose before point A",
    "Restore the configured initial joint pose",
)


class Status(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


@dataclass
class Context:
    config: dict[str, Any]
    client: "InferServerClient"
    error: str = ""
    completed_goals: list[str] = field(default_factory=list)
    stage_times: dict[str, float] = field(default_factory=dict)
    preload_times: dict[str, float] = field(default_factory=dict)
    action_times: list[dict[str, Any]] = field(default_factory=list)
    stage_meta: dict[str, dict[str, Any]] = field(default_factory=dict)


class Action:
    def __init__(self, name: str, fn: Callable[[Context], bool]) -> None:
        self.name = name
        self.fn = fn

    def tick(self, context: Context) -> Status:
        LOG.info("[BT] %s", self.name)
        started_at = time.perf_counter()
        status = Status.FAILURE
        try:
            if self.fn(context):
                status = Status.SUCCESS
            else:
                if not context.error:
                    context.error = f"{self.name} failed"
                LOG.error("[BT] %s", context.error)
        except Exception as exc:  # preserve the actual remote error in the log
            context.error = f"{self.name}: {exc}"
            LOG.error("[BT] %s failed: %s", self.name, exc, exc_info=True)
            status = Status.FAILURE
        elapsed_s = time.perf_counter() - started_at
        context.action_times.append({
            "name": self.name,
            "elapsed_s": elapsed_s,
            "status": status.value,
        })
        LOG.info("[BT] %s %s in %.2fs", self.name, status.value, elapsed_s)
        return status


class Sequence:
    def __init__(self, name: str, *children: Action) -> None:
        self.name = name
        self.children = children

    def run(self, context: Context) -> Status:
        LOG.info("[BT] start sequence: %s", self.name)
        started_at = time.perf_counter()
        status = Status.SUCCESS
        for child in self.children:
            status = child.tick(context)
            if status is Status.FAILURE:
                LOG.error("[BT] sequence stopped at %s", child.name)
                break
        else:
            LOG.info("[BT] sequence completed: %s", self.name)
        context.stage_times["tree_wall"] = time.perf_counter() - started_at
        log_timing_summary(context)
        return status


class RemoteCallError(RuntimeError):
    """HTTP 200 with success=false, or a payload that still contains timing."""

    def __init__(self, message: str, result: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.result = result or {}


class InferServerClient:
    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}", body, method="POST" if body is not None else "GET",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} {path}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Cannot reach infer_server at {self.base_url}: {exc.reason}") from exc
        if not result.get("success", False):
            raise RemoteCallError(result.get("message", f"{path} returned success=false"), result)
        return result

    def health(self) -> dict[str, Any]:
        return self.request("/health")

    def execute(self, profile: str, instruction: str, max_steps: int) -> dict[str, Any]:
        return self.request("/execute", {
            "profile": profile, "instruction": instruction, "max_steps": max_steps,
        })

    def togoal(
        self, goal_name: str, points_file: str, timeout_seconds: float, retries: int,
    ) -> dict[str, Any]:
        return self.request("/togoal", {
            "goal_name": goal_name,
            "points_file": points_file,
            "timeout_seconds": timeout_seconds,
            "retries": retries,
        })

    def restore_initial_pose(self) -> dict[str, Any]:
        return self.request("/restore_initial_pose", {})

    def restore_intermediate_arm_pose(self) -> dict[str, Any]:
        return self.request("/restore_intermediate_arm_pose", {})


def _load_nav_points(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    points = data.get("points")
    if not isinstance(points, dict) or not points:
        raise ValueError(f"no points in {path}")
    return points


def _navigation_config(config: dict[str, Any], config_path: Path | None = None) -> dict[str, Any]:
    nav = config.get("navigation")
    if not isinstance(nav, dict):
        raise ValueError("Missing mapping 'navigation'")
    points_file = nav.get("points_file")
    pick = nav.get("pick")
    place = nav.get("place")
    if not points_file or not pick or not place:
        raise ValueError("navigation.points_file, pick, and place are required")
    path = Path(str(points_file))
    if not path.is_absolute() and config_path is not None:
        path = (config_path.parent / path).resolve()
    if not path.is_file():
        raise ValueError(f"navigation.points_file does not exist: {path}")
    points = _load_nav_points(path)
    for name in (pick, place):
        rec = points.get(name)
        if rec is None:
            raise ValueError(f"navigation point '{name}' not in {path}")
        pose = rec.get("pose") if isinstance(rec, dict) else None
        if not isinstance(pose, list) or len(pose) != 7:
            raise ValueError(f"point '{name}' pose must be 7 numbers")
    nav["points_file"] = str(path)
    nav["pick"] = str(pick)
    nav["place"] = str(place)
    nav["timeout_seconds"] = float(nav.get("timeout_seconds", 60.0))
    nav["retries"] = int(nav.get("retries", 3))
    if nav["timeout_seconds"] <= 0:
        raise ValueError("navigation.timeout_seconds must be > 0")
    if nav["retries"] < 1:
        raise ValueError("navigation.retries must be >= 1")
    return nav


def _ms_to_s(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value) / 1000.0
    except (TypeError, ValueError):
        return None


def _busy_vs_wall(
    compute_s: float, network_s: float, robot_s: float, wall_s: float,
) -> tuple[float, float]:
    """Reconcile overlapping busy times against exclusive wall-clock.

    RTC runs inference (compute + network) on a side thread while the robot
    consumes the previous action chunk. Those busy clocks can sum to more
    than the stage wall. ``other_s`` and ``overlap_s`` are mutually exclusive:
    ``compute + network + robot + other - overlap == wall``.
    """
    busy = float(compute_s) + float(network_s) + float(robot_s)
    wall = float(wall_s)
    return max(0.0, wall - busy), max(0.0, busy - wall)


def _reconcile_exec_meta(meta: dict[str, Any] | None, wall_s: float) -> dict[str, Any]:
    out = dict(meta or {})
    compute_s = float(out.get("compute_s") or 0.0)
    network_s = float(out.get("network_s") or 0.0)
    robot_s = float(out.get("robot_s") or 0.0)
    initial_phase_s = out.get("rtc_initial_phase_s")
    control_loop_s = out.get("rtc_control_loop_s")
    if initial_phase_s is not None and control_loop_s is not None:
        # These two RTC phases are serial. Background inference overlaps the
        # control loop, so it must not inflate "other" through subtraction.
        other_s = max(0.0, wall_s - float(initial_phase_s) - float(control_loop_s))
        overlap_s = max(
            0.0, compute_s + network_s + robot_s + other_s - wall_s,
        )
    else:
        other_s, overlap_s = _busy_vs_wall(compute_s, network_s, robot_s, wall_s)
    out["other_s"] = other_s
    out["overlap_s"] = overlap_s
    return out


def _log_exec_breakdown(meta: dict[str, Any]) -> None:
    LOG.info("[BT]   模型计算: %.2fs", float(meta.get("compute_s") or 0.0))
    LOG.info("[BT]   网络通信: %.2fs", float(meta.get("network_s") or 0.0))
    count = int(meta.get("count") or 0)
    attribution = meta.get("network_attribution_ms") or {}
    detail = meta.get("rpc_detail") or {}
    if count:
        LOG.info(
            "[BT]     网络归因: 观测RPC %.2fs + 基线RTT %.2fs + 异常等待 %.2fs",
            float(attribution.get("send_rpc_ms") or 0.0) / 1000.0,
            float(attribution.get("baseline_rtt_total_ms") or 0.0) / 1000.0,
            float(attribution.get("wait_spike_ms") or 0.0) / 1000.0,
        )
        send_ms = float(detail.get("send_rpc_ms") or 0.0)
        actions_ms = float(detail.get("get_actions_rpc_ms") or 0.0)
        payload_mib = float(detail.get("payload_bytes") or 0.0) / 1024.0 / 1024.0
        LOG.info(
            "[BT]     RPC统计: %d次, payload %.2fMiB, 观测RPC %.2fs (%.0fms/次), "
            "动作RPC %.2fs (%.0fms/次)",
            count, payload_mib, send_ms / 1000.0, send_ms / count,
            actions_ms / 1000.0, actions_ms / count,
        )
        LOG.info(
            "[BT]     观测请求细分: 打包 %.2fs, 序列化 %.2fs, 请求迭代 %.2fs, ACK等待 %.2fs",
            float(detail.get("pack_ms") or 0.0) / 1000.0,
            float(detail.get("serialize_ms") or 0.0) / 1000.0,
            float(detail.get("request_iterator_ms") or 0.0) / 1000.0,
            float(detail.get("send_ack_wait_ms") or 0.0) / 1000.0,
        )
        LOG.info(
            "[BT]     服务端动作细分: 排队 %.2fs, 预处理 %.2fs, 模型 %.2fs, "
            "后处理 %.2fs, 序列化 %.2fs",
            float(detail.get("server_queue_ms") or 0.0) / 1000.0,
            float(detail.get("server_preprocess_ms") or 0.0) / 1000.0,
            float(detail.get("server_model_infer_ms") or 0.0) / 1000.0,
            float(detail.get("server_postprocess_ms") or 0.0) / 1000.0,
            float(detail.get("server_serialize_ms") or 0.0) / 1000.0,
        )
        LOG.info(
            "[BT]     客户端调度: 推理锁等待 %.2fs, 响应解码 %.2fs",
            float(detail.get("client_wait_ms") or 0.0) / 1000.0,
            float(detail.get("response_decode_ms") or 0.0) / 1000.0,
        )
    LOG.info("[BT]   机器人执行: %.2fs", float(meta.get("robot_s") or 0.0))
    if meta.get("rtc_control_loop_s") is not None:
        LOG.info(
            "[BT]     RTC关键路径: 首次推理阶段 %.2fs, 30Hz控制/动作播放 %.2fs",
            float(meta.get("rtc_initial_phase_s") or 0.0),
            float(meta.get("rtc_control_loop_s") or 0.0),
        )
        LOG.info(
            "[BT]     SDK下发调用累计: %.2fs（已包含在控制/动作播放时间内）",
            float(meta.get("robot_sdk_s") or 0.0),
        )
    overlap_s = float(meta.get("overlap_s") or 0.0)
    if overlap_s > 0.0:
        LOG.info("[BT]   流水线重叠: -%.2fs", overlap_s)
    LOG.info("[BT]   其他开销: %.2fs", float(meta.get("other_s") or 0.0))


def _vla_stage_prefix(ctx: Context, profile: str) -> str:
    point_a = (ctx.config.get("point_a") or {}).get("profile")
    point_b = (ctx.config.get("point_b") or {}).get("profile")
    if profile == point_a:
        return "model_a"
    if profile == point_b:
        return "model_b"
    return str(profile)


def _record_stage(ctx: Context, key: str, seconds: float, meta: dict[str, Any] | None = None) -> None:
    ctx.stage_times[key] = float(seconds)
    if meta:
        ctx.stage_meta[key] = dict(meta)
    labels = dict(STAGE_ORDER)
    LOG.info("[BT] %s: %.2fs", labels.get(key, key), seconds)
    if meta and key.endswith("_exec"):
        _log_exec_breakdown(meta)


def _record_vla_timing(ctx: Context, profile: str, result: dict[str, Any], wall_s: float) -> None:
    prefix = _vla_stage_prefix(ctx, profile)
    timing = result.get("timing") if isinstance(result, dict) else None
    timing = timing if isinstance(timing, dict) else {}
    runtime_profile = str(timing.get("runtime_profile") or profile)
    preload_s = _ms_to_s(timing.get("preload_ms"))
    if preload_s is not None:
        ctx.preload_times[runtime_profile] = preload_s

    load_wall_s = _ms_to_s(timing.get("load_wall_ms"))
    exec_wall_s = _ms_to_s(timing.get("inference_wall_ms"))
    load_compute_s = _ms_to_s(timing.get("load_ms"))
    exec_compute_s = _ms_to_s(timing.get("inference_ms"))
    load_net_s = _ms_to_s(timing.get("load_network_ms")) or 0.0
    exec_net_s = _ms_to_s(timing.get("inference_network_ms")) or 0.0
    robot_s = _ms_to_s(timing.get("robot_exec_ms")) or 0.0
    robot_sdk_s = _ms_to_s(timing.get("robot_sdk_command_ms")) or 0.0
    rtc_initial_phase_s = _ms_to_s(timing.get("rtc_initial_phase_ms"))
    rtc_control_loop_s = _ms_to_s(timing.get("rtc_control_loop_wall_ms"))

    if (
        load_wall_s is None and exec_wall_s is None
        and load_compute_s is None and exec_compute_s is None
    ):
        _record_stage(ctx, f"{prefix}_exec", wall_s, {
            "compute_s": 0.0,
            "network_s": 0.0,
            "robot_s": 0.0,
            "other_s": wall_s,
            "overlap_s": 0.0,
        })
        return

    if load_wall_s is None:
        load_wall_s = load_compute_s or 0.0
    if exec_wall_s is None:
        exec_wall_s = exec_compute_s if exec_compute_s is not None else max(0.0, wall_s - load_wall_s)
    exec_meta = _reconcile_exec_meta({
        "compute_s": exec_compute_s or 0.0,
        "network_s": exec_net_s,
        "robot_s": robot_s,
        "robot_sdk_s": robot_sdk_s,
        "rtc_initial_phase_s": rtc_initial_phase_s,
        "rtc_control_loop_s": rtc_control_loop_s,
        "count": timing.get("infer_count"),
        "network_attribution_ms": dict(timing.get("network_attribution_ms") or {}),
        "rpc_detail": dict(timing.get("rpc_detail") or {}),
    }, exec_wall_s)

    _record_stage(ctx, f"{prefix}_load", load_wall_s, {
        "compute_s": load_compute_s or 0.0,
        "network_s": load_net_s,
    })
    _record_stage(ctx, f"{prefix}_exec", exec_wall_s, exec_meta)


def _record_move_timing(ctx: Context, key: str, result: dict[str, Any], wall_s: float) -> None:
    elapsed_s = _ms_to_s((result or {}).get("elapsed_ms"))
    _record_stage(ctx, key, wall_s if elapsed_s is None else elapsed_s)


def _restore_pose_s(ctx: Context) -> float:
    return sum(
        float(item.get("elapsed_s") or 0.0)
        for item in ctx.action_times
        if item.get("name") in RESTORE_ACTION_NAMES
    )


def _stage_s(ctx: Context, key: str) -> float:
    value = ctx.stage_times.get(key)
    return 0.0 if value is None else float(value)


def _meta_s(ctx: Context, key: str, field: str) -> float:
    return float((ctx.stage_meta.get(key) or {}).get(field) or 0.0)


def build_timing_report(ctx: Context) -> dict[str, Any]:
    e2e = _stage_s(ctx, "tree_wall")
    restore_s = _restore_pose_s(ctx)
    ctx.stage_times["restore_pose"] = restore_s
    classified = sum(_stage_s(ctx, key) for key, _label in STAGE_ORDER)
    unclassified = max(0.0, e2e - classified)
    ctx.stage_times["unclassified"] = unclassified

    compute_total = (
        _meta_s(ctx, "model_a_load", "compute_s")
        + _meta_s(ctx, "model_a_exec", "compute_s")
        + _meta_s(ctx, "model_b_load", "compute_s")
        + _meta_s(ctx, "model_b_exec", "compute_s")
    )
    network_total = (
        _meta_s(ctx, "model_a_load", "network_s")
        + _meta_s(ctx, "model_a_exec", "network_s")
        + _meta_s(ctx, "model_b_load", "network_s")
        + _meta_s(ctx, "model_b_exec", "network_s")
    )
    robot_total = (
        _meta_s(ctx, "model_a_exec", "robot_s")
        + _meta_s(ctx, "model_b_exec", "robot_s")
    )
    exec_a = _reconcile_exec_meta(ctx.stage_meta.get("model_a_exec"), _stage_s(ctx, "model_a_exec"))
    exec_b = _reconcile_exec_meta(ctx.stage_meta.get("model_b_exec"), _stage_s(ctx, "model_b_exec"))
    overlap_total = float(exec_a.get("overlap_s") or 0.0) + float(exec_b.get("overlap_s") or 0.0)
    move_total = _stage_s(ctx, "a_to_b_move") + _stage_s(ctx, "b_to_a_move")
    other_system = max(
        0.0,
        e2e - compute_total - network_total - robot_total - move_total - restore_s + overlap_total,
    )
    return {
        "e2e_s": e2e,
        "stages_s": {key: _stage_s(ctx, key) for key, _label in STAGE_ORDER},
        "unclassified_s": unclassified,
        "exec_breakdown": {
            "model_a": exec_a,
            "model_b": exec_b,
        },
        "load_breakdown": {
            "model_a": dict(ctx.stage_meta.get("model_a_load") or {}),
            "model_b": dict(ctx.stage_meta.get("model_b_load") or {}),
        },
        "category": {
            "compute_s": compute_total,
            "network_s": network_total,
            "robot_s": robot_total,
            "move_s": move_total,
            "restore_s": restore_s,
            "overlap_s": overlap_total,
            "other_system_s": other_system,
            "e2e_s": e2e,
        },
        "cold_start": dict(ctx.preload_times),
        "actions": ctx.action_times,
    }


def log_timing_summary(ctx: Context) -> None:
    report = build_timing_report(ctx)
    e2e = float(report["e2e_s"])
    stages = report["stages_s"]
    LOG.info("[BT] 任务 End-to-End: %.2fs", e2e)
    LOG.info("[BT] ========== End-to-End 阶段耗时 ==========")
    for key, label in STAGE_ORDER:
        LOG.info("[BT] %s: %.2fs", label, float(stages.get(key) or 0.0))
        if key == "model_a_exec":
            breakdown = report["exec_breakdown"]["model_a"]
        elif key == "model_b_exec":
            breakdown = report["exec_breakdown"]["model_b"]
        else:
            breakdown = None
        if breakdown:
            _log_exec_breakdown(breakdown)
    LOG.info("[BT] 其他未分类开销: %.2fs", float(report["unclassified_s"]))
    LOG.info("[BT] 任务总耗时: %.2fs", e2e)

    category = report["category"]
    LOG.info("[BT] ========== 耗时分类汇总 ==========")
    LOG.info("[BT] 模型计算总计: %.2fs", float(category["compute_s"]))
    LOG.info("[BT] 网络通信总计: %.2fs", float(category["network_s"]))
    LOG.info("[BT] 机器人动作执行总计: %.2fs", float(category["robot_s"]))
    LOG.info("[BT] A/B位姿移动总计: %.2fs", float(category["move_s"]))
    LOG.info("[BT] 恢复初始姿态: %.2fs", float(category["restore_s"]))
    overlap_s = float(category.get("overlap_s") or 0.0)
    if overlap_s > 0.0:
        LOG.info("[BT] 流水线重叠: -%.2fs", overlap_s)
    LOG.info("[BT] 其他系统开销: %.2fs", float(category["other_system_s"]))
    LOG.info("[BT] End-to-End: %.2fs", e2e)

    LOG.info("[BT] ========== Cold Start ==========")
    cold = report["cold_start"]
    if cold:
        for name, seconds in cold.items():
            LOG.info("[BT] %s preload: %.2fs", name, float(seconds))
    else:
        LOG.info("[BT] (no preload times)")


def save_timing_report(ctx: Context, directory: Path | None = None) -> Path:
    log_dir = directory or Path(__file__).with_name("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"timing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    payload = build_timing_report(ctx)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("[BT] timing saved: %s", path)
    return path


def _run_vla(
    ctx: Context, profile: str, instruction: str, max_steps: int,
) -> bool:
    """Run one VLA stage. Head capture stays on working; FOV is emulated in preprocess."""
    started_at = time.perf_counter()
    try:
        result = ctx.client.execute(profile, instruction, max_steps)
    except RemoteCallError as exc:
        _record_vla_timing(ctx, profile, exc.result, time.perf_counter() - started_at)
        raise
    _record_vla_timing(ctx, profile, result, time.perf_counter() - started_at)
    return True


def _only_pick(config: dict[str, Any]) -> bool:
    value = config.get("only_pick", False)
    if not isinstance(value, bool):
        raise ValueError("only_pick must be a YAML boolean (true or false)")
    return value


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    only_pick = _only_pick(config)
    points = ("point_a",) if only_pick else ("point_a", "point_b")
    for key in ("infer_server", *points):
        if not isinstance(config.get(key), dict):
            raise ValueError(f"Missing mapping '{key}' in {path}")
    if not only_pick:
        _navigation_config(config, path)
    for point in points:
        stage = config[point]
        if not stage.get("profile") or not stage.get("instruction"):
            raise ValueError(f"{point}.profile and {point}.instruction are required")
    return config


def build_tree(context: Context) -> Sequence:
    config = context.config
    only_pick = _only_pick(config)
    point_a = config["point_a"]

    def healthcheck(ctx: Context) -> bool:
        health = ctx.client.health()
        for name, status in (health.get("profile_status") or {}).items():
            load_s = _ms_to_s((status or {}).get("load_ms"))
            if load_s is not None:
                ctx.preload_times[str(name)] = load_s
        return True

    def run_at_a(ctx: Context) -> bool:
        return _run_vla(
            ctx, point_a["profile"], point_a["instruction"], int(point_a.get("max_steps", 2000)),
        )

    def restore_initial_pose(ctx: Context) -> bool:
        try:
            ctx.client.restore_initial_pose()
        except RemoteCallError as exc:
            verification = exc.result.get("verification")
            detail = f"; verification={verification}" if verification is not None else ""
            LOG.warning(
                "[BT] Initial pose restore failed; continuing without restored pose: %s%s",
                exc,
                detail,
            )
        return True

    # Always restore the training pose before starting the policy.
    initial_actions = [
        Action("Check infer_server", healthcheck),
        Action("Restore the training pose before point A", restore_initial_pose),
        Action("Run VLA model at pick", run_at_a),
    ]
    LOG.info("[BT] Mode: %s", "only_pick (A only)" if only_pick else "full (A -> B -> pick)")
    if only_pick:
        return Sequence("G1_A_only", *initial_actions)

    point_b = config["point_b"]
    nav = _navigation_config(config)
    points_file = nav["points_file"]
    pick_name = nav["pick"]
    place_name = nav["place"]
    timeout_seconds = float(nav["timeout_seconds"])
    retries = int(nav["retries"])

    def move_to_place(ctx: Context) -> bool:
        started_at = time.perf_counter()
        try:
            result = ctx.client.togoal(place_name, points_file, timeout_seconds, retries)
        except RemoteCallError as exc:
            _record_move_timing(ctx, "a_to_b_move", exc.result, time.perf_counter() - started_at)
            raise
        _record_move_timing(ctx, "a_to_b_move", result, time.perf_counter() - started_at)
        ctx.completed_goals.append(place_name)
        return True

    def restore_intermediate_arm_pose(ctx: Context) -> bool:
        ctx.client.restore_intermediate_arm_pose()
        return True

    def run_at_b(ctx: Context) -> bool:
        return _run_vla(
            ctx, point_b["profile"], point_b["instruction"], int(point_b.get("max_steps", 2000)),
        )

    def return_to_pick(ctx: Context) -> bool:
        if not ctx.completed_goals:
            return True
        started_at = time.perf_counter()
        try:
            result = ctx.client.togoal(pick_name, points_file, timeout_seconds, retries)
        except RemoteCallError as exc:
            _record_move_timing(ctx, "b_to_a_move", exc.result, time.perf_counter() - started_at)
            raise
        _record_move_timing(ctx, "b_to_a_move", result, time.perf_counter() - started_at)
        ctx.completed_goals.clear()
        return True

    return Sequence(
        "G1_A_to_B",
        *initial_actions,
        Action("Restore both arms after point A", restore_intermediate_arm_pose),
        Action("Navigate from pick to place", move_to_place),
        Action("Run VLA model at place", run_at_b),
        Action("Navigate back to pick", return_to_pick),
        Action("Restore the configured initial joint pose", restore_initial_pose),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the G1 A -> B VLA behaviour tree")
    parser.add_argument("--config", default=Path(__file__).with_name("config.yaml"), type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = load_config(args.config)
        server = config["infer_server"]
        client = InferServerClient(
            str(server.get("url", "http://127.0.0.1:8006")), float(server.get("timeout_seconds", 3600)),
        )
        context = Context(config=config, client=client)
        status = build_tree(context).run(context)
        save_timing_report(context)
    except Exception as exc:
        LOG.error("Configuration or startup failure: %s", exc)
        return 2
    if status is Status.SUCCESS:
        return 0
    LOG.error("Task failed: %s", context.error)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
