#!/usr/bin/env python3
"""One-owner G1 inference server with safe A/B VLA model switching.

The VLA loop and ``RobotController.navigate_to_goal`` are reused from
``project_yuqiz/infer_policy``.  A single server owns the G1 SDK; this is
essential because starting one server per model would initialize/control the
same physical robot twice.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


LOG = logging.getLogger("g1_ab_infer_server")
SCRIPT_DIR = Path(__file__).resolve().parent
INFER_POLICY_DIR = SCRIPT_DIR.parent / "infer_policy"
if str(INFER_POLICY_DIR) not in sys.path:
    sys.path.insert(0, str(INFER_POLICY_DIR))

from head_capture_switch import EXPECTED_WH, ensure_mode, switch_to
from alignment_schema import apply_vla_defaults
RUNTIME_LOCK = threading.RLock()
REFERENCE: Any = None
SERVER_CONFIG: dict[str, Any] = {}
BASE_VLA_CONFIG: dict[str, Any] = {}
ACTIVE_PROFILE: str | None = None
PROFILE_RUNTIMES: dict[str, "ProfileRuntime"] = {}
WARMED_RUNTIME_PROFILES: set[str] = set()
_CONSOLE_LOG_HANDLE: Any = None

MULTI_MODEL_MODE = "multi_model"
SINGLE_MODEL_MODE = "single_model"


@dataclass(frozen=True)
class ProfileRuntime:
    """A policy client that is loaded once and kept alive for the process lifetime."""

    config: dict[str, Any]
    client: Any
    image_preprocessor: Any
    load_ms: float = 0.0


class ExecuteRequest(BaseModel):
    profile: str
    instruction: str
    max_steps: int = Field(default=2000, gt=0)


class ToGoalRequest(BaseModel):
    goal_name: str
    points_file: str
    timeout_seconds: float = Field(default=60.0, gt=0)
    retries: int = Field(default=3, ge=1)


class HeadCameraModeRequest(BaseModel):
    mode: Literal["working", "data_collection"]


def _load_goal_pose(points_file: str, goal_name: str) -> list[float]:
    path = Path(points_file)
    if not path.is_file():
        raise HTTPException(status_code=422, detail=f"points_file not found: {points_file}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=f"cannot read points_file: {exc}") from exc
    rec = (data.get("points") or {}).get(goal_name)
    if not isinstance(rec, dict):
        raise HTTPException(status_code=422, detail=f"goal '{goal_name}' not in {points_file}")
    pose = rec.get("pose")
    if not isinstance(pose, list) or len(pose) != 7:
        raise HTTPException(status_code=422, detail=f"goal '{goal_name}' pose must have 7 numbers")
    try:
        return [float(v) for v in pose]
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"goal '{goal_name}' pose is not numeric") from exc


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


class _TeeStream:
    """Write to the original stream and a log file."""

    def __init__(self, primary, log_file):
        self._primary = primary
        self._log = log_file

    def write(self, data):
        self._primary.write(data)
        try:
            self._log.write(data)
        except Exception:
            pass
        return len(data)

    def flush(self):
        self._primary.flush()
        try:
            self._log.flush()
        except Exception:
            pass

    def isatty(self):
        return False

    def fileno(self):
        return self._primary.fileno()

    @property
    def encoding(self):
        return getattr(self._primary, "encoding", "utf-8")

    def __getattr__(self, name):
        return getattr(self._primary, name)


def attach_console_file_log(log_path: Path) -> Path:
    """Mirror stdout/stderr (print + logging StreamHandler) into a file.

    Must run before importing ``server`` so ``galbot_vla`` binds to the tee.
    """
    global _CONSOLE_LOG_HANDLE
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8")
    _CONSOLE_LOG_HANDLE = log_file
    sys.stdout = _TeeStream(sys.__stdout__, log_file)
    sys.stderr = _TeeStream(sys.__stderr__, log_file)
    return log_path


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_reference(config_path: Path) -> None:
    global REFERENCE, SERVER_CONFIG, BASE_VLA_CONFIG
    SERVER_CONFIG = load_yaml(config_path)
    if not (INFER_POLICY_DIR / "server.py").is_file():
        raise RuntimeError(f"infer_policy has no server.py: {INFER_POLICY_DIR}")

    # The reference code uses absolute imports such as ``from robot_controller``.
    sys.path.insert(0, str(INFER_POLICY_DIR))
    argv_before = sys.argv[:]
    try:
        sys.argv = [str(INFER_POLICY_DIR / "server.py"), "--config", str(config_path)]
        REFERENCE = importlib.import_module("server")
    finally:
        sys.argv = argv_before
    BASE_VLA_CONFIG = copy.deepcopy(REFERENCE.CFG)


def _profile_config(name: str) -> dict[str, Any]:
    profiles = SERVER_CONFIG.get("vla_profiles") or {}
    profile = profiles.get(name)
    if not isinstance(profile, dict):
        raise ValueError(f"Unknown VLA profile '{name}'")
    if not profile.get("pretrained_name_or_path"):
        raise ValueError(f"vla_profiles.{name}.pretrained_name_or_path must be configured")
    # Robot ownership/mapping is shared. Profiles may override model inputs;
    # the current pair differs in camera count and action dimension.
    forbidden = {"robot", "observation_mapping"}.intersection(profile)
    if forbidden:
        raise ValueError(f"Profile '{name}' cannot override shared robot/observation config: {sorted(forbidden)}")
    return apply_vla_defaults(_deep_merge(BASE_VLA_CONFIG, profile))


def _runtime_mode() -> str:
    runtime = SERVER_CONFIG.get("vla_runtime") or {}
    if not isinstance(runtime, dict):
        raise ValueError("vla_runtime must be a mapping")
    mode = str(runtime.get("mode", MULTI_MODEL_MODE)).strip().lower()
    if mode not in {MULTI_MODEL_MODE, SINGLE_MODEL_MODE}:
        raise ValueError(
            f"vla_runtime.mode must be '{MULTI_MODEL_MODE}' or '{SINGLE_MODEL_MODE}', got {mode!r}"
        )
    return mode


def _shared_profile_name() -> str:
    profiles = SERVER_CONFIG.get("vla_profiles") or {}
    runtime = SERVER_CONFIG.get("vla_runtime") or {}
    name = str(runtime.get("shared_profile", "")).strip()
    if not name:
        raise ValueError("vla_runtime.shared_profile is required in single_model mode")
    if name not in profiles:
        raise ValueError(f"vla_runtime.shared_profile references unknown profile '{name}'")
    return name


def _checkpoint_source() -> str:
    runtime = SERVER_CONFIG.get("vla_runtime") or {}
    source = str(runtime.get("checkpoint_source", "auto")).strip().lower()
    if source not in {"auto", "local", "remote"}:
        raise ValueError(
            "vla_runtime.checkpoint_source must be 'auto', 'local', or 'remote', "
            f"got {source!r}"
        )
    return source


def _runtime_profile_name(requested_profile: str) -> str:
    """Map a location's logical profile to the physical loaded model."""
    profiles = SERVER_CONFIG.get("vla_profiles") or {}
    if _runtime_mode() == SINGLE_MODEL_MODE:
        return _shared_profile_name()
    if requested_profile not in profiles:
        raise ValueError(f"Unknown VLA profile '{requested_profile}'")
    return requested_profile


def _validate_profile_schema(name: str, config: dict[str, Any]) -> None:
    """Match one profile to its local checkpoint before any robot motion."""
    from alignment_schema import STATE_NAMES

    if _checkpoint_source() == "remote":
        return
    checkpoint_path = Path(config["pretrained_name_or_path"]) / "config.json"
    if not checkpoint_path.is_file():
        LOG.warning(
            "Profile %s checkpoint config is not locally readable: %s; "
            "skipping local schema validation (remote_pinned). "
            "PolicyServer must load this exact path.",
            name,
            checkpoint_path,
        )
        return
    checkpoint = load_yaml(checkpoint_path) if checkpoint_path.suffix in {".yaml", ".yml"} else json.loads(
        checkpoint_path.read_text(encoding="utf-8")
    )
    configured_features = config.get("observation_features") or {}
    checkpoint_features = checkpoint.get("input_features") or {}
    if set(configured_features) != set(checkpoint_features):
        raise ValueError(
            f"Profile '{name}' input keys mismatch: configured={sorted(configured_features)}, "
            f"checkpoint={sorted(checkpoint_features)}"
        )

    actual_camera_keys = {
        key for key, feature in checkpoint_features.items()
        if (feature or {}).get("type") == "VISUAL"
    }
    supported_camera_sets = {
        frozenset({
            "observation.images.head_right",
            "observation.images.left_arm",
            "observation.images.right_arm",
        }),
        frozenset({
            "observation.images.head_left",
            "observation.images.head_right",
            "observation.images.left_arm",
            "observation.images.right_arm",
        }),
    }
    if frozenset(actual_camera_keys) not in supported_camera_sets:
        raise ValueError(
            f"Profile '{name}' camera schema is unsupported: {sorted(actual_camera_keys)}"
        )
    image_hw: set[tuple[int, int]] = set()
    for key in actual_camera_keys:
        checkpoint_shape = list((checkpoint_features[key] or {}).get("shape") or [])
        configured_shape = list((configured_features[key] or {}).get("shape") or [])
        if len(checkpoint_shape) != 3 or len(configured_shape) != 3:
            raise ValueError(f"Profile '{name}' invalid image shape for {key}")
        expected_hwc = [checkpoint_shape[1], checkpoint_shape[2], checkpoint_shape[0]]
        if configured_shape != expected_hwc:
            raise ValueError(
                f"Profile '{name}' {key} shape mismatch: configured HWC={configured_shape}, "
                f"checkpoint CHW={checkpoint_shape}"
            )
        image_hw.add((configured_shape[0], configured_shape[1]))
    if len(image_hw) != 1:
        raise ValueError(f"Profile '{name}' cameras must share one input size, got {sorted(image_hw)}")
    target_hw = tuple(int(v) for v in ((config.get("image_preprocess") or {}).get("target_size") or []))
    if target_hw != next(iter(image_hw)):
        raise ValueError(
            f"Profile '{name}' image_preprocess.target_size={target_hw} does not match "
            f"checkpoint input={next(iter(image_hw))}"
        )

    state_shape = list((checkpoint_features.get("observation.state") or {}).get("shape") or [])
    state_names = list((configured_features.get("observation.state") or {}).get("names") or [])
    if state_shape != [23] or state_names != STATE_NAMES:
        raise ValueError(
            f"Profile '{name}' state schema mismatch: shape={state_shape}, names={state_names}"
        )
    output_shape = list(
        (((checkpoint.get("output_features") or {}).get("action") or {}).get("shape") or [])
    )
    action_names = list(checkpoint.get("action_feature_names") or [])
    supported_action_schemas = {
        16: list(STATE_NAMES[:16]),
        23: list(STATE_NAMES),
    }
    if len(output_shape) != 1 or output_shape[0] not in supported_action_schemas:
        raise ValueError(
            f"Profile '{name}' action schema mismatch: shape={output_shape}, names={action_names}"
        )
    expected_action_names = supported_action_schemas[output_shape[0]]
    if action_names and action_names != expected_action_names:
        raise ValueError(
            f"Profile '{name}' action names mismatch: expected={expected_action_names}, "
            f"got={action_names}"
        )
    # The shared controller is configured for the short prefix.  It accepts
    # either that prefix or the complete 23-D legacy action vector.
    if int(((config.get("robot") or {}).get("vla_action_dim", 0))) != 16:
        raise ValueError(f"Profile '{name}' requires shared robot.vla_action_dim=16")
    if not checkpoint.get("use_relative_actions", False):
        raise ValueError(f"Profile '{name}' must use relative actions")
    if list(checkpoint.get("relative_exclude_joints") or []) != ["gripper"]:
        raise ValueError(f"Profile '{name}' must exclude only gripper from relative actions")
    normalization = checkpoint.get("normalization_mapping") or {}
    if normalization.get("STATE") != "QUANTILES" or normalization.get("ACTION") != "QUANTILES":
        raise ValueError(
            f"Profile '{name}' must use QUANTILES state/action normalization, got {normalization}"
        )


def _configured_profiles() -> dict[str, dict[str, Any]]:
    profiles = SERVER_CONFIG.get("vla_profiles") or {}
    if not profiles:
        raise ValueError("vla_profiles must contain at least one profile")
    mode = _runtime_mode()
    selected_names = [_shared_profile_name()] if mode == SINGLE_MODEL_MODE else list(profiles)
    configs = {name: _profile_config(name) for name in selected_names}
    addresses: dict[str, str] = {}
    for name, config in configs.items():
        address = str(config.get("policy_server_address", "localhost:8080"))
        if address in addresses:
            raise ValueError(
                f"Profiles '{addresses[address]}' and '{name}' both use PolicyServer {address}. "
                "A PolicyServer owns one model; assign a distinct port to every preloaded profile."
            )
        addresses[address] = name
    from checkpoint_guard import format_checkpoint_report, inspect_checkpoint_files, resolve_checkpoint_path
    checkpoint_source = _checkpoint_source()
    for name, config in configs.items():
        resolved = resolve_checkpoint_path(config["pretrained_name_or_path"])
        config["pretrained_name_or_path"] = resolved
        config["checkpoint_source"] = checkpoint_source
        _validate_profile_schema(name, config)
        if checkpoint_source == "remote":
            LOG.info(
                "Remote VLA profile '%s': %s on %s",
                name,
                resolved,
                config.get("policy_server_address"),
            )
        else:
            info = inspect_checkpoint_files(resolved)
            if checkpoint_source == "local" and not info["local_readable"]:
                raise ValueError(f"Profile '{name}' local checkpoint is incomplete: {resolved}")
            LOG.info("Profile %s checkpoint:\n%s", name, format_checkpoint_report(info))
    return configs


def _preload_one_profile(name: str, config: dict[str, Any]) -> ProfileRuntime:
    t0 = time.perf_counter()
    client = REFERENCE.VLAClient({**config, **(config.get("grpc") or {})})
    try:
        connect_started = time.perf_counter()
        if not client.connect():
            raise RuntimeError(f"Could not connect to PolicyServer for profile '{name}'")
        connect_ms = (time.perf_counter() - connect_started) * 1000
        setup_started = time.perf_counter()
        if not client.setup_policy():
            raise RuntimeError(f"Could not preload policy for profile '{name}'")
        setup_ms = (time.perf_counter() - setup_started) * 1000
        load_ms = (time.perf_counter() - t0) * 1000
        runtime = ProfileRuntime(
            config=config,
            client=client,
            image_preprocessor=REFERENCE.ImagePreprocessor(config.get("image_preprocess", {})),
            load_ms=load_ms,
        )
        LOG.info(
            "Preloaded VLA profile '%s' in %.0fms (connect=%.0fms, remote_setup=%.0fms) "
            "on %s: %s",
            name,
            load_ms,
            connect_ms,
            setup_ms,
            config.get("policy_server_address"),
            config["pretrained_name_or_path"],
        )
        return runtime
    except Exception:
        client.disconnect()
        raise


def preload_profiles() -> None:
    """Load missing policies concurrently and keep successful clients connected."""
    global PROFILE_RUNTIMES
    configs = _configured_profiles()
    loaded = {
        name: runtime
        for name, runtime in PROFILE_RUNTIMES.items()
        if name in configs and runtime.client.policy_ready
    }
    pending = {name: config for name, config in configs.items() if name not in loaded}
    if not pending:
        return
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=len(pending), thread_name_prefix="vla-preload") as executor:
        futures = {
            executor.submit(_preload_one_profile, name, config): name
            for name, config in pending.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                loaded[name] = future.result()
            except Exception as exc:
                LOG.warning("VLA profile '%s' is not ready: %s", name, exc)
                errors.append(f"{name}: {exc}")
    # Publish partial success so a ready profile is never disconnected and
    # reloaded merely because the other PolicyServer is temporarily absent.
    PROFILE_RUNTIMES = loaded
    if errors:
        raise RuntimeError("VLA profile preload failed: " + "; ".join(errors))


def prepare_head_camera_before_robot_init() -> None:
    """Optionally restart head capture before GalbotMotion.init().

    Default is skip: restarting capture unmatches singorix/wbcs/target_server.
    Data-collection FOV is applied in ImagePreprocessor, not by switching para_dir.
    """
    head_cfg = SERVER_CONFIG.get("head_camera") or {}
    mode = str(head_cfg.get("startup_mode", "skip"))
    if mode in ("", "none", "skip"):
        LOG.info("head camera startup switch disabled")
        return
    LOG.info("Preparing head camera mode=%s before Motion.init", mode)
    ensure_mode(mode)


def activate_profile(name: str) -> None:
    """Select an already-loaded remote policy without reloading model weights."""
    global ACTIVE_PROFILE
    runtime_name = _runtime_profile_name(name)
    runtime = PROFILE_RUNTIMES.get(runtime_name)
    if runtime is None:
        raise ValueError(f"Unknown or non-ready VLA runtime profile '{runtime_name}'")
    if not runtime.client.policy_ready:
        raise RuntimeError(f"Preloaded VLA profile '{name}' is no longer connected/ready")
    if ACTIVE_PROFILE == name and REFERENCE.vla is runtime.client:
        return
    # run_vla reads these module globals on every action chunk. Keep the
    # physical model/schema from runtime.config, but apply operational
    # RTC settings from the requested logical task. This also supports A/B
    # sharing one global RTC policy while retaining per-profile enable flags.
    active_config = copy.deepcopy(runtime.config)
    logical_profile = (SERVER_CONFIG.get("vla_profiles") or {}).get(name) or {}
    if isinstance(logical_profile.get("rtc"), dict):
        active_config["rtc"] = _deep_merge(
            active_config.get("rtc") or {}, logical_profile["rtc"]
        )
    REFERENCE.CFG = active_config
    REFERENCE.VLA_CFG = {**active_config, **(active_config.get("grpc") or {})}
    REFERENCE.REQUIRED_CAMERAS = [
        key.rsplit(".", 1)[-1]
        for key, feature in (runtime.config.get("observation_features") or {}).items()
        if (feature or {}).get("dtype") == "image"
    ]
    REFERENCE.REQUIRED_STATE_NAMES = list(
        (((runtime.config.get("observation_features") or {}).get("observation.state") or {}).get("names"))
        or ((runtime.config.get("robot") or {}).get("state_feature_names"))
        or []
    )
    REFERENCE.vla = runtime.client
    REFERENCE.image_preprocessor = runtime.image_preprocessor
    ACTIVE_PROFILE = name
    if runtime_name == name:
        LOG.info("Selected preloaded VLA profile '%s'", name)
    else:
        LOG.info("Selected logical profile '%s' on shared VLA model '%s'", name, runtime_name)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global ACTIVE_PROFILE, PROFILE_RUNTIMES, WARMED_RUNTIME_PROFILES
    try:
        with RUNTIME_LOCK:
            REFERENCE.validate_runtime_config()
            prepare_head_camera_before_robot_init()
            preload_profiles()
            REFERENCE.robot = REFERENCE.RobotController(REFERENCE.ROBOT_CFG)
            if REFERENCE.robot.is_available():
                REFERENCE.robot.init()
            else:
                raise RuntimeError("Galbot SDK/robot unavailable; refusing non-dry-run server startup")
        yield
    finally:
        with RUNTIME_LOCK:
            for runtime in PROFILE_RUNTIMES.values():
                runtime.client.disconnect()
            PROFILE_RUNTIMES = {}
            WARMED_RUNTIME_PROFILES = set()
            ACTIVE_PROFILE = None
            REFERENCE.vla = None
            if REFERENCE.robot is not None:
                REFERENCE._shutdown_robot_with_timeout(REFERENCE.robot, timeout=5.0)


app = FastAPI(title="G1 A-B VLA infer_server", lifespan=lifespan)


@app.get("/health")
def health() -> Dict[str, Any]:
    profile_status = {
        name: {
            "policy_server_address": runtime.config.get("policy_server_address"),
            "ready": bool(runtime.client.policy_ready),
            "load_ms": float(runtime.load_ms),
        }
        for name, runtime in PROFILE_RUNTIMES.items()
    }
    robot_available = bool(REFERENCE and REFERENCE.robot and REFERENCE.robot.is_available())
    all_profiles_ready = bool(profile_status) and all(item["ready"] for item in profile_status.values())
    return {
        "success": robot_available and all_profiles_ready,
        "robot_available": robot_available,
        "vla_runtime_mode": _runtime_mode(),
        "shared_profile": _shared_profile_name() if _runtime_mode() == SINGLE_MODEL_MODE else None,
        "active_profile": ACTIVE_PROFILE,
        "profiles": sorted((SERVER_CONFIG.get("vla_profiles") or {}).keys()),
        "profile_status": profile_status,
        "all_profiles_ready": all_profiles_ready,
    }


def _warmup_wall_ms(details: dict[str, Any] | None) -> float:
    warmup = (details or {}).get("warmup") if isinstance(details, dict) else None
    if not isinstance(warmup, dict):
        return 0.0
    if warmup.get("wall_ms") is not None:
        return float(warmup["wall_ms"])
    try:
        return float(sum(warmup.get("warmup_ms") or []))
    except TypeError:
        return 0.0


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _attach_execute_timing(
    result: dict[str, Any], profile: str, activate_ms: float, started_at: float,
) -> dict[str, Any]:
    details = result.get("details") if isinstance(result.get("details"), dict) else {}
    warmup = details.get("warmup") if isinstance(details.get("warmup"), dict) else {}
    warmup_bd = warmup.get("timing_breakdown") if isinstance(warmup.get("timing_breakdown"), dict) else {}
    loop_bd = details.get("timing_breakdown") if isinstance(details.get("timing_breakdown"), dict) else {}
    warmup_wall = _warmup_wall_ms(details)
    inference_wall = _as_float(details.get("inference_wall_ms"))
    warmup_compute = _as_float(warmup_bd.get("compute_ms"))
    infer_compute = _as_float(loop_bd.get("compute_ms"))
    load_network_ms = _as_float(warmup_bd.get("network_ms"), 0.0) or 0.0
    inference_network_ms = _as_float(loop_bd.get("network_ms"), 0.0) or 0.0
    robot_sdk_command_ms = _as_float(loop_bd.get("robot_exec_ms"), 0.0) or 0.0
    rtc_initial_phase_ms = _as_float(details.get("rtc_initial_phase_ms"))
    rtc_control_loop_wall_ms = _as_float(details.get("rtc_control_loop_wall_ms"))
    # In RTC, one-action SDK calls are deliberately short; the robot is still
    # executing the trajectory throughout the 30 Hz control loop. Use that
    # wall-clock timeline for the user-facing robot execution category.
    robot_exec_ms = (
        float(rtc_control_loop_wall_ms)
        if rtc_control_loop_wall_ms is not None
        else robot_sdk_command_ms
    )
    if warmup_bd.get("count"):
        load_ms = float(activate_ms) + (warmup_compute or 0.0)
    else:
        load_ms = float(activate_ms) + warmup_wall
    if loop_bd.get("count"):
        inference_ms = infer_compute
    else:
        inference_ms = inference_wall
    load_wall_ms = float(activate_ms) + warmup_wall
    inference_wall_ms = 0.0 if inference_wall is None else float(inference_wall)
    compute_ms = 0.0 if inference_ms is None else float(inference_ms)
    busy_ms = compute_ms + inference_network_ms + robot_exec_ms
    if rtc_initial_phase_ms is not None and rtc_control_loop_wall_ms is not None:
        # Initial inference and control playback are serial critical-path
        # phases. Later RTC inference runs in the background and must be
        # represented as overlap, not subtracted again as "other".
        serial_phase_ms = float(rtc_initial_phase_ms) + float(rtc_control_loop_wall_ms)
        execute_other_ms = max(0.0, inference_wall_ms - serial_phase_ms)
        execute_overlap_ms = max(
            0.0, busy_ms + execute_other_ms - inference_wall_ms,
        )
    else:
        execute_other_ms = max(0.0, inference_wall_ms - busy_ms)
        execute_overlap_ms = max(0.0, busy_ms - inference_wall_ms)
    runtime_profile = _runtime_profile_name(profile)
    runtime = PROFILE_RUNTIMES.get(runtime_profile)
    result["timing"] = {
        "profile": profile,
        "runtime_profile": runtime_profile,
        "preload_ms": None if runtime is None else float(runtime.load_ms),
        "activate_ms": float(activate_ms),
        "warmup_ms": warmup_wall,
        "load_ms": load_ms,
        "inference_ms": inference_ms,
        "load_network_ms": load_network_ms,
        "inference_network_ms": inference_network_ms,
        "robot_exec_ms": robot_exec_ms,
        "robot_sdk_command_ms": robot_sdk_command_ms,
        "rtc_initial_phase_ms": rtc_initial_phase_ms,
        "rtc_control_loop_wall_ms": rtc_control_loop_wall_ms,
        "inference_wall_ms": inference_wall,
        "load_wall_ms": load_wall_ms,
        "execute_other_ms": execute_other_ms,
        "execute_overlap_ms": execute_overlap_ms,
        "total_ms": (time.perf_counter() - started_at) * 1000,
        "infer_count": loop_bd.get("count"),
        "median_wait_ms": loop_bd.get("median_wait_ms"),
        "policy_rtt_ms": details.get("policy_rtt_ms"),
        "network_attribution_ms": {
            "send_rpc_ms": loop_bd.get("send_ms"),
            "baseline_rtt_total_ms": (
                float(loop_bd.get("count") or 0) * float(loop_bd.get("rtt_ms") or 0.0)
            ),
            "wait_spike_ms": loop_bd.get("wait_spike_ms"),
        },
        "rpc_detail": dict(loop_bd.get("rpc_detail") or {}),
    }
    return result


@app.post("/execute")
def execute(request: ExecuteRequest) -> Dict[str, Any]:
    global WARMED_RUNTIME_PROFILES
    with RUNTIME_LOCK:
        started_at = time.perf_counter()
        activate_ms = 0.0
        try:
            t_act = time.perf_counter()
            activate_profile(request.profile)
            activate_ms = (time.perf_counter() - t_act) * 1000
            runtime_profile = _runtime_profile_name(request.profile)
            rtc_cfg = REFERENCE.CFG.get("rtc") or {}
            warmup_count = int(rtc_cfg.get("warmup_inferences", 0) or 0)
            warmup_once = bool(rtc_cfg.get("warmup_once_per_model", False))
            skip_warmup = warmup_once and runtime_profile in WARMED_RUNTIME_PROFILES
            if skip_warmup:
                rtc_cfg["warmup_inferences"] = 0
                LOG.info(
                    "Shared VLA model '%s' is already warm; skipping repeated warmup for task '%s'",
                    runtime_profile,
                    request.profile,
                )
            try:
                result = REFERENCE.run_vla(request.instruction, request.max_steps)
            finally:
                if skip_warmup:
                    rtc_cfg["warmup_inferences"] = warmup_count

            # In only-pick mode, release the object only after the requested
            # control horizon really completed.  A failed/early-stopped run
            # must leave the grippers unchanged for safety and diagnosis.
            raw_details = result.get("details") if isinstance(result, dict) else None
            details = raw_details if isinstance(raw_details, dict) else {}
            try:
                completed_steps = int(details.get("steps", -1))
            except (TypeError, ValueError):
                completed_steps = -1
            if (
                SERVER_CONFIG.get("only_pick") is True
                and isinstance(result, dict)
                and result.get("success") is True
                and completed_steps >= request.max_steps
            ):
                opened = bool(REFERENCE.robot.open_grippers())
                details["only_pick_grippers_opened"] = opened
                result["details"] = details
                if not opened:
                    result["success"] = False
                    result["message"] = (
                        "VLA completed max_steps, but failed to open both grippers"
                    )
                else:
                    LOG.info(
                        "only_pick completed %d/%d steps; both grippers opened",
                        completed_steps,
                        request.max_steps,
                    )
            warmup = ((result.get("details") or {}).get("warmup") or {}) if isinstance(result, dict) else {}
            if (
                warmup_once
                and warmup_count > 0
                and isinstance(warmup, dict)
                and warmup.get("success", True)
                and int(warmup.get("warmup_inferences", 0) or 0) > 0
            ):
                WARMED_RUNTIME_PROFILES.add(runtime_profile)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            LOG.exception("VLA execution failed")
            result = {"success": False, "message": str(exc), "details": None}
        return _attach_execute_timing(result, request.profile, activate_ms, started_at)


@app.post("/togoal")
def togoal(request: ToGoalRequest) -> Dict[str, Any]:
    if REFERENCE is None or getattr(REFERENCE, "robot", None) is None:
        return {"success": False, "message": "robot not initialized"}
    pose = _load_goal_pose(request.points_file, request.goal_name)
    last_error = ""
    reached = False
    started_at = time.perf_counter()
    with RUNTIME_LOCK:
        try:
            reached = bool(REFERENCE.robot.navigate_to_goal(
                pose, timeout=request.timeout_seconds, retry=request.retries,
            ))
        except Exception as exc:
            LOG.exception("togoal failed")
            last_error = str(exc)
            reached = False
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    LOG.info("togoal %s finished in %.0fms reached=%s", request.goal_name, elapsed_ms, reached)
    return {
        "success": reached,
        "message": (
            f"Reached map goal {request.goal_name}"
            if reached
            else f"Navigate to {request.goal_name} failed: {last_error or 'not arrived / chassis not ready'}"
        ),
        "goal_name": request.goal_name,
        "pose": pose,
        "elapsed_ms": elapsed_ms,
    }


@app.post("/head_camera_mode")
def head_camera_mode(request: HeadCameraModeRequest) -> Dict[str, Any]:
    """Manual capture restart. The A/B tree no longer calls this mid-episode."""
    if REFERENCE is None or getattr(REFERENCE, "robot", None) is None:
        return {"success": False, "message": "robot not initialized", "mode": request.mode}
    with RUNTIME_LOCK:
        try:
            switch_to(request.mode)
            # switch_to() also restarts wrist capture if the head restart
            # killed left_arm/right_arm drivers.
            REFERENCE.robot.reacquire_cameras_after_capture_restart(
                expected_head_wh=EXPECTED_WH[request.mode],
            )
        except Exception as exc:
            LOG.exception("head camera mode switch failed")
            return {"success": False, "message": str(exc), "mode": request.mode}
    return {
        "success": True,
        "message": f"Head camera capture switched to {request.mode}",
        "mode": request.mode,
        "expected_wh": list(EXPECTED_WH[request.mode]),
    }


@app.post("/restore_initial_pose")
def restore_initial_pose() -> Dict[str, Any]:
    """Restore the training initial pose through the existing sole SDK owner."""
    if REFERENCE.robot is None:
        return {"success": False, "message": "robot not initialized"}
    if not (REFERENCE.ROBOT_CFG.get("init_joint_positions") or {}):
        return {"success": False, "message": "robot.init_joint_positions is not configured"}
    with RUNTIME_LOCK:
        try:
            restored = REFERENCE.robot.set_init_pose()
        except Exception as exc:
            LOG.exception("Initial pose restore failed")
            return {"success": False, "message": f"Initial pose restore failed: {exc}"}
    return {
        "success": restored,
        "message": "Initial joint pose restored" if restored else "Initial joint pose restore failed",
        "verification": getattr(REFERENCE.robot, "last_init_pose_verification", None),
    }


@app.post("/restore_intermediate_arm_pose")
def restore_intermediate_arm_pose() -> Dict[str, Any]:
    """Restore only the two arms after model A; keep both grippers unchanged."""
    if REFERENCE.robot is None:
        return {"success": False, "message": "robot not initialized"}
    pose = SERVER_CONFIG.get("intermediate_arm_pose") or {}
    positions = {
        "right_arm": pose.get("right_arm"),
        "left_arm": pose.get("left_arm"),
    }
    if any(not isinstance(values, list) or len(values) != 7 for values in positions.values()):
        return {
            "success": False,
            "message": "intermediate_arm_pose requires 7 right_arm and 7 left_arm joints",
        }
    with RUNTIME_LOCK:
        try:
            restored = REFERENCE.robot.set_arm_pose_smooth(
                positions,
                duration_seconds=float(pose.get("duration_seconds", 3.0)),
                control_hz=float(pose.get("control_hz", 100.0)),
                settle_seconds=float(pose.get("settle_seconds", 0.2)),
                retries=max(1, int(pose.get("retries", 2))),
                joint_tolerance_rad=float(pose.get("joint_tolerance_rad", 0.03)),
            )
        except Exception as exc:
            LOG.exception("Intermediate arm pose restore failed")
            return {"success": False, "message": f"Intermediate arm pose restore failed: {exc}"}
    return {
        "success": restored,
        "message": (
            "Both arms restored after point A"
            if restored else "Intermediate arm pose restore failed"
        ),
        "verification": getattr(REFERENCE.robot, "last_arm_pose_verification", None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="G1 A/B multi-VLA inference server")
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config.yaml")
    args = parser.parse_args()
    console_log = attach_console_file_log(SCRIPT_DIR / "logs" / "infer_server_console.log")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    _load_reference(args.config.resolve())
    server = SERVER_CONFIG.get("server") or {}
    import uvicorn
    REFERENCE.start_stdin_monitor()
    LOG.warning("Right-arrow on THIS infer_server terminal restores the initial pose")
    LOG.info("Console log file: %s", console_log)
    uvicorn.run(app, host=server.get("host", "0.0.0.0"), port=int(server.get("port", 8006)))


if __name__ == "__main__":
    main()
