"""
Galbot VLA Client Server — VLA-Only Mode.

职责: 获取观测 → VLA 推理 → 执行动作。无导航、无放置、无任务流水线。

配置: vla_config.yaml (包含 VLA 模型、机器人硬件、HTTP 服务全部参数)
用法: python server.py [--config vla_config.yaml]
"""

import os
import json
import signal
import sys
import time
import yaml
import threading
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional, Dict

import numpy as np
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel

from robot_controller import RobotController
from vla_client import VLAClient
from rtc_controller import RTCController
import inference_logger as logger
from inference_logger import VLAInferenceLogger
from image_preprocessor import ImagePreprocessor
from alignment_schema import (
    ACTION_NAMES,
    CAMERA_KEYS,
    STATE_NAMES,
    TRAINING_FPS,
    TRAINING_LETTERBOX_PAD_TBLR,
)
from checkpoint_guard import format_checkpoint_report, inspect_checkpoint_files, resolve_checkpoint_path


# ------------------------------------------------------------------
# 配置加载
# ------------------------------------------------------------------

def load_config(config_path: str = "vla_config.yaml") -> dict:
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = "vla_config.yaml"
if len(sys.argv) > 1 and sys.argv[1] == "--config":
    CONFIG_PATH = sys.argv[2]
if not os.path.isabs(CONFIG_PATH):
    CONFIG_PATH = os.path.join(_SCRIPT_DIR, CONFIG_PATH)
CONFIG_PATH = os.path.abspath(CONFIG_PATH)

CFG = load_config(CONFIG_PATH)

ROBOT_CFG = CFG.get("robot", {})
VLA_CFG = {**CFG, **(CFG.get("grpc", {}))}  # 展平给 VLAClient
SERVER_CFG = CFG.get("server", {})

REQUIRED_CAMERAS = [
    key.rsplit(".", 1)[-1]
    for key, feature in (CFG.get("observation_features") or {}).items()
    if (feature or {}).get("dtype") == "image"
]
REQUIRED_STATE_NAMES = list(
    ((CFG.get("observation_features") or {}).get("observation.state") or {}).get("names")
    or ROBOT_CFG.get("state_feature_names")
    or []
)


def _log_runtime_identity():
    logger.info(f"script={os.path.abspath(__file__)}")
    logger.info(f"config={CONFIG_PATH}")
    logger.info(f"pretrained_name_or_path={CFG.get('pretrained_name_or_path')}")
    logger.info(f"observation_features={list((CFG.get('observation_features') or {}).keys())}")
    logger.info(f"robot.state_feature_names={ROBOT_CFG.get('state_feature_names')}")
    logger.info(f"observation_mapping={CFG.get('observation_mapping')}")


def validate_runtime_config():
    """Fail before robot motion when config and checkpoint semantics disagree."""
    features = CFG.get("observation_features") or {}
    expected_image_features = {f"observation.images.{name}" for name in REQUIRED_CAMERAS}
    configured_image_features = {
        key for key, value in features.items() if (value or {}).get("dtype") == "image"
    }
    if not REQUIRED_CAMERAS or configured_image_features != expected_image_features:
        raise ValueError(
            "observation image schema mismatch: "
            f"configured={sorted(configured_image_features)}, "
            f"expected={sorted(expected_image_features)}"
        )

    robot_state_names = list(ROBOT_CFG.get("state_feature_names") or [])
    if robot_state_names != STATE_NAMES or REQUIRED_STATE_NAMES != STATE_NAMES:
        raise ValueError(
            "23-D state order mismatch with the approved checkpoint schema: "
            f"observation={REQUIRED_STATE_NAMES}, robot={robot_state_names}, expected={STATE_NAMES}"
        )

    action_dim = int(ROBOT_CFG.get("vla_action_dim", len(REQUIRED_STATE_NAMES)))
    if action_dim != len(ACTION_NAMES):
        raise ValueError(
            f"robot.vla_action_dim must be exactly {len(ACTION_NAMES)}, got {action_dim}"
        )

    control = ROBOT_CFG.get("control") or {}
    if control.get("action_semantics", "absolute") != "absolute":
        raise ValueError(
            "PI05 policy postprocessor returns absolute joint targets; "
            "robot.control.action_semantics must be 'absolute'"
        )
    configured_fps = float(CFG.get("fps", 0) or 0)
    control_fps = float(control.get("fps", 0) or 0)
    if abs(configured_fps - TRAINING_FPS) > 1e-6 or abs(control_fps - TRAINING_FPS) > 1e-6:
        raise ValueError(
            f"training/control frequency mismatch: training={TRAINING_FPS}, "
            f"top_level={configured_fps}, robot.control={control_fps}"
        )
    control_dt = float(control.get("dt", 0) or 0)
    if control_dt <= 0 or abs(control_dt - 1.0 / TRAINING_FPS) > 0.002:
        raise ValueError(
            f"robot.control.dt={control_dt} is incompatible with {TRAINING_FPS} Hz training data"
        )

    image_shapes = {
        tuple(int(v) for v in (feature or {}).get("shape", ())[:2])
        for feature in features.values()
        if (feature or {}).get("dtype") == "image"
    }
    if len(image_shapes) != 1 or any(len(shape) != 2 for shape in image_shapes):
        raise ValueError(f"all configured camera features must share one HWC size, got {image_shapes}")
    configured_image_hw = next(iter(image_shapes))
    image_cfg = CFG.get("image_preprocess") or {}
    target_hw = tuple(int(x) for x in image_cfg.get("target_size", ()))
    if (
        not image_cfg.get("enabled", False)
        or not image_cfg.get("keep_aspect_ratio", False)
        or target_hw != configured_image_hw
    ):
        raise ValueError(
            "image_preprocess must letterbox native frames: enabled=true, "
            "keep_aspect_ratio=true, "
            f"target_size={list(configured_image_hw)}; "
            f"got {image_cfg}"
        )

    rtc_cfg = CFG.get("rtc") or {}
    if rtc_cfg.get("enabled", False):
        chunk_size = int(CFG.get("actions_per_chunk", 0) or 0)
        threshold = int(rtc_cfg.get("pre_infer_threshold", -1))
        if chunk_size <= 0 or not 0 < threshold < chunk_size:
            raise ValueError(
                f"RTC threshold must be inside the action chunk: threshold={threshold}, chunk={chunk_size}"
            )
        if int(rtc_cfg.get("warmup_inferences", 0) or 0) < 1:
            raise ValueError("RTC requires warmup_inferences >= 1 so compile output is never executed")
    interpolation = control.get("interpolation") or {}
    if interpolation.get("enabled", False):
        input_hz = float(interpolation.get("input_hz", CFG.get("fps", 30)))
        output_hz = float(interpolation.get("output_hz", 250))
        if input_hz <= 0 or output_hz <= 0:
            raise ValueError(
                f"invalid interpolation rates: input_hz={input_hz}, output_hz={output_hz}"
            )
        if output_hz < input_hz:
            logger.warning(
                f"interpolation downsamples policy frames: input_hz={input_hz}, "
                f"output_hz={output_hz}"
            )

    enabled_groups = {
        group.get("name")
        for group in (ROBOT_CFG.get("joint_groups") or [])
        if group.get("enabled", True)
    }
    if "head" in enabled_groups:
        raise ValueError(
            "robot.joint_groups.head must remain disabled: training head pose is fixed and "
            "the policy must not drive the head cameras"
        )

    checkpoint_dir = resolve_checkpoint_path(CFG.get("pretrained_name_or_path"))
    CFG["pretrained_name_or_path"] = checkpoint_dir
    VLA_CFG["pretrained_name_or_path"] = checkpoint_dir
    file_info = inspect_checkpoint_files(checkpoint_dir)
    logger.info(format_checkpoint_report(file_info))
    checkpoint_config_path = os.path.join(checkpoint_dir, "config.json")
    if file_info["local_readable"] and os.path.isfile(checkpoint_config_path):
        with open(checkpoint_config_path, "r", encoding="utf-8") as f:
            checkpoint_cfg = json.load(f)
        checkpoint_inputs = set((checkpoint_cfg.get("input_features") or {}).keys())
        configured_inputs = set(features.keys())
        if checkpoint_inputs != configured_inputs:
            raise ValueError(
                "checkpoint/config input_features mismatch: "
                f"checkpoint={sorted(checkpoint_inputs)}, configured={sorted(configured_inputs)}"
            )
        for key, feature in features.items():
            if (feature or {}).get("dtype") != "image":
                continue
            checkpoint_shape = list(
                (((checkpoint_cfg.get("input_features") or {}).get(key) or {}).get("shape") or [])
            )
            configured_shape = list((feature or {}).get("shape") or [])
            expected_hwc = (
                [checkpoint_shape[1], checkpoint_shape[2], checkpoint_shape[0]]
                if len(checkpoint_shape) == 3 else []
            )
            if configured_shape != expected_hwc:
                raise ValueError(
                    f"checkpoint/config image shape mismatch for {key}: "
                    f"checkpoint CHW={checkpoint_shape}, configured HWC={configured_shape}"
                )
        checkpoint_action_shape = list(
            ((((checkpoint_cfg.get("output_features") or {}).get("action") or {}).get("shape")) or [])
        )
        if checkpoint_action_shape != [action_dim]:
            raise ValueError(
                f"checkpoint action shape={checkpoint_action_shape}, configured action_dim={action_dim}"
            )
        checkpoint_action_names = list(checkpoint_cfg.get("action_feature_names") or [])
        expected_action_names = list(ACTION_NAMES)
        if checkpoint_action_names and checkpoint_action_names != expected_action_names:
            raise ValueError(
                "checkpoint action_feature_names do not match the configured action prefix: "
                f"checkpoint={checkpoint_action_names}, configured={expected_action_names}"
            )
        if not checkpoint_cfg.get("use_relative_actions", False):
            raise ValueError("checkpoint must use relative arm actions")
        relative_excludes = list(checkpoint_cfg.get("relative_exclude_joints") or [])
        if relative_excludes != ["gripper"]:
            raise ValueError(
                "checkpoint must exclude only gripper from relative actions; "
                f"got {relative_excludes}"
            )
        normalization = checkpoint_cfg.get("normalization_mapping") or {}
        if normalization.get("STATE") != "QUANTILES" or normalization.get("ACTION") != "QUANTILES":
            raise ValueError(
                "checkpoint STATE/ACTION normalization must both be QUANTILES; "
                f"got {normalization}"
            )
    else:
        logger.warning(
            f"checkpoint files are not locally readable ({checkpoint_dir}); "
            "PolicyServer must load the approved checkpoint including QUANTILES "
            "preprocessor/postprocessor files. Client will still send this exact path."
        )

    logger.info(
        "Runtime config validation passed "
        f"({len(REQUIRED_CAMERAS)} cameras, 23-D state, {action_dim}-D action, absolute execution)"
    )


def validate_observation(obs: Optional[dict]) -> bool:
    """Reject obs that PolicyServer cannot consume. Never send a bad request."""
    if not obs:
        logger.error("observation validation failed: obs is None")
        return False

    missing = []
    for cam in REQUIRED_CAMERAS:
        img = obs.get(f"observation.images.{cam}", obs.get(cam))
        if img is None:
            missing.append(cam)
            continue
        shape = getattr(img, "shape", None)
        if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[-1] != 3:
            logger.error(f"camera '{cam}' bad ndarray: type={type(img)} shape={shape} dtype={getattr(img, 'dtype', None)}")
            missing.append(cam)
            continue
        feature = (CFG.get("observation_features") or {}).get(f"observation.images.{cam}") or {}
        configured_shape = tuple(int(v) for v in feature.get("shape", ()))
        expected_hw = configured_shape[:2]
        h, w = int(img.shape[0]), int(img.shape[1])
        if len(expected_hw) != 2 or (h, w) != expected_hw:
            logger.error(
                f"camera '{cam}' shape={shape}, expected HWC {configured_shape} after preprocessing"
            )
            missing.append(cam)
            continue
        if img.dtype != np.uint8:
            logger.error(f"camera '{cam}' dtype={img.dtype}, expected uint8, shape={shape}")
            missing.append(cam)
            continue
        if img.size == 0 or not np.any(img):
            logger.error(f"camera '{cam}' is empty/all-zero, refusing to send, shape={shape}")
            missing.append(cam)
            continue
        image_cfg = CFG.get("image_preprocess") or {}
        if image_cfg.get("enforce_training_geometry", True):
            from image_preprocessor import count_letterbox_black_bar_rows

            top, bottom = count_letterbox_black_bar_rows(img)
            expected_top, expected_bottom, _, _ = TRAINING_LETTERBOX_PAD_TBLR
            tolerance = int(image_cfg.get("black_bar_tolerance_px", 4))
            if abs(top - expected_top) > tolerance or abs(bottom - expected_bottom) > tolerance:
                logger.error(
                    f"camera '{cam}' letterbox bars top/bottom={top}/{bottom}, expected "
                    f"{expected_top}/{expected_bottom}±{tolerance}; refusing geometry that differs "
                    "from the configured training geometry"
                )
                missing.append(cam)
                continue

    if len(REQUIRED_STATE_NAMES) != 23:
        logger.error(
            f"state_feature_names length={len(REQUIRED_STATE_NAMES)}, expected 23: {REQUIRED_STATE_NAMES}"
        )
        logger.error(f"obs.keys()={list(obs.keys())}")
        return False

    finite_count = 0
    for name in REQUIRED_STATE_NAMES:
        if name not in obs:
            missing.append(name)
            continue
        try:
            val = float(obs[name])
        except (TypeError, ValueError):
            logger.error(f"state '{name}' is not numeric: {obs[name]!r}")
            missing.append(name)
            continue
        if not np.isfinite(val):
            logger.error(f"state '{name}' is not finite: {val}")
            missing.append(name)
            continue
        finite_count += 1

    missing = list(dict.fromkeys(missing))
    if missing or finite_count != 23:
        logger.error(
            f"observation validation failed, missing={missing}, "
            f"finite_state={finite_count}/23, obs.keys()={list(obs.keys())}"
        )
        return False
    return True


def summarize_postprocessed_action_delta(obs: dict, actions: list) -> dict:
    """Compare absolute PolicyServer targets with the observation state.

    Arm radians and gripper model units are reported separately.  A single
    max over all 16 dimensions is misleading because grippers use 0..100 while
    arm joints use radians.
    """
    state = np.asarray([obs[name] for name in REQUIRED_STATE_NAMES[:16]], dtype=np.float32)
    vectors = []
    invalid_actions = 0
    for item in actions:
        value = item.get("action") if isinstance(item, dict) else None
        arr = np.asarray(value, dtype=np.float32).reshape(-1) if value is not None else np.array([])
        if arr.size != 16 or not np.all(np.isfinite(arr)):
            invalid_actions += 1
            continue
        vectors.append(arr)
    if not vectors:
        return {
            "action_semantics": "absolute_postprocessed",
            "valid_action_count": 0,
            "invalid_action_count": invalid_actions,
        }

    action_array = np.stack(vectors)
    abs_delta = np.abs(action_array - state[None, :])
    arm_indices = list(range(7)) + list(range(8, 15))
    arm_delta = abs_delta[:, arm_indices]
    gripper_delta = abs_delta[:, [7, 15]]
    gripper_scale = float((ROBOT_CFG.get("control") or {}).get("gripper_action_scale", 0.0012))
    return {
        "action_semantics": "absolute_postprocessed",
        "valid_action_count": len(vectors),
        "invalid_action_count": invalid_actions,
        "arm_abs_delta_rad": {
            "max": float(np.max(arm_delta)),
            "median": float(np.median(arm_delta)),
            "p95": float(np.quantile(arm_delta, 0.95)),
            "first_frame_max": float(np.max(arm_delta[0])),
            "last_frame_max": float(np.max(arm_delta[-1])),
            "per_joint_max": [float(x) for x in np.max(arm_delta, axis=0)],
        },
        "gripper_abs_delta_model": {
            "right_max": float(np.max(gripper_delta[:, 0])),
            "left_max": float(np.max(gripper_delta[:, 1])),
        },
        "gripper_abs_delta_m": {
            "right_max": float(np.max(gripper_delta[:, 0]) * gripper_scale),
            "left_max": float(np.max(gripper_delta[:, 1]) * gripper_scale),
        },
    }


def log_inferred_actions(run_id: str, instruction: str, step: int, obs: dict,
                         actions: list, elapsed_ms: float,
                         selected_action_count: Optional[int] = None) -> dict:
    """Log the complete received chunk plus a concise hold/motion diagnostic."""
    summary = summarize_postprocessed_action_delta(obs, actions)
    arm = summary.get("arm_abs_delta_rad") or {}
    logger.info(
        "Policy action delta: semantics=absolute_postprocessed received=%d selected=%s "
        "arm_rad max=%.6f median=%.6f p95=%.6f first=%.6f last=%.6f "
        "gripper_model=%s gripper_m=%s",
        len(actions),
        selected_action_count,
        float(arm.get("max", float("nan"))),
        float(arm.get("median", float("nan"))),
        float(arm.get("p95", float("nan"))),
        float(arm.get("first_frame_max", float("nan"))),
        float(arm.get("last_frame_max", float("nan"))),
        summary.get("gripper_abs_delta_model"),
        summary.get("gripper_abs_delta_m"),
    )
    inference_logger.log_step(
        run_id=run_id,
        instruction=instruction,
        step=step,
        obs=obs,
        actions=actions,
        elapsed_ms=elapsed_ms,
        selected_action_count=selected_action_count,
        action_delta_summary=summary,
    )
    return summary


# ------------------------------------------------------------------
# 全局状态
# ------------------------------------------------------------------

robot: Optional[RobotController] = None
vla: Optional[VLAClient] = None
_stop_event = threading.Event()  # 用于中断正在运行的推理循环
_recovery_in_progress = threading.Event()  # 右方向键恢复期间拒绝新的 /execute
_recovery_guard = threading.Lock()
inference_logger = VLAInferenceLogger(CFG.get("inference_log", {}))
image_preprocessor = ImagePreprocessor(CFG.get("image_preprocess", {}))
logger.info(
    "image_preprocess enabled=%s target_size=[%s, %s] keep_aspect_ratio=%s "
    "crop_to_training_aspect=%s training_hw=%s "
    "emulate_data_collection=%s scheme=%s cameras=%s zoom=%s out_wh=%s "
    "(client letterbox 224 uint8; PolicyServer /255 + resize_with_pad + *2-1)"
    % (
        image_preprocessor.enabled,
        image_preprocessor._target_h,
        image_preprocessor._target_w,
        image_preprocessor._keep_ratio,
        image_preprocessor._crop_to_training,
        list(image_preprocessor._training_hw),
        image_preprocessor._emulate_dc,
        image_preprocessor._emulate_scheme,
        sorted(image_preprocessor._emulate_cameras),
        image_preprocessor._emulate_zoom,
        list(image_preprocessor._emulate_out_wh),
    )
)

# ------------------------------------------------------------------
# 生命周期
# ------------------------------------------------------------------

def _shutdown_robot_with_timeout(robot: RobotController, timeout: float = 5.0):
    """在独立线程中关闭机器人，超时后强制退出，防止 SDK 阻塞。"""
    result = {"done": False}

    def _do_shutdown():
        try:
            robot.shutdown()
            result["done"] = True
        except Exception as e:
            logger.warning(f"Robot shutdown error: {e}")

    t = threading.Thread(target=_do_shutdown, daemon=True)
    t.start()
    t.join(timeout)
    if not result["done"]:
        logger.warning(f"Robot shutdown timed out after {timeout}s, forcing exit")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global robot, vla
    _log_runtime_identity()
    validate_runtime_config()

    # ---- 初始化机器人 ----
    robot = RobotController(ROBOT_CFG)
    if robot.is_available():
        robot.init()
        time.sleep(2.0)
        robot.set_init_pose()
    else:
        logger.warning("Robot SDK not available, running in simulation mode")

    # ---- 初始化 VLA 客户端 ----
    vla = VLAClient(VLA_CFG)
    if vla.connect():
        auto_setup = VLA_CFG.get("auto_setup_policy", True)
        force_setup = VLA_CFG.get("force_setup_policy", False)
        if auto_setup:
            if force_setup:
                logger.info("force_setup_policy=true, sending setup with configured model path")
                if not vla.setup_policy():
                    logger.error("Policy setup failed; /execute will remain disabled")
            elif vla.check_policy_ready():
                logger.info("Policy already loaded on remote, skipping setup")
            else:
                logger.info("Policy not loaded, triggering setup...")
                if not vla.setup_policy():
                    logger.error("Policy setup failed; /execute will remain disabled")
        logger.info("Connected to PolicyServer")
    else:
        logger.warning("Failed to connect to PolicyServer")

    yield

    # ---- 清理（带超时，防止 SDK 阻塞导致进程无法退出） ----
    logger.info("Shutting down...")
    _stop_event.set()
    if robot is not None:
        _shutdown_robot_with_timeout(robot, timeout=5.0)
    if vla is not None:
        vla.disconnect()
    logger.info("Cleanup done, exiting")
    os._exit(0)  # 强制退出，绕过 SDK 残留的非 daemon 线程


app = FastAPI(title="Galbot VLA Client", lifespan=lifespan)


# ------------------------------------------------------------------
# Pydantic Models
# ------------------------------------------------------------------

class VLARequest(BaseModel):
    instruction: Optional[str] = None
    max_steps: int = 2000


class VLAResponse(BaseModel):
    success: bool
    message: str
    details: Optional[dict] = None


# ------------------------------------------------------------------
# VLA 推理循环
# ------------------------------------------------------------------

# Robot 操作锁（RTC 模式下主线程下发动作 vs 异步线程获取观测）
_robot_lock = threading.Lock()


def _request_interactive_recovery(source: str) -> bool:
    """Stop policy execution and restore the configured initial pose safely.

    The recovery runs in a worker because it may wait for an in-flight command
    chunk to release ``_robot_lock``.  The process and PolicyServer connection
    remain alive so the next task can be started normally after recovery.
    """
    with _recovery_guard:
        if _recovery_in_progress.is_set():
            logger.warning(f"Recovery already in progress; ignoring duplicate request from {source}")
            return False
        _recovery_in_progress.set()

    logger.warning(f"Interactive recovery requested from {source}: stopping VLA and returning to initial pose")
    _stop_event.set()
    if vla is not None:
        try:
            vla.reset()
        except Exception as exc:
            logger.warning(f"Policy reset during recovery failed: {exc}")

    def _recover():
        try:
            if robot is None or robot.robot is None:
                logger.error("Recovery skipped: robot is not initialized")
                return
            # An executing chunk observes _stop_event and releases this lock.
            # No VLA command can interleave with the return-to-home command.
            with _robot_lock:
                robot.reset_action_filter()
                ok = robot.set_recovery_pose()
            if ok:
                logger.warning("Interactive recovery complete: initial pose reached, grippers fully open")
            else:
                logger.error("Interactive recovery failed while sending the initial pose")
        except Exception as exc:
            logger.exception(f"Interactive recovery crashed: {exc}")
        finally:
            _recovery_in_progress.clear()

    threading.Thread(target=_recover, name="interactive-recovery", daemon=True).start()
    return True


def _obs_meta() -> dict:
    if robot is None:
        now = time.time()
        return {"obs_capture_ts": now, "state_ts": now, "image_ts": {}}
    return dict(getattr(robot, "last_obs_meta", None) or {})


def _capture_and_preprocess():
    obs = robot.get_observation(features=CFG.get("observation_features"))
    if obs is None:
        return None, _obs_meta()
    obs = image_preprocessor.process(obs)
    return obs, _obs_meta()


def _warmup_policy(instruction: str, warmup_n: int) -> dict:
    """Run compile/warmup inferences and DISCARD every returned chunk.

    The first torch.compile packet is ~2.9s. It must finish before motion starts.
    """
    if warmup_n <= 0:
        return {"success": True, "warmup_inferences": 0, "warmup_ms": []}
    logger.info(f"WARMUP: discarding {warmup_n} inference packet(s) before any robot motion")
    times_ms = []
    for i in range(warmup_n):
        with _robot_lock:
            obs, meta = _capture_and_preprocess()
        if obs is None or not validate_observation(obs):
            return {
                "success": False,
                "warmup_inferences": i,
                "warmup_ms": times_ms,
                "message": "Warmup observation invalid",
            }
        obs["task"] = instruction
        t0 = time.perf_counter()
        actions = vla.infer(obs, instruction, timestep=-1 - i)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        times_ms.append(elapsed_ms)
        n_act = len(actions) if actions else 0
        logger.info(
            f"WARMUP infer {i + 1}/{warmup_n}: {elapsed_ms:.0f}ms actions={n_act} "
            f"DISCARDED (not queued, robot not moving) obs_ts={meta.get('obs_capture_ts')}"
        )
        if not actions:
            return {
                "success": False,
                "warmup_inferences": i + 1,
                "warmup_ms": times_ms,
                "message": "Warmup inference returned no actions",
            }
    return {"success": True, "warmup_inferences": warmup_n, "warmup_ms": times_ms}


def run_vla(instruction: str, max_steps: int = 2000) -> dict:
    """VLA 推理入口，根据 rtc.enabled 自动选择同步/RTC 模式。"""
    if robot is None or vla is None:
        return {"success": False, "message": "Server not initialized", "details": None}
    if _recovery_in_progress.is_set():
        return {"success": False, "message": "Recovery in progress; wait for initial pose", "details": None}
    if not vla.connected:
        return {"success": False, "message": "PolicyServer is not connected", "details": None}
    if not vla.policy_ready:
        return {"success": False, "message": "Policy setup has not completed successfully", "details": None}

    rtc_cfg = CFG.get("rtc", {})
    # Compile/warmup must be the first operation after read-only diagnostics.
    # In particular, do not move the grippers before the multi-second compile
    # packet has completed and been discarded.
    warmup = _warmup_policy(instruction, int(rtc_cfg.get("warmup_inferences", 0) or 0))
    if not warmup.get("success", True):
        return {
            "success": False,
            "message": warmup.get("message", "Policy warmup failed"),
            "details": warmup,
        }

    if robot.robot is not None:
        if not robot.set_training_start_grippers():
            return {
                "success": False,
                "message": "Failed to set training-start gripper widths (0.0 m) after warmup",
                "details": {"warmup": warmup},
            }

    if rtc_cfg.get("enabled", False):
        result = _run_vla_rtc(instruction, max_steps, rtc_cfg)
    else:
        result = _run_vla_sync(instruction, max_steps)

    details = result.get("details")
    if details is None:
        details = {}
        result["details"] = details
    details["warmup"] = warmup
    return result


def _run_vla_sync(instruction: str, max_steps: int) -> dict:
    """同步推理模式（原有逻辑）。"""
    execute_count = CFG.get("execute_actions_per_chunk", 10)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    control = ROBOT_CFG.get("control", {})
    dt = control.get("dt", 0.033)

    inference_logger.start_run(run_id, instruction)

    if hasattr(vla, "update_instruction"):
        vla.update_instruction(instruction)
        time.sleep(0.2)

    step = 0
    total_actions = 0
    _stop_event.clear()
    robot.reset_action_filter()
    if hasattr(robot, "_pump_cameras"):
        logger.info("Pumping cameras on execute thread...")
        robot._pump_cameras(duration=5.0, interval=0.1)

    while step < max_steps:
        if _stop_event.is_set():
            logger.info(f"Stop requested at step {step}, ending inference")
            vla.reset()
            break

        # ---- 1. 获取观测 ----
        t0_obs = time.perf_counter()
        obs = robot.get_observation(features=CFG.get("observation_features"))
        elapsed_obs_ms = (time.perf_counter() - t0_obs) * 1000

        if obs is None:
            logger.warning(f"Failed to get observation at step {step}, retrying...")
            time.sleep(dt)
            continue

        # ---- 1b. 图像预处理 ----
        t0_prep = time.perf_counter()
        obs = image_preprocessor.process(obs)
        elapsed_prep_ms = (time.perf_counter() - t0_prep) * 1000

        inference_logger.log_obs(run_id=run_id, step=step, obs=obs, elapsed_ms=elapsed_obs_ms)
        logger.info(f"Get obs: {elapsed_obs_ms:.0f}ms | preprocess: {elapsed_prep_ms:.0f}ms")

        obs["task"] = instruction

        if not validate_observation(obs):
            logger.warning(f"Skip infer at step {step}: observation schema invalid")
            time.sleep(dt)
            continue

        # ---- 2. VLA 推理 ----
        t0 = time.perf_counter()
        actions = vla.infer(obs, instruction, timestep=step)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # A recovery key may arrive while remote inference is waiting. Never execute
        # the stale chunk after recovery has been requested.
        if _stop_event.is_set():
            logger.info(f"Stop requested while inferring at step {step}, discarding returned actions")
            vla.reset()
            break

        if not actions:
            logger.info(f"No actions returned at step {step}, ending")
            break

        # ---- 3. 提取动作向量 ----
        n = min(execute_count, len(actions))
        action_vectors = [a["action"] for a in actions[:n] if "action" in a]

        # Persist all 50 returned frames before attempting hardware execution.
        # This keeps a failed command path diagnosable and exposes whether only
        # the selected first N frames are hold actions.
        log_inferred_actions(
            run_id, instruction, step, obs, actions, elapsed_ms,
            selected_action_count=len(action_vectors),
        )

        # ---- 4. 下发动作 ----
        with _robot_lock:
            executed = robot.execute_action_chunk(
                action_vectors, dt=dt, stop_flag=_stop_event.is_set
            )
        if not executed:
            logger.error(f"Action execution failed at step {step}; stopping run")
            inference_logger.end_run(run_id)
            return {
                "success": False,
                "message": "Robot rejected or failed to execute action chunk",
                "details": {"run_id": run_id, "step": step},
            }
        total_actions += len(action_vectors)

        logger.info(f"Step {step}: {len(action_vectors)} actions, {elapsed_ms:.0f}ms")
        step += n

    inference_logger.end_run(run_id)

    return {
        "success": True,
        "message": "VLA loop completed",
        "details": {
            "run_id": run_id,
            "steps": step,
            "total_actions_executed": total_actions,
            "max_steps": max_steps,
        },
    }


def _run_vla_rtc(instruction: str, max_steps: int, rtc_cfg: dict) -> dict:
    """RTC 异步推理模式。

    流程:
      1. 首次推理（warmup 已在 run_vla 丢弃 compile 首包）→ fill buffer
      2. 主线程按 30 Hz 单调时钟逐帧取 buffer → 下发（sleep 不含处理时间叠加）
      3. buffer 剩余 ≤ pre_infer_threshold 时，后台线程异步 trigger
      4. 新 chunk 到达 → 按 frames_elapsed 丢掉过时帧后融合
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    control = ROBOT_CFG.get("control", {})
    fps = float(control.get("fps", CFG.get("fps", 30)) or 30)
    period = 1.0 / fps
    dt = float(control.get("dt", period))

    inference_logger.start_run(run_id, instruction)
    rtc = RTCController(rtc_cfg)

    if hasattr(vla, "update_instruction"):
        vla.update_instruction(instruction)
        time.sleep(0.2)

    _stop_event.clear()
    robot.reset_action_filter()

    logger.info("RTC: running initial inference (post-warmup, this chunk WILL be executed)...")
    t0_first = time.perf_counter()
    with _robot_lock:
        obs, meta = _capture_and_preprocess()
    if obs is None:
        logger.error("RTC: initial observation is None")
        inference_logger.end_run(run_id)
        return {"success": False, "message": "Invalid observation"}
    inference_logger.log_obs(run_id=run_id, step=0, obs=obs, elapsed_ms=(time.perf_counter() - t0_first) * 1000)
    obs["task"] = instruction

    if not validate_observation(obs):
        logger.error("RTC: initial observation schema invalid")
        inference_logger.end_run(run_id)
        return {"success": False, "message": "Invalid observation"}

    # Initial inference has no concurrently consumed action buffer. Do not set
    # RTC's async-inference latch here: leaving it set would prevent every
    # subsequent should_trigger_inference() call after the first chunk.
    actions = vla.infer(obs, instruction, timestep=0)
    if _stop_event.is_set():
        logger.info("RTC: stop requested while initial inference was running; discarding actions")
        vla.reset()
        inference_logger.end_run(run_id)
        return {"success": True, "message": "VLA stopped for recovery"}
    if not actions:
        logger.error("RTC: initial inference returned no actions")
        inference_logger.end_run(run_id)
        return {"success": False, "message": "Initial inference failed"}

    action_vectors = [a["action"] for a in actions if "action" in a]
    obs_age_s = time.time() - float(meta.get("obs_capture_ts") or time.time())
    stale_frames = int(round(max(0.0, obs_age_s) * fps))
    if stale_frames > 0:
        if stale_frames >= len(action_vectors):
            logger.error(
                f"RTC: initial chunk entirely stale obs_age={obs_age_s * 1000:.0f}ms "
                f"stale_frames={stale_frames} chunk={len(action_vectors)}"
            )
            inference_logger.end_run(run_id)
            return {"success": False, "message": "Initial chunk entirely stale after inference latency"}
        logger.warning(
            f"RTC: initial chunk obs_age={obs_age_s * 1000:.0f}ms -> skip first {stale_frames} actions "
            f"(do not execute chunk[0] against a {stale_frames}-frame-old state)"
        )
        action_vectors = action_vectors[stale_frames:]
    log_inferred_actions(
        run_id, instruction, 0, obs, actions,
        elapsed_ms=(time.perf_counter() - t0_first) * 1000,
        selected_action_count=len(action_vectors),
    )
    fusion0 = rtc.add_chunk(action_vectors)
    fusion0["skipped"] = stale_frames
    fusion0["start_index"] = stale_frames
    fusion0["mode"] = "replace_skip_stale" if stale_frames else "replace"
    first_infer_ms = (time.perf_counter() - t0_first) * 1000
    logger.info(
        f"RTC: initial inference done in {first_infer_ms:.0f}ms, buffer={rtc.buffer_remaining()} frames "
        f"fusion={fusion0}"
    )

    step = 0
    async_thread = None
    next_t = time.perf_counter()
    failure_message = None

    while step < max_steps:
        if _stop_event.is_set():
            logger.info(f"RTC: stop requested at step {step}")
            vla.reset()
            break

        action = rtc.get_next_action()
        if action is None:
            logger.warning(f"RTC: buffer empty at step {step}, ending")
            failure_message = "RTC action buffer underflow"
            break

        with _robot_lock:
            executed = robot.execute_action_chunk(
                [action], dt=dt, skip_interp=True, sleep=False, stop_flag=_stop_event.is_set
            )
        if not executed:
            logger.error(f"RTC: action execution failed at step {step}; stopping")
            failure_message = f"Robot rejected VLA action at RTC step {step}"
            _stop_event.set()
            break

        logger.info(f"RTC step {step}: buffer_remaining={rtc.buffer_remaining()}")
        step += 1

        if rtc.should_trigger_inference():
            logger.info(
                f"RTC: triggering async inference at step {step} "
                f"(remaining={rtc.buffer_remaining()})"
            )
            rtc.mark_inference_started()
            async_thread = threading.Thread(
                target=_rtc_async_infer,
                args=(rtc, instruction, step, run_id),
                daemon=True,
            )
            async_thread.start()

        if rtc.new_chunk_ready:
            report, _ = rtc.consume_new_chunk()
            logger.info(
                f"RTC: fused new chunk at step {step}, buffer={rtc.buffer_remaining()} "
                f"mode={(report or {}).get('mode')} skipped={(report or {}).get('skipped')} "
                f"start_index={(report or {}).get('start_index')} "
                f"frames_elapsed={(report or {}).get('frames_elapsed')}"
            )
            if async_thread is not None:
                async_thread.join(timeout=1.0)
                async_thread = None

        next_t += period
        remaining = next_t - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        else:
            # Late: do not skip actions to catch up (would accelerate playback).
            next_t = time.perf_counter()

    # A failed SDK command may coincide with a background request. Give that
    # daemon a short chance to leave the shared robot/VLA path before returning
    # failure to the behaviour tree.
    if async_thread is not None and async_thread.is_alive():
        async_thread.join(timeout=1.0)

    inference_logger.end_run(run_id)

    return {
        "success": failure_message is None,
        "message": failure_message or "VLA RTC loop completed",
        "details": {
            "run_id": run_id,
            "steps": step,
            "total_actions_executed": step,
            "max_steps": max_steps,
        },
    }


def _rtc_async_infer(rtc, instruction, current_step, run_id):
    """后台线程：获取观测 → VLA 推理 → 返回结果。"""
    try:
        t0 = time.perf_counter()
        with _robot_lock:
            obs, meta = _capture_and_preprocess()
        if obs is None:
            logger.error("RTC async: observation is None, skip infer")
            rtc.mark_inference_done([])
            return
        obs_elapsed_ms = (time.perf_counter() - t0) * 1000
        obs["task"] = instruction
        if not validate_observation(obs):
            logger.error("RTC async: observation schema invalid, skip infer")
            rtc.mark_inference_done([])
            return

        inference_logger.log_obs(run_id=run_id, step=current_step, obs=obs, elapsed_ms=obs_elapsed_ms)

        t_infer = time.perf_counter()
        actions = vla.infer(obs, instruction, timestep=current_step)
        infer_elapsed_ms = (time.perf_counter() - t_infer) * 1000

        if _stop_event.is_set():
            logger.info("RTC async: stop requested while inferring; discarding stale chunk")
            rtc.mark_inference_done([])
            return

        if actions:
            action_vectors = [a["action"] for a in actions if "action" in a]
            log_inferred_actions(
                run_id, instruction, current_step, obs, actions, infer_elapsed_ms,
                selected_action_count=len(action_vectors),
            )
            rtc.mark_inference_done(action_vectors)
            logger.info(
                f"RTC async infer done: obs={obs_elapsed_ms:.0f}ms, infer={infer_elapsed_ms:.0f}ms, "
                f"actions={len(action_vectors)}"
            )
        else:
            logger.warning("RTC async inference returned no actions")
            rtc.mark_inference_done([])
    except Exception as e:
        logger.error(f"RTC async inference error: {e}")
        rtc.mark_inference_done([])


# ------------------------------------------------------------------
# HTTP Endpoints
# ------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "robot_available": bool(robot and robot.is_available() and robot.robot is not None),
        "policy_server_connected": vla.connected if vla else False,
        "policy_ready": vla.policy_ready if vla else False,
        "configured_checkpoint": getattr(vla, "configured_checkpoint", None) if vla else None,
        "required_checkpoint": CFG.get("pretrained_name_or_path"),
        "training_fps": TRAINING_FPS,
        "image_preprocess": CFG.get("image_preprocess", {}),
        "instruction": CFG.get("instruction", ""),
    }


@app.post("/camera_snapshot")
def camera_snapshot():
    """Capture four raw camera frames for mapping/exposure/focus diagnosis."""
    if robot is None or robot.robot is None:
        return {"success": False, "message": "Robot camera service is not initialized"}

    with _robot_lock:
        obs = robot.get_observation(features=CFG.get("observation_features"))
    if obs is None:
        return {"success": False, "message": "Failed to capture all four cameras"}

    diag_cfg = CFG.get("camera_diagnostics") or {}
    snapshot_dir = diag_cfg.get("snapshot_dir", "logs/camera_snapshots")
    if not os.path.isabs(snapshot_dir):
        snapshot_dir = os.path.join(_SCRIPT_DIR, snapshot_dir)
    run_dir = os.path.join(snapshot_dir, datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    os.makedirs(run_dir, exist_ok=False)

    cameras = {}
    for name in REQUIRED_CAMERAS:
        image = obs.get(name)
        if not isinstance(image, np.ndarray) or image.ndim != 3:
            cameras[name] = {"error": "missing or invalid frame"}
            continue
        image = np.asarray(image, dtype=np.uint8)
        path = os.path.join(run_dir, f"{name}.jpg")
        Image.fromarray(image).save(path, quality=95)

        gray = image.astype(np.float32).mean(axis=2)
        # Mean absolute neighboring-pixel difference: simple dependency-free
        # sharpness indicator, useful for spotting continuous autofocus hunting.
        sharpness = 0.5 * (
            float(np.abs(np.diff(gray, axis=0)).mean())
            + float(np.abs(np.diff(gray, axis=1)).mean())
        )
        cameras[name] = {
            "path": path,
            "shape": list(image.shape),
            "mean_brightness": float(gray.mean()),
            "contrast_std": float(gray.std()),
            "dark_fraction": float((gray <= 2).mean()),
            "saturated_fraction": float((gray >= 253).mean()),
            "sharpness": sharpness,
        }

    return {
        "success": all("error" not in value for value in cameras.values()),
        "snapshot_dir": run_dir,
        "camera_mapping": (ROBOT_CFG.get("observation_camera_map") or {}),
        "head_state": {
            "head_joint1": obs.get("head_joint1"),
            "head_joint2": obs.get("head_joint2"),
        },
        "cameras": cameras,
    }


@app.post("/execute", response_model=VLAResponse)
def execute(req: VLARequest):
    """执行 VLA 推理。"""
    instruction = req.instruction or CFG.get("instruction", "")
    result = run_vla(instruction=instruction, max_steps=req.max_steps)
    return VLAResponse(**result)


@app.post("/reset")
def reset():
    """重置 VLA 策略状态。"""
    if vla is None:
        return {"success": False, "message": "VLA not initialized"}
    ok = vla.reset()
    return {"success": ok}


@app.post("/reconnect")
def reconnect():
    """重连 PolicyServer。"""
    if vla is None:
        return {"success": False, "message": "VLA not initialized"}
    if vla.connected:
        vla.disconnect()
    ok = vla.connect()
    if ok:
        ok = vla.setup_policy()
    return {
        "success": ok,
        "connected": vla.connected,
        "policy_ready": vla.policy_ready,
    }
    
@app.post("/stop")
def stop():
    _stop_event.set()          # 中断正在运行的 run_vla 循环
    if vla is not None:
        vla.reset()            # 重置远端 PolicyServer 模型状态
    return {"success": True, "message": "Stop signal sent, policy reset"}


@app.post("/shutdown")
def shutdown():
    """Return first, then release robot resources and terminate the process."""
    logger.warning("Shutdown requested via API")
    _stop_event.set()

    def _graceful_exit():
        # Give the HTTP worker enough time to flush the success response.
        time.sleep(0.3)
        if robot is not None:
            # Never destroy the SDK while an action chunk is still being sent.
            with _robot_lock:
                _shutdown_robot_with_timeout(robot, timeout=5.0)
        if vla is not None:
            vla.disconnect()
        logger.info("API shutdown complete")
        os._exit(0)

    threading.Thread(target=_graceful_exit, daemon=True).start()
    return {"success": True, "message": "Shutdown scheduled"}

# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def start_stdin_monitor() -> None:
    """Listen on this process's terminal for Ctrl+C and the right-arrow key.

    Right-arrow is usually ESC [ C; some terminals send ESC O C.
    The Galbot SDK ignores SIGINT in C, so Ctrl+C is detected as stdin 0x03.
    uvicorn may reset terminal flags, so raw mode is re-applied each loop.
    """
    import select
    import termios

    def _stdin_monitor():
        fd = sys.stdin.fileno()
        try:
            old = termios.tcgetattr(fd)
        except termios.error:
            logger.warning("stdin is not a TTY; right-arrow recovery is disabled")
            return

        raw = termios.tcgetattr(fd)
        raw[0] &= ~(termios.IXON | getattr(termios, "IXOFF", 0))
        raw[3] &= ~(termios.ICANON | termios.ECHO | termios.ISIG)
        raw[6][termios.VMIN] = 1
        raw[6][termios.VTIME] = 0
        escape_sequence = b""
        escape_deadline = 0.0
        logger.warning("Right-arrow on THIS terminal restores initial pose (grippers open)")

        while True:
            try:
                cur = termios.tcgetattr(fd)
                if (
                    (cur[3] & (termios.ICANON | termios.ISIG))
                    or (cur[0] & termios.IXON)
                ):
                    termios.tcsetattr(fd, termios.TCSANOW, raw)

                r, _, _ = select.select([fd], [], [], 0.3)
                if not r:
                    if escape_sequence and time.monotonic() >= escape_deadline:
                        escape_sequence = b""
                    continue

                ch = os.read(fd, 1)
                if ch == b"\x03":
                    termios.tcsetattr(fd, termios.TCSADRAIN, old)
                    logger.warning("Ctrl+C detected via stdin, shutting down")
                    _stop_event.set()
                    if vla is not None:
                        vla.disconnect()
                    os._exit(0)

                if ch == b"\x1b":
                    escape_sequence = ch
                    escape_deadline = time.monotonic() + 0.2
                    continue
                if escape_sequence:
                    escape_sequence += ch
                    if escape_sequence in (b"\x1b[C", b"\x1bOC"):
                        escape_sequence = b""
                        _request_interactive_recovery("terminal right-arrow key")
                    elif escape_sequence not in (b"\x1b[", b"\x1bO"):
                        escape_sequence = b""
            except Exception:
                time.sleep(0.2)

    threading.Thread(target=_stdin_monitor, name="stdin-monitor", daemon=True).start()


if __name__ == "__main__":
    import uvicorn

    start_stdin_monitor()
    host = SERVER_CFG.get("host", "0.0.0.0")
    port = SERVER_CFG.get("port", 8005)
    logger.info(f"Starting VLA server on {host}:{port}")
    logger.info(f"Instruction: '{CFG.get('instruction', '')}'")
    _log_runtime_identity()
    uvicorn.run(app, host=host, port=port)
    os._exit(0)
