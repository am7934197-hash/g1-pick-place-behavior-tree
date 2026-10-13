"""
Galbot VLA Client Server — VLA-Only Mode.

职责: 获取观测 → VLA 推理 → 执行动作。无导航、无放置、无任务流水线。

配置: vla_config.yaml (包含 VLA 模型、机器人硬件、HTTP 服务全部参数)
用法: python server.py [--config vla_config.yaml]
"""

import os
import json
import signal
import socket
import statistics
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

REQUIRED_CAMERAS = ["head_left", "head_right", "left_arm", "right_arm"]
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
    if configured_image_features != expected_image_features:
        raise ValueError(
            "observation image schema mismatch: "
            f"configured={sorted(configured_image_features)}, "
            f"expected={sorted(expected_image_features)}"
        )

    robot_state_names = list(ROBOT_CFG.get("state_feature_names") or [])
    if len(REQUIRED_STATE_NAMES) != 23 or robot_state_names != REQUIRED_STATE_NAMES:
        raise ValueError(
            "23-D state order mismatch between observation.state.names and "
            f"robot.state_feature_names: observation={REQUIRED_STATE_NAMES}, robot={robot_state_names}"
        )

    control = ROBOT_CFG.get("control") or {}
    if control.get("action_semantics", "absolute") != "absolute":
        raise ValueError(
            "PI05 policy postprocessor returns absolute joint targets; "
            "robot.control.action_semantics must be 'absolute'"
        )
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

    checkpoint_dir = CFG.get("pretrained_name_or_path")
    checkpoint_config_path = os.path.join(checkpoint_dir, "config.json") if checkpoint_dir else ""
    if checkpoint_config_path and os.path.isfile(checkpoint_config_path):
        with open(checkpoint_config_path, "r", encoding="utf-8") as f:
            checkpoint_cfg = json.load(f)
        checkpoint_inputs = set((checkpoint_cfg.get("input_features") or {}).keys())
        configured_inputs = set(features.keys())
        if checkpoint_inputs != configured_inputs:
            raise ValueError(
                "checkpoint/config input_features mismatch: "
                f"checkpoint={sorted(checkpoint_inputs)}, configured={sorted(configured_inputs)}"
            )
        checkpoint_action_names = list(checkpoint_cfg.get("action_feature_names") or [])
        if checkpoint_action_names != REQUIRED_STATE_NAMES:
            raise ValueError(
                "checkpoint action_feature_names do not match configured state order: "
                f"checkpoint={checkpoint_action_names}, configured={REQUIRED_STATE_NAMES}"
            )
        if not checkpoint_cfg.get("use_relative_actions", False):
            logger.warning(
                "checkpoint use_relative_actions=false; action output is still expected to be "
                "absolute, but verify this is the intended checkpoint"
            )
    else:
        logger.warning(
            f"checkpoint config is not locally readable: {checkpoint_config_path}; "
            "skipping checkpoint/schema validation"
        )

    logger.info("Runtime config validation passed (4 cameras, 23-D state, absolute execution)")


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
        h, w = int(img.shape[0]), int(img.shape[1])
        if h <= 0 or w <= 0:
            logger.error(f"camera '{cam}' invalid size shape={shape}")
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


def measure_task_network_latency(config: dict) -> dict:
    """Measure TCP handshake latency once at the beginning of a task.

    This intentionally probes the remote SSH endpoint rather than the local
    forwarded port: connecting to 127.0.0.1:30017 would only measure the local
    end of the tunnel, not the network path to the inference machine.
    Probe failure is reported but does not prevent the robot task from running.
    """
    if not config.get("enabled", True):
        return {"enabled": False}

    host = str(config.get("host", "112.65.216.193"))
    port = int(config.get("port", 50017))
    sample_count = max(1, int(config.get("samples", 10)))
    timeout = max(0.1, float(config.get("timeout_seconds", 3.0)))
    interval = max(0.0, float(config.get("interval_seconds", 0.15)))
    # Keep these unparameterized for compatibility with the robot's older
    # Python environments (built-in list[...] requires Python 3.9+).
    samples_ms = []
    failures = []

    logger.info(
        f"Task network latency probe: target={host}:{port}, "
        f"samples={sample_count}, timeout={timeout:.1f}s"
    )
    for index in range(sample_count):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        started = time.perf_counter()
        try:
            sock.connect((host, port))
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            samples_ms.append(elapsed_ms)
            logger.info(
                f"Network TCP sample {index + 1:02d}/{sample_count}: "
                f"{elapsed_ms:.1f} ms"
            )
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            failures.append(failure)
            logger.warning(
                f"Network TCP sample {index + 1:02d}/{sample_count}: FAIL {failure}"
            )
        finally:
            sock.close()

        if index + 1 < sample_count and interval > 0:
            time.sleep(interval)

    result = {
        "enabled": True,
        "target": f"{host}:{port}",
        "requested_samples": sample_count,
        "successful_samples": len(samples_ms),
        "failed_samples": len(failures),
    }
    if samples_ms:
        result.update(
            {
                "min_ms": round(min(samples_ms), 1),
                "avg_ms": round(sum(samples_ms) / len(samples_ms), 1),
                "max_ms": round(max(samples_ms), 1),
                "jitter_ms": round(statistics.stdev(samples_ms), 1)
                if len(samples_ms) > 1
                else 0.0,
            }
        )
        logger.info(
            "Task network latency result: "
            f"min={result['min_ms']:.1f} avg={result['avg_ms']:.1f} "
            f"max={result['max_ms']:.1f} jitter={result['jitter_ms']:.1f} ms, "
            f"success={len(samples_ms)}/{sample_count}"
        )
    else:
        result["error"] = failures[-1] if failures else "no successful samples"
        logger.error(
            f"Task network latency probe failed: target={host}:{port}, "
            f"success=0/{sample_count}"
        )
    return result


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

    # One probe per submitted task. Keep it above the sync/RTC split so neither
    # repeated synchronous inference nor RTC background inference reruns it.
    try:
        network_latency = measure_task_network_latency(CFG.get("network_latency_probe", {}))
    except Exception as exc:
        # Diagnostics must never turn a reachable policy/robot task into a
        # failure (for example because of a malformed optional probe config).
        logger.exception(f"Task network latency probe crashed; continuing task: {exc}")
        network_latency = {
            "enabled": True,
            "error": f"{type(exc).__name__}: {exc}",
        }

    rtc_cfg = CFG.get("rtc", {})
    if rtc_cfg.get("enabled", False):
        result = _run_vla_rtc(instruction, max_steps, rtc_cfg)
    else:
        result = _run_vla_sync(instruction, max_steps)

    details = result.get("details")
    if details is None:
        details = {}
        result["details"] = details
    details["network_latency"] = network_latency
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

        # ---- 5. 结构化日志 ----
        inference_logger.log_step(
            run_id=run_id, instruction=instruction, step=step,
            obs=obs, actions=actions[:n], elapsed_ms=elapsed_ms,
        )

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
      1. 首次推理 → fill buffer[50]
      2. 主线程逐帧取 buffer → 下发
      3. buffer 剩余 ≤ pre_infer_threshold 时，后台线程异步 trigger
      4. 新 chunk 到达 → 融合入 buffer
      5. 机器人无停顿
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    control = ROBOT_CFG.get("control", {})
    dt = control.get("dt", 0.033)

    inference_logger.start_run(run_id, instruction)
    rtc = RTCController(rtc_cfg)

    if hasattr(vla, "update_instruction"):
        vla.update_instruction(instruction)
        time.sleep(0.2)

    _stop_event.clear()
    robot.reset_action_filter()

    # ---- 首次推理填充 buffer ----
    logger.info("RTC: running initial inference...")
    t0_first = time.perf_counter()
    with _robot_lock:
        obs = robot.get_observation(features=CFG.get("observation_features"))
    if obs is None:
        logger.error("RTC: initial observation is None")
        inference_logger.end_run(run_id)
        return {"success": False, "message": "Invalid observation"}
    obs = image_preprocessor.process(obs)
    inference_logger.log_obs(run_id=run_id, step=0, obs=obs, elapsed_ms=(time.perf_counter() - t0_first) * 1000)
    obs["task"] = instruction

    if not validate_observation(obs):
        logger.error("RTC: initial observation schema invalid")
        inference_logger.end_run(run_id)
        return {"success": False, "message": "Invalid observation"}

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
    rtc.add_chunk(action_vectors)
    total_actions = len(action_vectors)
    first_infer_ms = (time.perf_counter() - t0_first) * 1000
    logger.info(f"RTC: initial inference done in {first_infer_ms:.0f}ms, buffer={total_actions} frames")

    step = 0
    async_thread = None

    while step < max_steps:
        if _stop_event.is_set():
            logger.info(f"RTC: stop requested at step {step}")
            vla.reset()
            break

        # ---- 1. 从 buffer 取下一帧 ----
        action = rtc.get_next_action()
        if action is None:
            logger.warning(f"RTC: buffer empty at step {step}, ending")
            break

        # ---- 2. 下发单帧（带安全裁剪+滤波，跳过插值） ----
        with _robot_lock:
            executed = robot.execute_action_chunk(
                [action], dt=dt, skip_interp=True, stop_flag=_stop_event.is_set
            )
        if not executed:
            logger.error(f"RTC: action execution failed at step {step}; stopping")
            break

        # ---- 3. 日志 ----
        logger.info(f"RTC step {step}: buffer_remaining={rtc.buffer_remaining()}")

        step += 1

        # ---- 4. 触发异步推理 ----
        if rtc.should_trigger_inference():
            logger.info(f"RTC: triggering async inference at step {step} (remaining={rtc.buffer_remaining()})")
            rtc.mark_inference_started()
            async_thread = threading.Thread(
                target=_rtc_async_infer,
                args=(rtc, instruction, step, run_id),
                daemon=True,
            )
            async_thread.start()

        # ---- 5. 消费异步推理结果 ----
        if rtc.new_chunk_ready:
            rtc.consume_new_chunk()
            logger.info(f"RTC: fused new chunk at step {step}, buffer={rtc.buffer_remaining()}")
            if async_thread is not None:
                async_thread.join(timeout=1.0)
                async_thread = None

    inference_logger.end_run(run_id)

    return {
        "success": True,
        "message": "VLA RTC loop completed",
        "details": {
            "run_id": run_id,
            "steps": step,
            "total_actions_executed": step,
            "max_steps": max_steps,
        },
    }


def _rtc_async_infer(rtc: RTCController, instruction: str, current_step: int, run_id: str):
    """后台线程：获取观测 → VLA 推理 → 返回结果。"""
    try:
        t0 = time.perf_counter()
        with _robot_lock:
            obs = robot.get_observation(features=CFG.get("observation_features"))
        if obs is None:
            logger.error("RTC async: observation is None, skip infer")
            rtc.mark_inference_done([])
            return
        obs = image_preprocessor.process(obs)
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
            rtc.mark_inference_done(action_vectors)
            logger.info(f"RTC async infer done: obs={obs_elapsed_ms:.0f}ms, infer={infer_elapsed_ms:.0f}ms, "
                        f"actions={len(action_vectors)}")
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

if __name__ == "__main__":
    import uvicorn
    import select
    import termios

    # 后台线程: stdin raw 模式监听 Ctrl+C (字节 0x03) 和右方向键。
    # 右方向键通常由终端编码为 ESC [ C，部分终端使用 ESC O C。
    # galbot SDK 在 C 层用 signal(SIGINT, SIG_IGN) 忽略了 SIGINT，
    # signal handler 和 sigwait 都收不到信号。
    # 但终端 Ctrl+C 同时向 stdin 写入 0x03 字节，关掉 ISIG 后只走字符通路。
    # uvicorn 启动后可能重置终端设置 → 每轮 select 前检查并重设 raw 模式。
    def _stdin_monitor():
        fd = sys.stdin.fileno()
        try:
            old = termios.tcgetattr(fd)
        except termios.error:
            return  # 非 TTY，跳过

        # 目标 raw 模式: 关行缓冲/回显/信号生成，单字节立即可读。
        raw = termios.tcgetattr(fd)
        raw[0] &= ~(termios.IXON | getattr(termios, "IXOFF", 0))          # iflag
        raw[3] &= ~(termios.ICANON | termios.ECHO | termios.ISIG)       # lflag 索引3
        raw[6][termios.VMIN] = 1                                          # cc 索引6
        raw[6][termios.VTIME] = 0
        escape_sequence = b""
        escape_deadline = 0.0

        while True:
            try:
                # 每轮确保 raw 模式激活 (uvicorn 可能重置了)
                cur = termios.tcgetattr(fd)
                if (
                    (cur[3] & (termios.ICANON | termios.ISIG))
                    or (cur[0] & termios.IXON)
                ):
                    termios.tcsetattr(fd, termios.TCSANOW, raw)

                # select 0.3s 超时，不永久阻塞
                r, _, _ = select.select([fd], [], [], 0.3)
                if not r:
                    if escape_sequence and time.monotonic() >= escape_deadline:
                        escape_sequence = b""
                    continue  # 超时，重新检查 raw 模式

                ch = os.read(fd, 1)
                if ch == b"\x03":
                    termios.tcsetattr(fd, termios.TCSADRAIN, old)
                    logger.warning("Ctrl+C detected via stdin, shutting down")
                    _stop_event.set()
                    if vla is not None:
                        vla.disconnect()
                    os._exit(0)

                # 跨多次 os.read 组装方向键的终端转义序列。
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

    _monitor_thread = threading.Thread(target=_stdin_monitor, daemon=True)
    _monitor_thread.start()

    host = SERVER_CFG.get("host", "0.0.0.0")
    port = SERVER_CFG.get("port", 8005)
    logger.info(f"Starting VLA server on {host}:{port}")
    logger.info(f"Instruction: '{CFG.get('instruction', '')}'")
    _log_runtime_identity()
    uvicorn.run(app, host=host, port=port)
    os._exit(0)
