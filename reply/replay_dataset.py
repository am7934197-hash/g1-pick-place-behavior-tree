#!/usr/bin/env python3
"""Replay a LeRobot parquet episode on G1 (dual arms + dual grippers).

Uses set_joint_positions only to reach frame 0, then streams the rest with
set_joint_commands at the recorded frequency. Leg and head are not commanded.
Each replay frame is captured from the four VLA cameras and saved as JPEGs.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Optional, Sequence

MODEL_DIM = 23
CMD_DIM = 16  # right_arm(7) + right_gripper(1) + left_arm(7) + left_gripper(1)

RIGHT_ARM = slice(0, 7)
RIGHT_GRIPPER = 7
LEFT_ARM = slice(8, 15)
LEFT_GRIPPER = 15
ARM_INDICES = tuple(list(range(0, 7)) + list(range(8, 15)))
GRIPPER_INDICES = (RIGHT_GRIPPER, LEFT_GRIPPER)

DEFAULT_PARQUET_PATH = Path("/home/galbot/project_yuqiz/reply/replay_episode_000.parquet")
DEFAULT_CONFIG_PATH = Path(__file__).with_name("vla_config.yaml")
DEFAULT_COLUMNS = ("action", "observation.state")
REPLAY_JOINT_GROUPS = ("right_arm", "right_gripper", "left_arm", "left_gripper")

GRIPPER_ACTION_SCALE = 0.0012
GRIPPER_MIN_WIDTH_M = 0.0
GRIPPER_MAX_WIDTH_M = 0.12
GRIPPER_VELOCITY = 0.15
GRIPPER_EFFORT = 10.0
ARM_MAX_SPEED = 0.15
GRIPPER_MAX_SPEED = 0.1
MOVE_TIMEOUT_S = 45.0
GRIPPER_SETTLE_S = 2.0
GRIPPER_OPEN_WIDTH_M = 0.12
JUMP_WARN_RAD = 0.08
TIME_FROM_START_S = 0.0
DEFAULT_FPS = 30.0
CAMERA_WARMUP_S = 5.0
CAMERA_NAMES = ("head_left", "head_right", "left_arm", "right_arm")
DEFAULT_IMAGE_ROOT = Path(__file__).resolve().parent / "replay_images"
STATE_SAMPLE_EVERY = 15
RIGHT_JOINT4 = 3
LEFT_JOINT4 = 11
RIGHT_JOINT2 = 1
LEFT_JOINT2 = 9
RIGHT_DESCENT_LIMIT = 1.178
LEFT_DESCENT_LIMIT = -1.248
ELBOW_DESCENT_EPS = 0.05
TRACK_JOINTS = {
    "right_arm_joint2": RIGHT_JOINT2,
    "right_arm_joint4": RIGHT_JOINT4,
    "right_arm_gripper": RIGHT_GRIPPER,
    "left_arm_joint2": LEFT_JOINT2,
    "left_arm_joint4": LEFT_JOINT4,
    "left_arm_gripper": LEFT_GRIPPER,
}

# Keep in sync with reply/init_position/set_init_*_positions.py
INIT_LEG = [0.495931, 1.230984, 0.801337, 0.065899, -0.003332]
INIT_HEAD = [-0.000024, 0.458594]
INIT_LEFT_ARM = [0.835587, -0.895822, -0.366071, -1.918482, 0.419088, 0.479749, 1.538200]
INIT_RIGHT_ARM = [-0.572941, 0.791077, 0.144674, 2.023465, -0.470260, -0.732356, -1.538127]

# Fallback copy of vla_config.yaml robot.control.joint_limits (23 x [lo, hi]).
DEFAULT_JOINT_LIMITS = (
    (-1.83, 0.11),
    (0.29, 1.6081),
    (-0.84, 1.17),
    (1.178, 2.47),
    (-0.76, 1.04),
    (-0.7354, 0.8227),
    (-1.5382, 0.61),
    (0.0, 100.0),
    (0.0, 1.45),
    (-1.6081, -0.27),
    (-0.77, 0.74),
    (-2.38, -1.248),
    (-1.16, 0.81),
    (-0.64, 0.7354),
    (-1.26, 1.5382),
    (0.0, 100.0),
    (0.0, 0.9374),
    (0.0, 2.5847),
    (0.0, 2.3262),
    (-1.5906, 1.5906),
    (-0.1645, 0.1645),
    (-1.5208, 1.5208),
    (-0.2143, 0.4936),
)


def load_galbot_sdk():
    try:
        from galbot_sdk.g1 import (
            ControlStatus,
            GalbotRobot,
            G1JointGroup,
            JointCommand,
            SensorType,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing galbot_sdk. Install/configure GalbotSDK before sending commands. "
            "Use --dry-run to inspect parquet without the SDK."
        ) from exc
    return ControlStatus, GalbotRobot, G1JointGroup, JointCommand, SensorType


def camera_sensors(SensorType) -> dict:
    return {
        "head_left": SensorType.HEAD_LEFT_CAMERA,
        "head_right": SensorType.HEAD_RIGHT_CAMERA,
        "left_arm": SensorType.LEFT_ARM_CAMERA,
        "right_arm": SensorType.RIGHT_ARM_CAMERA,
    }


def decode_rgb_bgr(compressed) -> Optional["object"]:
    if not compressed or "data" not in compressed:
        return None
    try:
        import cv2
        import numpy as np
    except ModuleNotFoundError:
        return None
    img = cv2.imdecode(np.frombuffer(compressed["data"], np.uint8), cv2.IMREAD_COLOR)
    return img


class ImageWriter:
    """Decode-ready BGR frames are written off the control loop."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue = queue.Queue(maxsize=128)
        self._stop = object()
        self.dropped = 0
        self.saved = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, frame: int, camera_name: str, img) -> None:
        try:
            self._queue.put_nowait((frame, camera_name, img))
        except queue.Full:
            self.dropped += 1

    def _run(self) -> None:
        try:
            import cv2
        except ModuleNotFoundError:
            print("opencv-python missing; cannot save replay images")
            return
        while True:
            item = self._queue.get()
            if item is self._stop:
                break
            frame, camera_name, img = item
            path = self.out_dir / f"step_{frame:04d}_{camera_name}.jpg"
            if cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 85]):
                self.saved += 1

    def close(self) -> None:
        self._queue.put(self._stop)
        self._thread.join(timeout=120)


def load_dataframe(parquet_path: Path):
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing parquet dependency. Install pandas and a parquet engine, for example: "
            "pip install pandas pyarrow"
        ) from exc

    if parquet_path.is_file():
        return pd.read_parquet(parquet_path)

    files = sorted(parquet_path.glob("data/**/*.parquet"))
    if not files:
        files = sorted(parquet_path.glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {parquet_path}")
    return pd.concat((pd.read_parquet(p) for p in files), ignore_index=True)


def load_joint_limits(config_path: Path) -> list[tuple[float, float]]:
    if not config_path.exists():
        print(f"Config not found ({config_path}), using built-in joint limits")
        return list(DEFAULT_JOINT_LIMITS)
    try:
        import yaml
    except ModuleNotFoundError:
        print("PyYAML not installed, using built-in joint limits")
        return list(DEFAULT_JOINT_LIMITS)

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    limits = ((cfg.get("robot") or {}).get("control") or {}).get("joint_limits")
    if not limits or len(limits) != MODEL_DIM:
        print(f"Invalid joint_limits in {config_path}, using built-in")
        return list(DEFAULT_JOINT_LIMITS)
    return [(float(lo), float(hi)) for lo, hi in limits]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay dual-arm + dual-gripper commands from a LeRobot parquet file."
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=DEFAULT_PARQUET_PATH,
        help="Parquet file or LeRobot dataset directory.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="vla_config.yaml used for joint limits.",
    )
    parser.add_argument(
        "--column",
        choices=DEFAULT_COLUMNS,
        default=None,
        help="Column to replay. Defaults to action when present, otherwise observation.state.",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Replay only this episode_index. Defaults to 0.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override replay rate. Default: median timestamp interval, else 30.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print schema, dimensions, and the first frames without moving the robot.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive safety confirmation before sending commands.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Directory for per-frame 4-camera JPEGs. Default: reply/replay_images/<timestamp>.",
    )
    parser.add_argument(
        "--image-every",
        type=int,
        default=15,
        help="Save cameras every N frames (default 15 ≈ 2 Hz at 30 fps). Also saves first and last frame.",
    )
    parser.add_argument(
        "--no-save-images",
        action="store_true",
        help="Do not initialize cameras or save images.",
    )
    parser.add_argument(
        "--no-restore",
        action="store_true",
        help="Do not return to init_position / open grippers after replay.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Replay the same episode this many times (default 1). SDK init happens once.",
    )
    parser.add_argument(
        "--state-every",
        type=int,
        default=STATE_SAMPLE_EVERY,
        help="Read joint state every N frames for tracking RMSE (default 15).",
    )
    return parser.parse_args()


def resolve_parquet_path(path: Path) -> Path:
    if path.exists():
        return path
    raise FileNotFoundError(f"Parquet path not found: {path}")


def select_column(df, requested_column: Optional[str]) -> str:
    if requested_column is not None:
        if requested_column not in df.columns:
            raise KeyError(f"Column not found in parquet: {requested_column}")
        return requested_column
    for column in DEFAULT_COLUMNS:
        if column in df.columns:
            return column
    raise KeyError(
        f"None of the expected columns exist in parquet: {', '.join(DEFAULT_COLUMNS)}"
    )


def filter_episode(df, requested_episode: Optional[int]):
    if requested_episode is None:
        return df
    if "episode_index" not in df.columns:
        print("No episode_index column; replaying all rows in the parquet file")
        return df
    available = sorted(int(ep) for ep in df["episode_index"].dropna().unique())
    filtered = df[df["episode_index"] == requested_episode]
    if filtered.empty:
        raise ValueError(
            f"Episode {requested_episode} not found in parquet. "
            f"Available episodes: {available}"
        )
    return filtered.reset_index(drop=True)


def as_float_list(action) -> list[float]:
    if hasattr(action, "tolist"):
        action = action.tolist()
    try:
        values = list(action)
    except TypeError as exc:
        raise ValueError(f"Action must be a sequence, got {type(action).__name__}") from exc
    return [float(v) for v in values]


def extract_model_action(action) -> list[float]:
    values = as_float_list(action)
    if len(values) != MODEL_DIM:
        raise ValueError(
            f"Unsupported action dimension {len(values)}. Expected {MODEL_DIM} "
            "(right_arm 7 + right_gripper 1 + left_arm 7 + left_gripper 1 + leg 5 + head 2)."
        )
    return values


def clip_to_joint_limits(
    values: Sequence[float], limits: Sequence[tuple[float, float]]
) -> list[float]:
    clipped = []
    for i, value in enumerate(values):
        lo, hi = limits[i]
        clipped.append(min(max(float(value), lo), hi))
    return clipped


def gripper_width_m(value: float) -> float:
    """Convert dataset gripper units (0~100) to SDK meters."""
    return min(max(float(value) * GRIPPER_ACTION_SCALE, GRIPPER_MIN_WIDTH_M), GRIPPER_MAX_WIDTH_M)


def infer_control_interval(timestamps: Iterable, fps_override: Optional[float]) -> float:
    if fps_override is not None:
        if fps_override <= 0:
            raise ValueError(f"--fps must be positive, got {fps_override}")
        return 1.0 / float(fps_override)
    stamps = list(timestamps)
    if len(stamps) <= 1:
        return 1.0 / DEFAULT_FPS
    diffs = []
    for i in range(1, len(stamps)):
        dt = float(stamps[i]) - float(stamps[i - 1])
        if dt > 0:
            diffs.append(dt)
    if not diffs:
        return 1.0 / DEFAULT_FPS
    return float(median(diffs))


def split_cmd_groups(values: Sequence[float]) -> tuple[list[float], float, list[float], float]:
    return (
        list(values[RIGHT_ARM]),
        float(values[RIGHT_GRIPPER]),
        list(values[LEFT_ARM]),
        float(values[LEFT_GRIPPER]),
    )


def build_joint_commands(clipped_23: Sequence[float], JointCommand) -> list:
    commands = [JointCommand() for _ in range(CMD_DIM)]
    cmd_values = list(clipped_23[:CMD_DIM])
    for j, cmd in enumerate(commands):
        if j in (7, 15):
            cmd.position = gripper_width_m(cmd_values[j])
            cmd.velocity = GRIPPER_VELOCITY
            cmd.effort = GRIPPER_EFFORT
        else:
            cmd.position = float(cmd_values[j])
            cmd.velocity = 0.0
            cmd.effort = 0.0
    return commands


def warn_arm_jumps(prev: Sequence[float], current: Sequence[float], frame: int) -> None:
    for j in ARM_INDICES:
        delta = abs(float(current[j]) - float(prev[j]))
        if delta > JUMP_WARN_RAD:
            print(
                f"warning: frame {frame} arm joint {j} jumped {delta:.4f} rad "
                f"(>{JUMP_WARN_RAD}); trajectory not modified"
            )


def summarize_actions(actions: Sequence[Sequence[float]]) -> None:
    import numpy as np

    arr = np.asarray(actions, dtype=float)
    print(f"frames={len(arr)} dim={arr.shape[1]}")
    print(
        "first frame 23-dim: "
        + ", ".join(f"{v:.4f}" for v in arr[0].tolist())
    )
    rg = arr[:, RIGHT_GRIPPER]
    lg = arr[:, LEFT_GRIPPER]
    print(
        f"gripper dataset units: right min={rg.min():.3f} max={rg.max():.3f}, "
        f"left min={lg.min():.3f} max={lg.max():.3f}"
    )
    print(
        f"gripper SDK meters (*{GRIPPER_ACTION_SCALE}): "
        f"right [{gripper_width_m(rg.min()):.4f}, {gripper_width_m(rg.max()):.4f}], "
        f"left [{gripper_width_m(lg.min()):.4f}, {gripper_width_m(lg.max()):.4f}]"
    )
    preview = min(3, len(arr))
    for i in range(preview):
        right_arm, rg_v, left_arm, lg_v = split_cmd_groups(arr[i])
        print(
            f"preview[{i}] right_arm={['%.3f' % v for v in right_arm]} "
            f"rg={rg_v:.3f}->{gripper_width_m(rg_v):.4f}m "
            f"left_arm={['%.3f' % v for v in left_arm]} "
            f"lg={lg_v:.3f}->{gripper_width_m(lg_v):.4f}m"
        )


def capture_cameras(robot, sensors: dict) -> dict:
    images = {}
    for name, sensor in sensors.items():
        try:
            images[name] = decode_rgb_bgr(robot.get_rgb_data(sensor))
        except Exception as exc:
            print(f"warning: get_rgb_data({name}) failed: {exc}")
            images[name] = None
    return images


def warmup_cameras(robot, sensors: dict, seconds: float = CAMERA_WARMUP_S) -> None:
    print(f"Waiting {seconds:.0f}s for camera streams...")
    time.sleep(seconds)
    got = []
    for name, sensor in sensors.items():
        img = None
        for _ in range(8):
            img = decode_rgb_bgr(robot.get_rgb_data(sensor))
            if img is not None:
                break
            time.sleep(0.3)
        if img is None:
            print(f"warning: {name} produced no frame during warmup")
        else:
            got.append(f"{name}{tuple(img.shape)}")
    print("camera warmup: " + (", ".join(got) if got else "no frames"))


def confirm_safety() -> None:
    print(
        "Safety: e-stop unlocked, workspace clear, and no other G1 SDK process "
        "(server.py / infer_server.py) is running."
    )
    print("set_joint_commands will not interpolate; frame 0 is moved slowly first.")
    answer = input("Type yes to start replay: ").strip().lower()
    if answer not in ("y", "yes"):
        raise SystemExit("Replay cancelled")


def move_to_first_frame(robot, ControlStatus, G1JointGroup, first_23: Sequence[float]) -> None:
    right_arm, rg, left_arm, lg = split_cmd_groups(first_23)
    rg_m = gripper_width_m(rg)
    lg_m = gripper_width_m(lg)

    arm_moves = (
        (right_arm, ["right_arm"], "right_arm"),
        (left_arm, ["left_arm"], "left_arm"),
    )
    for positions, groups, label in arm_moves:
        print(f"Slow move {label} via set_joint_positions: {positions}")
        status = robot.set_joint_positions(
            list(positions),
            groups,
            [],
            True,
            ARM_MAX_SPEED,
            MOVE_TIMEOUT_S,
        )
        if status != ControlStatus.SUCCESS:
            raise RuntimeError(f"set_joint_positions failed for {label}: {status}")

    # Fully closed (0.0 m) often never reports arrival, so blocking
    # set_joint_positions times out. Use the gripper API and do not abort replay.
    gripper_targets = (
        ("right_gripper", rg_m),
        ("left_gripper", lg_m),
    )
    for name, width_m in gripper_targets:
        group = name if G1JointGroup is None else getattr(G1JointGroup, name)
        print(f"Move {name} via set_gripper_command: {width_m:.4f} m")
        status = robot.set_gripper_command(
            group,
            width_m,
            GRIPPER_MAX_SPEED,
            GRIPPER_EFFORT,
            False,
        )
        if status != ControlStatus.SUCCESS:
            print(f"warning: set_gripper_command {name} status={status}; continuing")
    time.sleep(GRIPPER_SETTLE_S)


def _set_group_positions(robot, ControlStatus, positions, groups, label, max_speed) -> None:
    print(f"Restore {label} via set_joint_positions: {positions}")
    status = robot.set_joint_positions(
        list(positions),
        groups,
        [],
        True,
        max_speed,
        MOVE_TIMEOUT_S,
    )
    if status != ControlStatus.SUCCESS:
        print(f"warning: set_joint_positions failed for {label}: {status}")


def restore_init_pose(robot, ControlStatus, G1JointGroup) -> None:
    """Open grippers, then slowly return to reply/init_position."""
    print("Restoring init_position with grippers open")
    gripper_targets = (
        ("right_gripper", GRIPPER_OPEN_WIDTH_M),
        ("left_gripper", GRIPPER_OPEN_WIDTH_M),
    )
    for name, width_m in gripper_targets:
        group = name if G1JointGroup is None else getattr(G1JointGroup, name)
        print(f"Open {name} via set_gripper_command: {width_m:.4f} m")
        status = robot.set_gripper_command(
            group,
            width_m,
            GRIPPER_MAX_SPEED,
            GRIPPER_EFFORT,
            False,
        )
        if status != ControlStatus.SUCCESS:
            print(f"warning: set_gripper_command {name} status={status}; continuing")
    time.sleep(GRIPPER_SETTLE_S)

    _set_group_positions(
        robot, ControlStatus, INIT_RIGHT_ARM, ["right_arm"], "right_arm", ARM_MAX_SPEED
    )
    _set_group_positions(
        robot, ControlStatus, INIT_LEFT_ARM, ["left_arm"], "left_arm", ARM_MAX_SPEED
    )
    _set_group_positions(robot, ControlStatus, INIT_HEAD, ["head"], "head", ARM_MAX_SPEED)
    _set_group_positions(robot, ControlStatus, INIT_LEG, ["leg"], "leg", ARM_MAX_SPEED)
    print("Restore complete")


def gripper_state_to_vla(position: float) -> float:
    """SDK meters -> dataset 0~100. Values already in 0~100 pass through."""
    position = float(position)
    if abs(position) <= GRIPPER_MAX_WIDTH_M * 1.5:
        return min(max(position / GRIPPER_MAX_WIDTH_M * 100.0, 0.0), 100.0)
    return min(max(position, 0.0), 100.0)


def flatten_joint_positions(joint_states) -> Optional[list[float]]:
    values: list[float] = []
    for js in joint_states or []:
        pos = getattr(js, "position", js)
        if hasattr(pos, "tolist"):
            pos = pos.tolist()
        if isinstance(pos, (list, tuple)):
            values.extend(float(x) for x in pos)
        else:
            values.append(float(pos))
    if len(values) != CMD_DIM:
        return None
    values[RIGHT_GRIPPER] = gripper_state_to_vla(values[RIGHT_GRIPPER])
    values[LEFT_GRIPPER] = gripper_state_to_vla(values[LEFT_GRIPPER])
    return values


def read_arm_state_16(robot, G1JointGroup) -> Optional[list[float]]:
    if robot is None or G1JointGroup is None:
        return None
    try:
        groups = [
            G1JointGroup.right_arm,
            G1JointGroup.right_gripper,
            G1JointGroup.left_arm,
            G1JointGroup.left_gripper,
        ]
        return flatten_joint_positions(robot.get_joint_states(groups, []))
    except Exception as exc:
        print(f"warning: get_joint_states failed: {exc}")
        return None


def status_kind(status, ControlStatus) -> str:
    if status is None:
        return "unknown"
    if ControlStatus is not None and status == ControlStatus.SUCCESS:
        return "SUCCESS"
    name = str(getattr(status, "name", status)).upper()
    if "FAULT" in name:
        return "FAULT"
    return name


def elbow_report(series: Sequence[float], *, right: bool) -> dict:
    values = [float(v) for v in series]
    start = values[0]
    mn, mx, last = min(values), max(values), values[-1]
    if right:
        extreme = mn
        descent = start - extreme
        descended = extreme < start - ELBOW_DESCENT_EPS
        reached_limit = extreme <= RIGHT_DESCENT_LIMIT + 0.02
        limit = RIGHT_DESCENT_LIMIT
    else:
        extreme = mx
        descent = extreme - start
        descended = extreme > start + ELBOW_DESCENT_EPS
        reached_limit = extreme >= LEFT_DESCENT_LIMIT - 0.02
        limit = LEFT_DESCENT_LIMIT
    return {
        "start": start,
        "min": mn,
        "max": mx,
        "last": last,
        "extreme_in_descent_direction": extreme,
        "descent_from_start": descent,
        "descended": descended,
        "reached_training_descent_limit": reached_limit,
        "training_descent_limit": limit,
    }


def tracking_rmse(samples: Sequence[Sequence[float]]) -> dict:
    if not samples:
        return {"rmse_mean": None, "rmse_max": None, "n_samples": 0, "by_joint": {}}
    per_sample = []
    by_joint_sq = {name: [] for name in TRACK_JOINTS}
    for err in samples:
        sq = [float(v) * float(v) for v in err]
        per_sample.append(math.sqrt(sum(sq) / len(sq)))
        for name, idx in TRACK_JOINTS.items():
            by_joint_sq[name].append(sq[idx])
    return {
        "rmse_mean": mean(per_sample),
        "rmse_max": max(per_sample),
        "n_samples": len(per_sample),
        "by_joint": {
            name: math.sqrt(mean(vals)) if vals else None
            for name, vals in by_joint_sq.items()
        },
    }


def commanded_elbow_reports(clipped: Sequence[Sequence[float]]) -> dict:
    return {
        "right_arm_joint4": elbow_report([row[RIGHT_JOINT4] for row in clipped], right=True),
        "left_arm_joint4": elbow_report([row[LEFT_JOINT4] for row in clipped], right=False),
    }


def write_consistency_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def print_consistency_summary(runs: Sequence[dict]) -> None:
    print("\n=== replay consistency ===")
    for item in runs:
        track = item.get("tracking") or {}
        right = (item.get("commanded_elbow") or {}).get("right_arm_joint4") or {}
        left = (item.get("commanded_elbow") or {}).get("left_arm_joint4") or {}
        print(
            f"run {item.get('run', '?')}: fault={item.get('fault_count', 0)} "
            f"non_success={item.get('non_success_count', 0)} "
            f"first_fault={item.get('first_fault_frame')} "
            f"rmse_mean={track.get('rmse_mean')} rmse_max={track.get('rmse_max')} "
            f"right_j4_descended={right.get('descended')} "
            f"left_j4_descended={left.get('descended')}"
        )
    faults = [int(item.get("fault_count") or 0) for item in runs]
    rmses = [item.get("tracking", {}).get("rmse_mean") for item in runs]
    rmses = [v for v in rmses if isinstance(v, (int, float))]
    print(
        f"FAULT per run: min={min(faults) if faults else None} "
        f"max={max(faults) if faults else None}"
    )
    if rmses:
        print(f"tracking RMSE mean across runs: {mean(rmses):.4f} (max run {max(rmses):.4f})")


def replay(
    robot,
    ControlStatus,
    G1JointGroup,
    JointCommand,
    parquet_path: Path,
    config_path: Path,
    requested_column: Optional[str],
    requested_episode: Optional[int],
    fps_override: Optional[float],
    dry_run: bool,
    skip_confirm: bool,
    sensors: Optional[dict] = None,
    image_dir: Optional[Path] = None,
    image_every: int = 1,
    restore: bool = True,
    run_index: int = 0,
    state_every: int = STATE_SAMPLE_EVERY,
    preloaded: Optional[dict] = None,
) -> dict:
    if preloaded is None:
        parquet_path = resolve_parquet_path(parquet_path)
        limits = load_joint_limits(config_path)
        df = load_dataframe(parquet_path)
        df = filter_episode(df, requested_episode)
        column = select_column(df, requested_column)
        raw_actions = df[column].tolist()
        if not raw_actions:
            print("No action data found in parquet file")
            return {"run": run_index, "error": "no_actions"}
        actions = [extract_model_action(row) for row in raw_actions]
        timestamps = df["timestamp"].tolist() if "timestamp" in df.columns else []
        dt = infer_control_interval(timestamps, fps_override)
        clipped = [clip_to_joint_limits(row, limits) for row in actions]
        n_clipped = sum(
            1
            for original, limited in zip(actions, clipped)
            if any(abs(a - b) > 1e-6 for a, b in zip(original, limited))
        )
        episode_label = requested_episode if requested_episode is not None else "all"
        print(
            f"Replay: file={parquet_path}, column={column}, episode={episode_label}, "
            f"frames={len(actions)}, dt={dt:.4f}s ({1.0 / dt:.1f} Hz), dry_run={dry_run}"
        )
        print(f"parquet columns: {list(df.columns)}")
        summarize_actions(actions)
        if n_clipped:
            print(f"joint_limits clipped {n_clipped}/{len(actions)} frames (parquet unchanged)")
    else:
        clipped = preloaded["clipped"]
        dt = preloaded["dt"]
        n_clipped = preloaded["n_clipped"]

    elbow = commanded_elbow_reports(clipped)
    metrics = {
        "run": run_index,
        "frames": len(clipped),
        "dt": dt,
        "joint_limits_clipped_frames": n_clipped,
        "commanded_elbow": elbow,
        "non_success_count": 0,
        "fault_count": 0,
        "first_fail_frame": None,
        "first_fault_frame": None,
        "image_dir": str(image_dir) if image_dir is not None else None,
        "tracking": {"rmse_mean": None, "rmse_max": None, "n_samples": 0, "by_joint": {}},
    }
    print(
        f"commanded elbow: right_j4 start={elbow['right_arm_joint4']['start']:.3f} "
        f"min={elbow['right_arm_joint4']['min']:.3f} descended={elbow['right_arm_joint4']['descended']}; "
        f"left_j4 start={elbow['left_arm_joint4']['start']:.3f} "
        f"max={elbow['left_arm_joint4']['max']:.3f} descended={elbow['left_arm_joint4']['descended']}"
    )

    if dry_run:
        print("dry-run complete; no commands sent")
        return metrics

    if not skip_confirm:
        confirm_safety()

    if G1JointGroup is None:
        joint_groups = list(REPLAY_JOINT_GROUPS)
    else:
        joint_groups = [getattr(G1JointGroup, name) for name in REPLAY_JOINT_GROUPS]

    writer = None
    if sensors and image_dir is not None:
        writer = ImageWriter(image_dir)
        print(f"Saving 4-camera JPEGs every {image_every} frame(s) to {image_dir}")

    print(f"run {run_index}: moving slowly to frame 0")
    should_restore = restore
    missing = {name: 0 for name in CAMERA_NAMES}
    errors: list[list[float]] = []
    measured_j4_right: list[float] = []
    measured_j4_left: list[float] = []
    state_every = max(1, int(state_every))
    try:
        move_to_first_frame(robot, ControlStatus, G1JointGroup, clipped[0])
        time.sleep(0.5)

        start_time = time.time()
        prev = clipped[0]
        last_frame = len(clipped) - 1
        for i, target in enumerate(clipped):
            warn_arm_jumps(prev, target, i)
            elapsed = time.time() - start_time
            if i == 0 or i == last_frame or i % 30 == 0:
                print(f"run {run_index} frame {i}/{last_frame} t={elapsed:.2f}s")

            status = robot.set_joint_commands(
                build_joint_commands(target, JointCommand),
                joint_groups,
                [],
                TIME_FROM_START_S,
            )
            kind = status_kind(status, ControlStatus)
            if kind != "SUCCESS":
                metrics["non_success_count"] += 1
                if metrics["first_fail_frame"] is None:
                    metrics["first_fail_frame"] = i
                print(f"warning: set_joint_commands status={status} at frame {i}")
                if kind == "FAULT":
                    metrics["fault_count"] += 1
                    if metrics["first_fault_frame"] is None:
                        metrics["first_fault_frame"] = i

            if i == 0 or i == last_frame or i % state_every == 0:
                measured = read_arm_state_16(robot, G1JointGroup)
                if measured is not None:
                    cmd16 = list(target[:CMD_DIM])
                    errors.append([measured[j] - cmd16[j] for j in range(CMD_DIM)])
                    measured_j4_right.append(measured[RIGHT_JOINT4])
                    measured_j4_left.append(measured[LEFT_JOINT4])

            if writer is not None and (i == 0 or i == last_frame or i % image_every == 0):
                frames = capture_cameras(robot, sensors)
                for name, img in frames.items():
                    if img is None:
                        missing[name] += 1
                    else:
                        writer.submit(i, name, img)

            prev = target
            time.sleep(dt)
    finally:
        if writer is not None:
            writer.close()
            print(
                f"images saved={writer.saved} dropped={writer.dropped} "
                f"missing={dict(missing)} dir={image_dir}"
            )
            metrics["images_saved"] = writer.saved
            metrics["images_dropped"] = writer.dropped
            metrics["images_missing"] = dict(missing)
        if should_restore:
            try:
                restore_init_pose(robot, ControlStatus, G1JointGroup)
            except Exception as exc:
                print(f"warning: restore_init_pose failed: {exc}")
                metrics["restore_error"] = str(exc)

    metrics["tracking"] = tracking_rmse(errors)
    if measured_j4_right:
        metrics["measured_elbow"] = {
            "right_arm_joint4": elbow_report(measured_j4_right, right=True),
            "left_arm_joint4": elbow_report(measured_j4_left, right=False),
        }
    print(f"run {run_index} end: fault={metrics['fault_count']} non_success={metrics['non_success_count']}")
    return metrics


def load_replay_payload(
    parquet_path: Path,
    config_path: Path,
    requested_column: Optional[str],
    requested_episode: Optional[int],
    fps_override: Optional[float],
) -> dict:
    parquet_path = resolve_parquet_path(parquet_path)
    limits = load_joint_limits(config_path)
    df = load_dataframe(parquet_path)
    df = filter_episode(df, requested_episode)
    column = select_column(df, requested_column)
    raw_actions = df[column].tolist()
    if not raw_actions:
        raise ValueError("No action data found in parquet file")
    actions = [extract_model_action(row) for row in raw_actions]
    timestamps = df["timestamp"].tolist() if "timestamp" in df.columns else []
    dt = infer_control_interval(timestamps, fps_override)
    clipped = [clip_to_joint_limits(row, limits) for row in actions]
    n_clipped = sum(
        1
        for original, limited in zip(actions, clipped)
        if any(abs(a - b) > 1e-6 for a, b in zip(original, limited))
    )
    episode_label = requested_episode if requested_episode is not None else "all"
    print(
        f"Replay: file={parquet_path}, column={column}, episode={episode_label}, "
        f"frames={len(actions)}, dt={dt:.4f}s ({1.0 / dt:.1f} Hz)"
    )
    print(f"parquet columns: {list(df.columns)}")
    summarize_actions(actions)
    if n_clipped:
        print(f"joint_limits clipped {n_clipped}/{len(actions)} frames (parquet unchanged)")
    return {
        "clipped": clipped,
        "dt": dt,
        "n_clipped": n_clipped,
        "parquet": str(parquet_path),
        "column": column,
        "episode": episode_label,
        "frames": len(clipped),
    }


def main() -> None:
    args = parse_args()
    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")

    payload = load_replay_payload(
        args.parquet, args.config, args.column, args.episode, args.fps
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = args.image_dir if args.image_dir is not None else DEFAULT_IMAGE_ROOT / stamp
    session_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        dry = replay(
            robot=None,
            ControlStatus=None,
            G1JointGroup=None,
            JointCommand=None,
            parquet_path=args.parquet,
            config_path=args.config,
            requested_column=args.column,
            requested_episode=args.episode,
            fps_override=args.fps,
            dry_run=True,
            skip_confirm=True,
            preloaded=payload,
        )
        report = {
            "session_dir": str(session_dir),
            "repeat": args.repeat,
            "dry_run": True,
            "parquet": payload["parquet"],
            "episode": payload["episode"],
            "frames": payload["frames"],
            "runs": [dry],
        }
        write_consistency_json(session_dir / "consistency.json", report)
        print(f"wrote {session_dir / 'consistency.json'}")
        return

    ControlStatus, GalbotRobot, G1JointGroup, JointCommand, SensorType = load_galbot_sdk()
    robot = GalbotRobot()
    save_images = not args.no_save_images
    sensors = camera_sensors(SensorType) if save_images else None
    if save_images:
        ok = robot.init(set(sensors.values()))
    else:
        ok = robot.init()
    if not ok:
        print("System initialization failed")
        return
    print("System initialized successfully")
    if save_images:
        warmup_cameras(robot, sensors)
    else:
        time.sleep(2)

    image_every = max(1, int(args.image_every))
    runs = []
    try:
        if not args.yes:
            confirm_safety()
        for run_index in range(args.repeat):
            run_dir = session_dir / f"run_{run_index:02d}"
            print(f"\n=== replay run {run_index + 1}/{args.repeat} -> {run_dir} ===")
            metrics = replay(
                robot,
                ControlStatus,
                G1JointGroup,
                JointCommand,
                args.parquet,
                args.config,
                args.column,
                args.episode,
                args.fps,
                dry_run=False,
                skip_confirm=True,
                sensors=sensors if save_images else None,
                image_dir=run_dir if save_images else None,
                image_every=image_every,
                restore=not args.no_restore,
                run_index=run_index,
                state_every=args.state_every,
                preloaded=payload,
            )
            runs.append(metrics)
            report = {
                "session_dir": str(session_dir),
                "repeat": args.repeat,
                "dry_run": False,
                "parquet": payload["parquet"],
                "episode": payload["episode"],
                "frames": payload["frames"],
                "runs": runs,
            }
            write_consistency_json(session_dir / "consistency.json", report)
        print_consistency_summary(runs)
        print(f"wrote {session_dir / 'consistency.json'}")
    finally:
        robot.request_shutdown()
        robot.wait_for_shutdown()
        robot.destroy()
        print("Program exited")


if __name__ == "__main__":
    main()
