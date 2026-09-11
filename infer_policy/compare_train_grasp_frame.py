#!/usr/bin/env python3
"""Compare an inference run against training grasp kinematics.

Training videos are not always on the robot. This script still answers the
plan question using:

1. Training start pose (episode-0 median in vla_config.yaml)
2. Dataset/manual elbow descent limits used as joint_limits
3. Optional local LeRobot parquet if --dataset is given
4. Inference step_*.json from a saved image run

Right arm joint4 lowers as the angle *decreases* toward 1.178.
Left arm joint4 lowers as the angle *increases* toward -1.248.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

from alignment_schema import INIT_JOINT_POSITIONS


# Episode-0 median shared with vla_config.yaml. Import the pinned values instead
# of maintaining another hard-coded copy that can silently become stale.
TRAINING_START = {
    "right_arm_joint4": INIT_JOINT_POSITIONS["right_arm"][3],
    "left_arm_joint4": INIT_JOINT_POSITIONS["left_arm"][3],
    "right_arm_gripper": INIT_JOINT_POSITIONS["right_gripper"][0],
    "left_arm_gripper": INIT_JOINT_POSITIONS["left_gripper"][0],
}

# vla_config joint_limits: dataset extrema expanded, then manual descent pose.
RIGHT_DESCENT_LIMIT = 1.178  # right joint4 smaller = into the bin
LEFT_DESCENT_LIMIT = -1.248  # left joint4 larger (less negative) = into the bin

GRIPPER_OPEN = 50.0
GRIPPER_CLOSED = 10.0


def load_inference_steps(run_dir: str) -> List[Dict[str, Any]]:
    paths = sorted(glob(os.path.join(run_dir, "step_*.json")))
    steps = []
    for path in paths:
        name = os.path.basename(path)
        try:
            step = int(name.split("_")[1].split(".")[0])
        except (IndexError, ValueError):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["_step"] = step
        payload["_path"] = path
        steps.append(payload)
    steps.sort(key=lambda item: item["_step"])
    return steps


def _closing_events(steps: List[Dict[str, Any]], gripper_key: str) -> List[Dict[str, Any]]:
    events = []
    prev = None
    for item in steps:
        value = item.get(gripper_key)
        if value is None:
            continue
        if prev is not None and prev.get(gripper_key) is not None:
            if prev[gripper_key] >= GRIPPER_OPEN and value <= GRIPPER_CLOSED:
                events.append(item)
        prev = item
    return events


def _series(steps: List[Dict[str, Any]], key: str) -> List[Tuple[int, float]]:
    out = []
    for item in steps:
        value = item.get(key)
        if value is None:
            continue
        out.append((int(item["_step"]), float(value)))
    return out


def summarize_arm(steps: List[Dict[str, Any]], joint_key: str, gripper_key: str, descend_sign: int, start: float, limit: float) -> Dict[str, Any]:
    joints = _series(steps, joint_key)
    grippers = _series(steps, gripper_key)
    closes = _closing_events(steps, gripper_key)
    values = [value for _, value in joints]
    start_value = values[0] if values else None
    extreme = min(values) if descend_sign < 0 else max(values) if values else None
    descent = None
    if start_value is not None and extreme is not None:
        descent = (start_value - extreme) if descend_sign < 0 else (extreme - start_value)
    reached_limit = False
    if extreme is not None:
        if descend_sign < 0:
            reached_limit = extreme <= limit + 0.05
        else:
            reached_limit = extreme >= limit - 0.05
    close_joints = []
    for event in closes:
        close_joints.append({
            "step": event["_step"],
            joint_key: event.get(joint_key),
            gripper_key: event.get(gripper_key),
        })
    return {
        "start": start_value,
        "training_start": start,
        "extreme_in_descent_direction": extreme,
        "descent_from_start": descent,
        "training_descent_limit": limit,
        "reached_training_descent_limit": reached_limit,
        "gripper_start": grippers[0][1] if grippers else None,
        "gripper_min": min((v for _, v in grippers), default=None),
        "gripper_max": max((v for _, v in grippers), default=None),
        "close_events": close_joints,
        "descended": bool(descent is not None and descent > 0.05),
    }


def inspect_saved_letterbox_jpegs(run_dir: str) -> Dict[str, Any]:
    """Measure black bars on already-saved 224 letterbox JPEGs from a run."""
    try:
        import numpy as np
        from PIL import Image
        from image_preprocessor import count_letterbox_black_bar_rows
    except ImportError as exc:
        return {"error": str(exc)}

    cameras = {}
    for cam in ("head_left", "left_arm", "right_arm"):
        path = os.path.join(run_dir, f"step_0000_{cam}.jpg")
        if not os.path.isfile(path):
            cameras[cam] = {"error": "missing"}
            continue
        arr = np.array(Image.open(path).convert("RGB"))
        top, bottom = count_letterbox_black_bar_rows(arr)
        cameras[cam] = {
            "path": path,
            "shape": list(arr.shape),
            "black_bar_rows": {"top": top, "bottom": bottom},
            "matches_training_28px_bars": top == 28 and bottom == 28,
        }
    return cameras


def analyze_inference_run(run_dir: str) -> Dict[str, Any]:
    steps = load_inference_steps(run_dir)
    if not steps:
        raise FileNotFoundError(f"No step_*.json in {run_dir}")
    right = summarize_arm(
        steps, "right_arm_joint4", "right_arm_gripper", -1,
        TRAINING_START["right_arm_joint4"], RIGHT_DESCENT_LIMIT,
    )
    left = summarize_arm(
        steps, "left_arm_joint4", "left_arm_gripper", 1,
        TRAINING_START["left_arm_joint4"], LEFT_DESCENT_LIMIT,
    )
    gripper_mismatch = (
        (right.get("gripper_start") or 0) > 50
        or (left.get("gripper_start") or 0) > 50
    )
    if not right["descended"] and not left["descended"]:
        verdict = (
            "policy_did_not_descend: inference never moved joint4 toward the "
            "training/manual in-bin limit; this is not a demo-that-grasps-at-rim "
            "story, because training joint_limits were taken from dataset extrema "
            "plus a taught descent pose (right 1.178 / left -1.248)."
        )
    elif (right["descended"] or left["descended"]) and not (
        right["reached_training_descent_limit"] or left["reached_training_descent_limit"]
    ):
        verdict = (
            "partial_descent: elbows moved toward the bin but stopped short of "
            "the training descent pose, consistent with closing in air above the object."
        )
    else:
        verdict = (
            "descended_like_training: at least one elbow reached the training "
            "in-bin limit; if the grasp still missed, look at XY / vision geometry."
        )
    return {
        "run_dir": os.path.abspath(run_dir),
        "n_steps": len(steps),
        "first_step": steps[0]["_step"],
        "last_step": steps[-1]["_step"],
        "gripper_started_open_unlike_training": gripper_mismatch,
        "right": right,
        "left": left,
        "saved_letterbox_jpegs": inspect_saved_letterbox_jpegs(run_dir),
        "verdict": verdict,
    }


def _parquet_files(dataset_dir: str) -> List[str]:
    return sorted(glob(os.path.join(dataset_dir, "**", "*.parquet"), recursive=True))


def analyze_training_dataset(dataset_dir: str, max_episodes: int = 8) -> Dict[str, Any]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to read a LeRobot parquet dataset") from exc

    files = _parquet_files(dataset_dir)
    if not files:
        raise FileNotFoundError(f"No parquet files under {dataset_dir}")

    right_mins = []
    left_maxs = []
    start_right = []
    start_left = []
    n_episodes = 0
    for path in files[:max_episodes]:
        frame = pd.read_parquet(path, columns=None)
        cols = list(frame.columns)
        right_col = next((c for c in cols if "right_arm_joint4" in c), None)
        left_col = next((c for c in cols if "left_arm_joint4" in c), None)
        if right_col is None and "observation.state" in cols:
            # Fallback: 23-D state vector, joint4 at index 3 / 11.
            states = frame["observation.state"].tolist()
            right_vals = [float(row[3]) for row in states if row is not None and len(row) > 11]
            left_vals = [float(row[11]) for row in states if row is not None and len(row) > 11]
        else:
            if right_col is None or left_col is None:
                continue
            right_vals = [float(v) for v in frame[right_col].tolist()]
            left_vals = [float(v) for v in frame[left_col].tolist()]
        if not right_vals:
            continue
        n_episodes += 1
        start_right.append(right_vals[0])
        start_left.append(left_vals[0])
        right_mins.append(min(right_vals))
        left_maxs.append(max(left_vals))

    if n_episodes == 0:
        raise RuntimeError(f"Could not find joint4 columns in parquet under {dataset_dir}")

    median = lambda values: sorted(values)[len(values) // 2]
    return {
        "dataset_dir": os.path.abspath(dataset_dir),
        "episodes_read": n_episodes,
        "right_joint4_start_median": median(start_right),
        "left_joint4_start_median": median(start_left),
        "right_joint4_min_median": median(right_mins),
        "left_joint4_max_median": median(left_maxs),
        "demos_descend_right": median(right_mins) < TRAINING_START["right_arm_joint4"] - 0.05,
        "demos_descend_left": median(left_maxs) > TRAINING_START["left_arm_joint4"] + 0.05,
    }


def default_dataset_candidates() -> List[str]:
    env = os.environ.get("TRAIN_DATASET_DIR")
    candidates = []
    if env:
        candidates.append(env)
    candidates.extend([
        "/home/zhangyuqi/zhangyuqi/data_zhangyq/G1/pick/lerobot",
        "/home/galbot/data_zhangyq/G1/pick/lerobot",
        "/home/galbot/data/G1/pick/lerobot",
    ])
    return [path for path in candidates if os.path.isdir(path)]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "behavior_tree",
            "logs",
            "images",
            "20260907_092522",
        ),
        help="Inference image run directory with step_*.json",
    )
    parser.add_argument("--dataset", default="", help="Optional local LeRobot dataset root")
    parser.add_argument("--out", default="", help="Write JSON report to this path")
    args = parser.parse_args(argv)

    report: Dict[str, Any] = {
        "training_start": TRAINING_START,
        "training_descent_limits": {
            "right_arm_joint4": RIGHT_DESCENT_LIMIT,
            "left_arm_joint4": LEFT_DESCENT_LIMIT,
        },
        "inference": analyze_inference_run(args.run_dir),
        "training_dataset": None,
    }

    dataset_dir = args.dataset or (default_dataset_candidates()[0] if default_dataset_candidates() else "")
    if dataset_dir:
        try:
            report["training_dataset"] = analyze_training_dataset(dataset_dir)
        except Exception as exc:
            report["training_dataset"] = {"error": str(exc), "dataset_dir": dataset_dir}
    else:
        report["training_dataset"] = {
            "error": "training parquet not present on this machine",
            "note": (
                "joint_limits.right_arm_joint4 lower=1.178 and left_arm_joint4 "
                "upper=-1.248 come from dataset extrema plus a taught in-bin pose, "
                "so demos did descend; this inference run can be judged against those limits."
            ),
        }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        out_path = os.path.abspath(args.out)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
