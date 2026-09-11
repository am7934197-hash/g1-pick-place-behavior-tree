#!/usr/bin/env python3
"""手动摆位采集 G1 软件关节限位。

流程：
  1. 慢速把目标机械臂摆到你认为安全的各个边界
  2. 本脚本持续记录每个关节的 min/max
  3. 结束后向内缩安全余量，再与 URDF 硬件限位取交集
  4. 打印可直接粘贴到 vla_config.yaml 的 joint_limits

本脚本只读关节状态，默认不下发运动。手臂需用示教器 / 平板拖动，
或机器人已有的慢速点动来摆。不要与 infer_policy/server.py 同时占用 SDK。

示例：
  python3 record_joint_limits.py --groups right_arm
  python3 record_joint_limits.py --groups right_arm left_arm --margin 0.05
  python3 record_joint_limits.py --groups right_arm --apply
"""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import termios
import time
import tty
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from galbot_sdk.g1 import GalbotRobot


# 与 vla_config.yaml state_feature_names / joint_limits 顺序一致。
STATE_NAMES: List[str] = [
    "right_arm_joint1",
    "right_arm_joint2",
    "right_arm_joint3",
    "right_arm_joint4",
    "right_arm_joint5",
    "right_arm_joint6",
    "right_arm_joint7",
    "right_arm_gripper",
    "left_arm_joint1",
    "left_arm_joint2",
    "left_arm_joint3",
    "left_arm_joint4",
    "left_arm_joint5",
    "left_arm_joint6",
    "left_arm_joint7",
    "left_arm_gripper",
    "leg_joint1",
    "leg_joint2",
    "leg_joint3",
    "leg_joint4",
    "leg_joint5",
    "head_joint1",
    "head_joint2",
]

GROUP_JOINTS: Dict[str, List[str]] = {
    "right_arm": [
        "right_arm_joint1",
        "right_arm_joint2",
        "right_arm_joint3",
        "right_arm_joint4",
        "right_arm_joint5",
        "right_arm_joint6",
        "right_arm_joint7",
    ],
    "left_arm": [
        "left_arm_joint1",
        "left_arm_joint2",
        "left_arm_joint3",
        "left_arm_joint4",
        "left_arm_joint5",
        "left_arm_joint6",
        "left_arm_joint7",
    ],
    "leg": ["leg_joint1", "leg_joint2", "leg_joint3", "leg_joint4", "leg_joint5"],
    "head": ["head_joint1", "head_joint2"],
}

# URDF galbot_one_golf.urdf 硬件限位（rad）。夹爪用 VLA 的 [0, 100]。
HARDWARE_LIMITS: Dict[str, Tuple[float, float]] = {
    "right_arm_joint1": (-3.00432619, 3.00432619),
    "right_arm_joint2": (-1.608062789, 1.608062789),
    "right_arm_joint3": (-2.916972222, 2.916972222),
    "right_arm_joint4": (-1.869862177, 2.5679938779914944),
    "right_arm_joint5": (-2.916972222, 2.916972222),
    "right_arm_joint6": (-0.7353981633974483, 0.8226646259971648),
    "right_arm_joint7": (-1.538202778, 1.538202778),
    "right_arm_gripper": (0.0, 100.0),
    "left_arm_joint1": (-3.00432619, 3.00432619),
    "left_arm_joint2": (-1.608062789, 1.608062789),
    "left_arm_joint3": (-2.916972222, 2.916972222),
    "left_arm_joint4": (-2.5679938779914944, 1.869862177),
    "left_arm_joint5": (-2.916972222, 2.916972222),
    "left_arm_joint6": (-0.8226646259971648, 0.7353981633974483),
    "left_arm_joint7": (-1.538202778, 1.538202778),
    "left_arm_gripper": (0.0, 100.0),
    "leg_joint1": (0.0, 0.9374),
    "leg_joint2": (0.0, 2.5847),
    "leg_joint3": (0.0, 2.3262),
    "leg_joint4": (-1.5906, 1.5906),
    "leg_joint5": (-0.1645, 0.1645),
    "head_joint1": (-1.5208, 1.5208),
    "head_joint2": (-0.2143, 0.4936),
}

EXISTING_LIMITS: Dict[str, Tuple[float, float]] = {
    "right_arm_joint1": (-1.83, 0.11),
    "right_arm_joint2": (0.29, 1.6081),
    "right_arm_joint3": (-0.84, 1.17),
    "right_arm_joint4": (1.10, 2.47),
    "right_arm_joint5": (-0.76, 1.04),
    "right_arm_joint6": (-0.7354, 0.8227),
    "right_arm_joint7": (-1.5382, 0.61),
    "right_arm_gripper": (0.0, 100.0),
    "left_arm_joint1": (0.0, 1.45),
    "left_arm_joint2": (-1.6081, -0.27),
    "left_arm_joint3": (-0.77, 0.74),
    "left_arm_joint4": (-2.38, -0.94),
    "left_arm_joint5": (-1.16, 0.81),
    "left_arm_joint6": (-0.64, 0.7354),
    "left_arm_joint7": (-1.26, 1.5382),
    "left_arm_gripper": (0.0, 100.0),
    "leg_joint1": (0.0, 0.9374),
    "leg_joint2": (0.0, 2.5847),
    "leg_joint3": (0.0, 2.3262),
    "leg_joint4": (-1.5906, 1.5906),
    "leg_joint5": (-0.1645, 0.1645),
    "head_joint1": (-1.5208, 1.5208),
    "head_joint2": (-0.2143, 0.4936),
}

NEAR_HW_RAD = 0.02
DEFAULT_CONFIG = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "infer_policy", "vla_config.yaml")
)


class Envelope:
    def __init__(self, names: Sequence[str]) -> None:
        self.names = list(names)
        self.min_pos = [None] * len(names)  # type: List[Optional[float]]
        self.max_pos = [None] * len(names)  # type: List[Optional[float]]
        self.samples = 0

    def update(self, values: Sequence[float]) -> None:
        if len(values) != len(self.names):
            raise ValueError("joint value count mismatch")
        for i, val in enumerate(values):
            x = float(val)
            if self.min_pos[i] is None or x < self.min_pos[i]:
                self.min_pos[i] = x
            if self.max_pos[i] is None or x > self.max_pos[i]:
                self.max_pos[i] = x
        self.samples += 1

    def reset(self) -> None:
        self.min_pos = [None] * len(self.names)
        self.max_pos = [None] * len(self.names)
        self.samples = 0

    def span(self, i: int) -> float:
        lo, hi = self.min_pos[i], self.max_pos[i]
        if lo is None or hi is None:
            return 0.0
        return float(hi - lo)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="手动摆位采集软件关节限位")
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["right_arm"],
        choices=list(GROUP_JOINTS.keys()),
        help="要采集的关节组，默认右臂",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="采集包络向内缩的安全余量，单位 rad，默认 0.05",
    )
    parser.add_argument(
        "--min-span",
        type=float,
        default=0.15,
        dest="min_span",
        help="关节实际摆过的最小幅度才写入新限位，默认 0.15 rad",
    )
    parser.add_argument("--hz", type=float, default=20.0, help="采样频率")
    parser.add_argument(
        "--output",
        default="",
        help="结果保存路径，默认写到本目录 recorded_joint_limits_*.yaml",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="现有 vla_config.yaml，未摆到的关节沿用其中限位",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="采集结束后把 joint_limits 写回 --config（会先备份）",
    )
    return parser.parse_args()


def expand_names(groups: Iterable[str]) -> List[str]:
    names: List[str] = []
    seen = set()
    for group in groups:
        for name in GROUP_JOINTS[group]:
            if name not in seen:
                names.append(name)
                seen.add(name)
    return names


def read_key(timeout_s: float) -> Optional[str]:
    ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if not ready:
        return None
    ch = sys.stdin.read(1)
    if ch == "\x1b":
        extra = ""
        if select.select([sys.stdin], [], [], 0.0)[0]:
            extra += sys.stdin.read(1)
        if select.select([sys.stdin], [], [], 0.0)[0]:
            extra += sys.stdin.read(1)
        return "esc"
    if ch in ("\n", "\r"):
        return "enter"
    if ch == "\x03":
        raise KeyboardInterrupt
    return ch.lower()


def fmt(x: Optional[float], width: int = 8) -> str:
    if x is None:
        return " " * (width - 1) + "-"
    return "{:{}.4f}".format(x, width)


def render(
    names: Sequence[str],
    current: Sequence[float],
    env: Envelope,
    recording: bool,
    elapsed_s: float,
    margin: float,
) -> str:
    lines = [
        "G1 软件限位采集  |  {}  |  samples={}  |  t={:.1f}s  |  margin={:.3f} rad".format(
            "RECORDING" if recording else "PAUSED",
            env.samples,
            elapsed_s,
            margin,
        ),
        "操作: 慢速摆到安全边界   [space]暂停/继续  [r]清包络  [q]结束并内缩",
        "",
        "{:<20} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}  {}".format(
            "joint", "now", "min", "max", "span", "hw_lo", "hw_hi", "note"
        ),
    ]
    for i, name in enumerate(names):
        hw_lo, hw_hi = HARDWARE_LIMITS[name]
        lo, hi = env.min_pos[i], env.max_pos[i]
        span = env.span(i)
        notes = []
        now = current[i]
        if now <= hw_lo + NEAR_HW_RAD or now >= hw_hi - NEAR_HW_RAD:
            notes.append("near_hw")
        if lo is not None and lo <= hw_lo + NEAR_HW_RAD:
            notes.append("min~hw")
        if hi is not None and hi >= hw_hi - NEAR_HW_RAD:
            notes.append("max~hw")
        lines.append(
            "{:<20} {} {} {} {} {} {}  {}".format(
                name,
                fmt(now),
                fmt(lo),
                fmt(hi),
                fmt(span),
                fmt(hw_lo),
                fmt(hw_hi),
                ",".join(notes),
            )
        )
    return "\n".join(lines)


def shrink_limits(
    names: Sequence[str],
    env: Envelope,
    margin: float,
    min_span: float,
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, str]]:
    results: Dict[str, Tuple[float, float]] = {}
    status: Dict[str, str] = {}
    for i, name in enumerate(names):
        lo = env.min_pos[i]
        hi = env.max_pos[i]
        hw_lo, hw_hi = HARDWARE_LIMITS[name]
        if lo is None or hi is None:
            status[name] = "no_sample"
            continue
        span = hi - lo
        if span < min_span:
            status[name] = "span_too_small ({:.4f} < {:.4f})".format(span, min_span)
            continue
        soft_lo = lo + margin
        soft_hi = hi - margin
        if soft_lo >= soft_hi:
            status[name] = "margin_too_large_for_span ({:.4f})".format(span)
            continue
        soft_lo = max(soft_lo, hw_lo)
        soft_hi = min(soft_hi, hw_hi)
        if soft_lo >= soft_hi:
            status[name] = "empty_after_hardware_clip"
            continue
        results[name] = (round(soft_lo, 4), round(soft_hi, 4))
        flags = []
        if abs(soft_lo - hw_lo) < 1e-4:
            flags.append("hit_hw_lo")
        if abs(soft_hi - hw_hi) < 1e-4:
            flags.append("hit_hw_hi")
        status[name] = "updated" + ((":" + ",".join(flags)) if flags else "")
    return results, status


def format_yaml_limits(limits: Dict[str, Tuple[float, float]]) -> str:
    lines = ["    joint_limits:"]
    for name in STATE_NAMES:
        lo, hi = limits[name]
        # 夹爪保持整数写法，其余保留最多 4 位
        if "gripper" in name:
            lines.append("      - [{:.0f}, {:.0f}]             # {}".format(lo, hi, name))
        else:
            lines.append("      - [{:.4f}, {:.4f}]   # {}".format(lo, hi, name))
    return "\n".join(lines)


def merge_limits(
    recorded: Dict[str, Tuple[float, float]],
) -> Dict[str, Tuple[float, float]]:
    merged = dict(EXISTING_LIMITS)
    merged.update(recorded)
    return merged


def backup_and_apply(config_path: str, yaml_block: str) -> str:
    if not os.path.isfile(config_path):
        raise FileNotFoundError(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        text = f.read()
    start = text.find("    joint_limits:")
    if start < 0:
        raise RuntimeError("config 里找不到 joint_limits")
    # 替换到下一个同级或更高级键之前（action_filter）
    end_marker = "\n    action_filter:"
    end = text.find(end_marker, start)
    if end < 0:
        raise RuntimeError("config 里找不到 action_filter，无法安全替换 joint_limits")
    backup = config_path + ".bak." + datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(backup, "w", encoding="utf-8") as f:
        f.write(text)
    new_text = text[:start] + yaml_block + text[end:]
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(new_text)
    return backup


def shutdown(robot: GalbotRobot) -> None:
    try:
        robot.request_shutdown()
        robot.wait_for_shutdown()
        robot.destroy()
    except Exception as exc:
        print("SDK 释放异常: {}".format(exc))


def main() -> int:
    args = parse_args()
    names = expand_names(args.groups)
    period = 1.0 / max(args.hz, 1.0)

    print("初始化 G1 SDK ...")
    robot = GalbotRobot()
    if not robot.init():
        print("Initialization failed")
        return 1
    print("Initialization succeeded")
    time.sleep(1.0)

    env = Envelope(names)
    recording = True
    started = time.time()
    old_term = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    print("请慢速把 {} 摆到安全边界，脚本会记录每个关节的极值。".format(",".join(args.groups)))
    print("急停随时可按。摆完按 q。\n")
    time.sleep(1.0)

    try:
        while True:
            loop_t = time.time()
            try:
                values = robot.get_joint_positions([], names)
            except Exception as exc:
                print("\n读关节失败: {}".format(exc))
                values = None
            if values and len(values) == len(names):
                if recording:
                    env.update(values)
                sys.stdout.write("\033[H\033[J")
                sys.stdout.write(
                    render(names, values, env, recording, time.time() - started, args.margin)
                )
                sys.stdout.write("\n")
                sys.stdout.flush()

            wait = max(0.0, period - (time.time() - loop_t))
            key = read_key(wait)
            if key is None:
                continue
            if key in ("q", "esc"):
                break
            if key == " ":
                recording = not recording
            if key == "r":
                env.reset()
                started = time.time()
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)

    print("\n采集结束，计算软件限位 ...")
    recorded, status = shrink_limits(names, env, args.margin, args.min_span)
    merged = merge_limits(recorded)
    yaml_block = format_yaml_limits(merged)

    print("\n关节处理结果:")
    for name in names:
        lohi = recorded.get(name)
        extra = ""
        if lohi is not None:
            extra = " -> [{:.4f}, {:.4f}]".format(lohi[0], lohi[1])
        print("  {:<20} {}{}".format(name, status.get(name, "?"), extra))

    print("\n未更新的关节仍沿用现有 vla_config 限位。")
    print("\n可粘贴的 YAML:\n")
    print(yaml_block)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_yaml = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "recorded_joint_limits_{}.yaml".format(stamp),
    )
    payload = {
        "time": stamp,
        "groups": list(args.groups),
        "margin_rad": args.margin,
        "min_span_rad": args.min_span,
        "samples": env.samples,
        "raw_envelope": {
            name: [env.min_pos[i], env.max_pos[i]] for i, name in enumerate(names)
        },
        "status": status,
        "updated_limits": {k: list(v) for k, v in recorded.items()},
        "merged_limits": {k: list(merged[k]) for k in STATE_NAMES},
    }
    with open(out_yaml, "w", encoding="utf-8") as f:
        f.write("# generated by record_joint_limits.py\n")
        f.write(yaml_block + "\n")
    out_json = os.path.splitext(out_yaml)[0] + ".json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("\n已保存:\n  {}\n  {}".format(out_yaml, out_json))

    if args.apply:
        backup = backup_and_apply(args.config, yaml_block)
        print("已写回 {} ，备份: {}".format(args.config, backup))

    shutdown(robot)
    print("Resources released successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
