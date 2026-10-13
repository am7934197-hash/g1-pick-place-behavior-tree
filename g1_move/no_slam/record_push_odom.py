#!/usr/bin/env python3
"""手推底盘，记录相对起点的最终位移。

用法:
    python3 record_push_odom.py

重要:
    G1 底盘电机上电后是锁死的，硬推轮子不转，get_odom() 就不会变。
    本脚本会先释放 chassis 控制器。请旋开急停，让轮子能被推动。
"""

from __future__ import annotations

import json
import math
import os
import select
import sys
import time
from datetime import datetime

from galbot_sdk.g1 import ControlStatus, GalbotRobot

STILL_POSE_M = 0.005
STILL_YAW_DEG = 0.5
PRINT_PERIOD_S = 0.4


def _as_list(value, n):
    if value is None:
        return [0.0] * n
    if isinstance(value, dict):
        keys = ("x", "y", "z", "w")[:n]
        if any(k in value for k in keys):
            return [float(value.get(k, 0.0)) for k in keys]
    if hasattr(value, "x") and n >= 2:
        vals = [float(value.x), float(value.y)]
        if n >= 3:
            vals.append(float(getattr(value, "z", 0.0)))
        if n >= 4:
            vals.append(float(getattr(value, "w", 1.0)))
        return vals[:n]
    try:
        seq = list(value)
        if len(seq) >= n:
            return [float(v) for v in seq[:n]]
    except TypeError:
        pass
    return [0.0] * n


def dump_raw_odom(raw):
    print("原始 get_odom() 类型:", type(raw).__name__)
    if isinstance(raw, dict):
        print("原始 keys:", list(raw.keys()))
        for k, v in raw.items():
            print(f"  {k} = {v}")
        return
    attrs = [a for a in dir(raw) if not a.startswith("_")]
    print("原始属性:", attrs)
    for name in (
        "timestamp_ns",
        "position",
        "orientation",
        "linear_velocity",
        "angular_velocity",
        "x",
        "y",
        "z",
    ):
        if hasattr(raw, name):
            print(f"  .{name} = {getattr(raw, name)}")


def read_odom(robot):
    raw = robot.get_odom()
    if raw is None:
        return None
    if isinstance(raw, dict):
        pos = _as_list(raw.get("position", raw.get("pose")), 3)
        if pos == [0.0, 0.0, 0.0] and "x" in raw:
            pos = [float(raw.get("x", 0.0)), float(raw.get("y", 0.0)), float(raw.get("z", 0.0))]
        ori = _as_list(raw.get("orientation"), 4)
        lin = _as_list(raw.get("linear_velocity"), 3)
        ang = _as_list(raw.get("angular_velocity"), 3)
        ts = raw.get("timestamp_ns", raw.get("timestamp", 0))
    else:
        pos = _as_list(getattr(raw, "position", None), 3)
        if pos == [0.0, 0.0, 0.0] and hasattr(raw, "x"):
            pos = [float(raw.x), float(raw.y), float(getattr(raw, "z", 0.0))]
        ori = _as_list(getattr(raw, "orientation", None), 4)
        lin = _as_list(getattr(raw, "linear_velocity", None), 3)
        ang = _as_list(getattr(raw, "angular_velocity", None), 3)
        ts = getattr(raw, "timestamp_ns", getattr(raw, "timestamp", 0))
    if ori == [0.0, 0.0, 0.0, 0.0]:
        ori = [0.0, 0.0, 0.0, 1.0]
    yaw = math.atan2(
        2.0 * (ori[3] * ori[2] + ori[0] * ori[1]),
        1.0 - 2.0 * (ori[1] * ori[1] + ori[2] * ori[2]),
    )
    return {
        "timestamp_ns": int(ts or 0),
        "x": pos[0],
        "y": pos[1],
        "z": pos[2] if len(pos) > 2 else 0.0,
        "yaw": yaw,
        "qx": ori[0],
        "qy": ori[1],
        "qz": ori[2],
        "qw": ori[3],
        "vx": lin[0] if lin else 0.0,
        "vy": lin[1] if len(lin) > 1 else 0.0,
        "wz": ang[2] if len(ang) > 2 else 0.0,
    }


def wrap_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def relative_to_start(start, cur):
    dx = cur["x"] - start["x"]
    dy = cur["y"] - start["y"]
    c = math.cos(start["yaw"])
    s = math.sin(start["yaw"])
    return c * dx + s * dy, -s * dx + c * dy, wrap_pi(cur["yaw"] - start["yaw"])


def move_straight_target(local_x, local_y, dtheta):
    return [
        round(local_x, 6),
        round(local_y, 6),
        0.0,
        0.0,
        0.0,
        round(math.sin(dtheta / 2.0), 6),
        round(math.cos(dtheta / 2.0), 6),
    ]


def pose_snapshot(odom):
    return {
        "x": odom["x"],
        "y": odom["y"],
        "z": odom["z"],
        "yaw_rad": odom["yaw"],
        "yaw_deg": math.degrees(odom["yaw"]),
        "qx": odom["qx"],
        "qy": odom["qy"],
        "qz": odom["qz"],
        "qw": odom["qw"],
    }


def stdin_ready():
    if not sys.stdin.isatty():
        return False
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    return bool(ready)


def wait_odom(robot, timeout_s=5.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        odom = read_odom(robot)
        if odom is not None:
            return odom
        time.sleep(0.1)
    return None


def read_chassis_joints(robot):
    try:
        pos = robot.get_joint_positions(["chassis"], [])
        if not pos:
            return None
        return [float(v) for v in pos]
    except Exception:
        return None


def release_chassis(robot):
    try:
        status = robot.release_controller("chassis")
    except Exception as e:
        print(f"释放底盘失败: {e}")
        return False
    ok = status == ControlStatus.SUCCESS
    print(f"release_controller('chassis') -> {status} ({'成功' if ok else '失败'})")
    return ok


def record_one_push(robot, index):
    start = wait_odom(robot)
    if start is None:
        print("读不到里程计，无法记录。")
        return None
    start_joints = read_chassis_joints(robot)

    print("\n" + "=" * 64)
    print(f"第 {index} 段  起点已记下")
    print(
        f"  odom绝对: x={start['x']:.4f}  y={start['y']:.4f}  "
        f"yaw={math.degrees(start['yaw']):.1f}°  ts={start['timestamp_ns']}"
    )
    if start_joints is not None:
        print("  chassis关节:", [round(v, 4) for v in start_joints])
    print("  当前朝向 = +X（前方），左手边 = +Y")
    print("  请慢慢推，让轮子转起来；到位后按回车")
    print("=" * 64)

    moving = False
    peak_speed = 0.0
    path_len = 0.0
    last = start
    cur = start
    last_ts = start["timestamp_ns"]
    ts_changed = False
    last_print = 0.0
    started_at = time.time()
    frozen_warned = False

    try:
        while True:
            sample = read_odom(robot)
            if sample is not None:
                if sample["timestamp_ns"] != last_ts:
                    ts_changed = True
                    last_ts = sample["timestamp_ns"]
                step = math.hypot(sample["x"] - last["x"], sample["y"] - last["y"])
                path_len += step
                last = sample
                cur = sample
                speed = math.hypot(sample["vx"], sample["vy"])
                peak_speed = max(peak_speed, speed)
                lx, ly, dth = relative_to_start(start, cur)
                pose_moved = (
                    math.hypot(lx, ly) > STILL_POSE_M
                    or abs(math.degrees(dth)) > STILL_YAW_DEG
                )
                if pose_moved:
                    moving = True
                now = time.time()
                if now - last_print >= PRINT_PERIOD_S:
                    last_print = now
                    joints = read_chassis_joints(robot)
                    joint_delta = ""
                    if joints is not None and start_joints is not None:
                        dj = [a - b for a, b in zip(joints, start_joints)]
                        joint_delta = "  轮角Δ=" + str([round(v, 3) for v in dj])
                    stale = "" if ts_changed else "  [里程计时间戳没变，底盘可能仍锁着]"
                    if stale and not frozen_warned and now - started_at > 2.0:
                        frozen_warned = True
                        print("提示: odom 没更新。请确认急停已旋开，且轮子真的在转，不要抬起来搬。")
                    state = "已离开起点" if pose_moved else "还在起点"
                    print(
                        f"[{state}] 相对 前X={lx:+.3f}m 左Y={ly:+.3f}m "
                        f"转={math.degrees(dth):+.1f}° | "
                        f"绝对x={sample['x']:.3f} y={sample['y']:.3f} "
                        f"yaw={math.degrees(sample['yaw']):.1f}° "
                        f"路程={path_len:.3f}m{joint_delta}{stale}"
                    )
            if stdin_ready():
                sys.stdin.readline()
                break
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，以当前位姿作为终点。")

    lx, ly, dth = relative_to_start(start, cur)
    target = move_straight_target(lx, ly, dth)
    elapsed = time.time() - started_at
    record = {
        "index": index,
        "elapsed_s": round(elapsed, 3),
        "detected_motion": moving,
        "odom_timestamp_changed": ts_changed,
        "peak_speed_mps": round(peak_speed, 4),
        "path_length_m": round(path_len, 4),
        "start_odom": pose_snapshot(start),
        "end_odom": pose_snapshot(cur),
        "relative_base_link": {
            "forward_x_m": round(lx, 6),
            "left_y_m": round(ly, 6),
            "yaw_rad": round(dth, 6),
            "yaw_deg": round(math.degrees(dth), 3),
        },
        "move_straight_to_target": target,
    }

    print("\n" + "-" * 64)
    print(f"第 {index} 段  最终位置（相对推动前的机体系）")
    print(f"  前方 +X : {lx:+.4f} m")
    print(f"  左方 +Y : {ly:+.4f} m")
    print(f"  左转    : {math.degrees(dth):+.2f} °")
    print(f"  路程    : {path_len:.4f} m")
    print(f"  里程计是否在更新: {'是' if ts_changed else '否（手推没被记上）'}")
    print("  可直接用于 move_straight_to:")
    print(f"    target = {target}")
    print("-" * 64)
    if not ts_changed or (abs(lx) < 1e-3 and abs(ly) < 1e-3 and abs(dth) < 1e-3):
        print("这次没有有效位移。常见原因：")
        print("  1. 急停还按着 → 旋开急停")
        print("  2. 底盘电机锁着、轮子没转 → 确认已释放 chassis，推的时候轮子要滚")
        print("  3. 把机器人抬起来搬了 → 编码器不会转")
    return record


def save_records(records):
    out_dir = os.path.dirname(os.path.abspath(__file__))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"push_odom_{stamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"records": records}, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {path}")
    return path


def main():
    print("手推底盘记录：不建图，只读 odom。")
    print("先 Ctrl+C 停掉旧进程，再运行本脚本。")
    print("急停必须旋开；硬推锁死的轮子，数字不会变。")
    robot = GalbotRobot()
    if not robot.init():
        print("GalbotRobot.init() 失败")
        return 1
    time.sleep(1.0)

    raw = robot.get_odom()
    if raw is None:
        print("初始化后仍读不到 odom，请检查底盘服务。")
        robot.request_shutdown()
        robot.wait_for_shutdown()
        robot.destroy()
        return 1
    dump_raw_odom(raw)
    print("---")
    release_chassis(robot)

    records = []
    try:
        wait_enter = lambda msg: input(msg)
        wait_enter("确认急停已旋开、轮子能被推动后，按回车开始第 1 段...")
        index = 1
        while True:
            rec = record_one_push(robot, index)
            if rec is not None:
                records.append(rec)
            again = input("再记一段？(y/n，默认 n): ").strip().lower()
            if again != "y":
                break
            index += 1
            wait_enter("把机器人摆到下一段起点后按回车...")
    except KeyboardInterrupt:
        print("\n提前结束。")
    finally:
        if records:
            save_records(records)
        print("正在释放 SDK...")
        robot.request_shutdown()
        robot.wait_for_shutdown()
        robot.destroy()
        print("已退出")
    return 0


if __name__ == "__main__":
    sys.exit(main())
