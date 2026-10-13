#!/usr/bin/env python3
"""导航到示教记下的地图点。

用法:
  python3 goto_nav_point.py --name pick
  python3 goto_nav_point.py --name pick --file nav_points.json
  python3 goto_nav_point.py --pose 4.08 -1.43 0.04 0.002  -0.003 0.094 0.996
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from galbot_sdk.g1 import GalbotNavigation, GalbotRobot

from nav_common import (
    DEFAULT_POINTS_FILE,
    ensure_chassis_pose_ctrl,
    format_pose,
    load_points,
    map_ready,
    navigate_to_pose,
    print_localization_status,
    shutdown,
    wait_localized,
)


def parse_pose(values: list[float]) -> list[float]:
    if len(values) != 7:
        raise argparse.ArgumentTypeError("pose 必须是 7 个数: x y z qx qy qz qw")
    return [float(v) for v in values]


def main() -> int:
    parser = argparse.ArgumentParser(description="导航到 SLAM 示教点")
    parser.add_argument("--name", default="", help="nav_points.json 里的点名")
    parser.add_argument("--file", default=DEFAULT_POINTS_FILE, help="记点 JSON")
    parser.add_argument(
        "--pose",
        nargs=7,
        type=float,
        metavar=("X", "Y", "Z", "QX", "QY", "QZ", "QW"),
        help="直接指定地图位姿，不读文件",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.8)
    parser.add_argument("--collision-check", action="store_true", help="打开动态碰撞检测，默认关")
    parser.add_argument("--relocalize", action="store_true")
    args = parser.parse_args()

    if args.pose is None and not args.name:
        print("必须给 --name 或 --pose")
        return 1

    ok_map, map_msg = map_ready()
    print(f"地图: {map_msg}")
    if not ok_map:
        return 1

    if args.pose is not None:
        goal = parse_pose(args.pose)
        goal_name = "cli-pose"
    else:
        data = load_points(args.file)
        rec = data.get("points", {}).get(args.name)
        if rec is None:
            names = ", ".join(sorted(data.get("points", {})) ) or "(空)"
            print(f"文件里没有点「{args.name}」。已有: {names}")
            print(f"文件: {args.file}")
            return 1
        goal = rec["pose"]
        goal_name = args.name

    print(f"目标 {goal_name}: {format_pose(goal)}")
    print("请确认：急停已旋开，沿途无人无障碍。")
    if input("开始导航？(y/n): ").strip().lower() != "y":
        print("已取消")
        return 0

    robot = GalbotRobot()
    nav = GalbotNavigation()
    if not robot.init():
        print("GalbotRobot.init() 失败")
        return 1
    if not nav.init():
        print("GalbotNavigation.init() 失败")
        shutdown(robot)
        return 1
    time.sleep(1.0)

    if not ensure_chassis_pose_ctrl(robot):
        shutdown(robot)
        return 1

    init_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] if args.relocalize else None
    wait_localized(nav, min_score=args.min_score, timeout_s=8.0, init_pose=init_pose)
    if not print_localization_status(nav, min_score=args.min_score):
        print("定位未达标，不发导航。")
        shutdown(robot)
        return 1

    ok = navigate_to_pose(
        nav,
        goal,
        timeout_s=args.timeout,
        retry=args.retry,
        enable_collision_check=args.collision_check,
    )
    shutdown(robot)
    print("已发出目标，脚本退出；底盘继续走。")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
