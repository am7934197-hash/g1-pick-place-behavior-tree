#!/usr/bin/env python3
"""示教记录地图坐标系停靠点（完成 2.1 定位后再用）。

流程:
  1. 地图已放到 /var/maps/cur/，localization_server 已启动，score ≥ 0.8
  2. 手推/遥控到目标位置，朝向摆好
  3. 回车记下当前 get_current_pose()

用法:
  python3 record_nav_points.py
  python3 record_nav_points.py --name pick
  python3 record_nav_points.py --file /tmp/nav_points.json --name place
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
    print_localization_status,
    record_named_pose,
    release_chassis,
    save_points,
    shutdown,
    wait_localized,
)


def ask_yes(msg: str, default_no: bool = True) -> bool:
    hint = "y/N" if default_no else "Y/n"
    key = input(f"{msg} ({hint}): ").strip().lower()
    if not key:
        return not default_no
    return key == "y"


def main() -> int:
    parser = argparse.ArgumentParser(description="示教记录 SLAM 地图停靠点")
    parser.add_argument("--name", default="", help="点名。不传则循环询问")
    parser.add_argument("--file", default=DEFAULT_POINTS_FILE, help="保存 JSON 路径")
    parser.add_argument("--min-score", type=float, default=0.8, help="定位分数下限")
    parser.add_argument("--no-release-chassis", action="store_true", help="不释放底盘（遥控时用）")
    parser.add_argument("--relocalize", action="store_true", help="未定位时用地图原点重定位")
    args = parser.parse_args()

    ok_map, map_msg = map_ready()
    print(f"地图: {map_msg}")
    if not ok_map:
        return 1

    print("请确认：急停已旋开；localization_server 已加载 /var/maps/cur/")
    if input("开始记点？(y/n): ").strip().lower() != "y":
        print("已取消")
        return 0

    robot = GalbotRobot()
    nav = GalbotNavigation()
    if not robot.init():
        print("GalbotRobot.init() 失败。先停掉占用 SDK 的其它脚本。")
        return 1
    if not nav.init():
        print("GalbotNavigation.init() 失败。确认导航/定位服务已起来。")
        shutdown(robot)
        return 1
    time.sleep(1.0)

    init_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] if args.relocalize else None
    if not wait_localized(nav, min_score=args.min_score, timeout_s=8.0, init_pose=init_pose):
        pass
    if not print_localization_status(nav, min_score=args.min_score):
        if not ask_yes("定位未达标，仍要记点？", default_no=True):
            shutdown(robot)
            return 1

    if not args.no_release_chassis:
        print("释放底盘，方便手推。遥控的话下次加 --no-release-chassis")
        release_chassis(robot)
    else:
        ensure_chassis_pose_ctrl(robot)

    data = load_points(args.file)
    data.setdefault("map_dir", "/var/maps/cur")

    names = [args.name] if args.name else None
    try:
        index = 1
        while True:
            if names is not None:
                name = names[0]
            else:
                name = input(f"点名（默认 p{index}，直接回车用默认，q 结束）: ").strip()
                if name.lower() == "q":
                    break
                if not name:
                    name = f"p{index}"
            input(f"把机器人推到「{name}」，朝向摆好后按回车记录...")
            if not nav.is_localized():
                print("此刻未定位，不记录。先保证 score ≥ 0.8")
                if names is not None:
                    return 1
                continue
            rec = record_named_pose(nav, name)
            data["points"][name] = rec
            print(f"已记录 {name}: {format_pose(rec['pose'])}")
            print("  pose =", rec["pose"])
            save_points(args.file, data)
            if names is not None:
                break
            if not ask_yes("再记一个点？", default_no=False):
                break
            index += 1
    except KeyboardInterrupt:
        print("\n提前结束")
    finally:
        save_points(args.file, data)
        shutdown(robot)
        print("已退出")

    print("之后去这个点:")
    print(f"  python3 goto_nav_point.py --name {args.name or '点名'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
