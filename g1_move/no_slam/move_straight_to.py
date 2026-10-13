#!/usr/bin/env python3
"""让 SDK 自己走底盘（不建图、不启导航服务）。

未指定的轴就是 0，不会偷偷加前进。
只往左 1m:  python3 move_straight_to.py --y 1
只往前 0.2m: python3 move_straight_to.py --x 0.2
只左转 90°:  python3 move_straight_to.py --yaw 90
"""

from __future__ import annotations

import argparse
import math
import sys
import time

from galbot_sdk.g1 import ControlStatus, G1ControllerName, GalbotRobot


def yaw_from_quat(qx, qy, qz, qw):
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def read_odom_xyyaw(robot):
    raw = robot.get_odom()
    if raw is None:
        return None
    if isinstance(raw, dict):
        pos = list(raw.get("position") or [0, 0, 0])
        ori = list(raw.get("orientation") or [0, 0, 0, 1])
    else:
        pos = list(getattr(raw, "position", [0, 0, 0]))
        ori = list(getattr(raw, "orientation", [0, 0, 0, 1]))
    yaw = yaw_from_quat(*[float(v) for v in ori[:4]])
    return {
        "x": float(pos[0]),
        "y": float(pos[1]),
        "yaw": yaw,
        "yaw_deg": math.degrees(yaw),
    }


def print_status(name, status):
    print(f"  {name} -> {status}")
    return status == ControlStatus.SUCCESS


def ensure_chassis_pose_ctrl(robot) -> bool:
    try:
        print("当前 chassis 控制器:", robot.get_active_controller("chassis"))
    except Exception as e:
        print("查询 chassis 控制器失败:", e)

    if print_status(
        "switch_controller",
        robot.switch_controller(G1ControllerName.CHASSIS_POSE_CTRL),
    ):
        return True

    print("切换失败，尝试恢复底盘控制器")
    print_status("reload_controller(chassis)", robot.reload_controller("chassis"))
    time.sleep(0.5)
    print_status(
        "acquire_controller",
        robot.acquire_controller(G1ControllerName.CHASSIS_POSE_CTRL),
    )
    time.sleep(0.5)
    print_status("start_controller", robot.start_controller("chassis"))
    time.sleep(0.5)
    return print_status(
        "switch_controller",
        robot.switch_controller(G1ControllerName.CHASSIS_POSE_CTRL),
    )


def shutdown(robot):
    robot.request_shutdown()
    robot.wait_for_shutdown()
    robot.destroy()


def main():
    parser = argparse.ArgumentParser(description="G1 相对当前朝向移动（未指定的轴=0）")
    parser.add_argument("--x", type=float, default=0.0, help="前方米数，正=机头。不传就是 0")
    parser.add_argument("--y", type=float, default=0.0, help="左方米数，正=左手，负=右手。不传就是 0")
    parser.add_argument("--yaw", type=float, default=0.0, help="左转角度（度）。不传就是 0")
    parser.add_argument("--timeout", type=float, default=20.0, help="超时秒数")
    parser.add_argument("--time", type=float, default=None, help="底盘插值时间（秒）")
    args = parser.parse_args()

    if abs(args.x) < 1e-9 and abs(args.y) < 1e-9 and abs(args.yaw) < 1e-9:
        print("必须指定 --x / --y / --yaw 至少一个。")
        print("只沿 Y 走: python3 move_straight_to.py --y 1")
        return 1

    dtheta = math.radians(args.yaw)
    dist = math.hypot(args.x, args.y)
    interp_s = args.time if args.time is not None else max(3.0, dist / 0.15 + abs(args.yaw) / 30.0)
    interp_s = min(interp_s, args.timeout)

    print("即将相对当前车体移动（没写的轴 = 0）：")
    print(f"  前方 +X = {args.x:+.3f} m")
    print(f"  左方 +Y = {args.y:+.3f} m")
    print(f"  左转     = {args.yaw:+.1f} °")
    print(f"  预计用时 = {interp_s:.1f} s")
    print("请确认：急停已旋开，前方/侧方无人无障碍。")
    if input("开始？(y/n): ").strip().lower() != "y":
        print("已取消")
        return 0

    robot = GalbotRobot()
    if not robot.init():
        print("GalbotRobot.init() 失败。若刚跑过别的 SDK 脚本，先把它 Ctrl+C 停掉。")
        return 1
    time.sleep(1.0)

    if not ensure_chassis_pose_ctrl(robot):
        shutdown(robot)
        return 1

    before = read_odom_xyyaw(robot)
    if before is None:
        print("读不到 odom，无法计算相对目标")
        shutdown(robot)
        return 1
    print(f"出发前 odom: x={before['x']:.3f} y={before['y']:.3f} yaw={before['yaw_deg']:.1f}°")

    yaw0 = before["yaw"]
    goal_x = before["x"] + args.x * math.cos(yaw0) - args.y * math.sin(yaw0)
    goal_y = before["y"] + args.x * math.sin(yaw0) + args.y * math.cos(yaw0)
    goal_yaw = yaw0 + dtheta
    print(f"odom目标: x={goal_x:.3f} y={goal_y:.3f} yaw={math.degrees(goal_yaw):.1f}°")

    print("开始走...")
    status = robot.set_base_pose(
        float(goal_x),
        float(goal_y),
        float(goal_yaw),
        "odom",
        "odom",
        float(interp_s),
        True,
        float(args.timeout),
    )
    print(f"set_base_pose -> {status}")

    after = read_odom_xyyaw(robot)
    if after:
        print(f"结束后 odom: x={after['x']:.3f} y={after['y']:.3f} yaw={after['yaw_deg']:.1f}°")
        print(
            f"odom变化: dx={after['x']-before['x']:+.3f} "
            f"dy={after['y']-before['y']:+.3f} "
            f"dyaw={after['yaw_deg']-before['yaw_deg']:+.1f}°"
        )

    shutdown(robot)
    print("已退出")
    return 0 if status == ControlStatus.SUCCESS else 1


if __name__ == "__main__":
    sys.exit(main())
