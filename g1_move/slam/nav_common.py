#!/usr/bin/env python3
"""地图导航共用：定位检查、读 score、记点 JSON、导航到目标。"""

from __future__ import annotations

import json
import math
import os
import re
import time
from datetime import datetime
from typing import Optional

from galbot_sdk.g1 import ControlStatus, G1ControllerName, GalbotNavigation, GalbotRobot

MAP_DIR = "/var/maps/cur"
LOC_LOG = "/userdata/log/localization_server/localization_server.INFO"
DEFAULT_POINTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nav_points.json")
SCORE_RE = re.compile(r"localization_server\.cpp:\d+\] score:\s*([0-9.]+)\s+time delay:")


def yaw_from_quat(qx, qy, qz, qw) -> float:
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def pose_list(pose) -> list[float]:
    return [float(v) for v in list(pose)[:7]]


def format_pose(pose) -> str:
    p = pose_list(pose)
    yaw = math.degrees(yaw_from_quat(*p[3:7]))
    return (
        f"[x={p[0]:.4f}, y={p[1]:.4f}, z={p[2]:.4f}, "
        f"qx={p[3]:.6f}, qy={p[4]:.6f}, qz={p[5]:.6f}, qw={p[6]:.6f}]  "
        f"yaw={yaw:.1f}°"
    )


def read_localization_score(log_path: str = LOC_LOG) -> Optional[float]:
    if not os.path.isfile(log_path):
        return None
    try:
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 128 * 1024), os.SEEK_SET)
            text = f.read().decode("utf-8", errors="ignore")
    except OSError:
        return None
    matches = SCORE_RE.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def map_ready() -> tuple[bool, str]:
    pcd = os.path.join(MAP_DIR, "global_cloud.pcd")
    if not os.path.isdir(MAP_DIR):
        return False, f"{MAP_DIR} 不存在。先把地图放到这里（见 2.1）"
    if not os.path.isfile(pcd):
        return False, f"{pcd} 不存在。当前 cur 里没有点云地图"
    return True, MAP_DIR


def print_localization_status(nav: GalbotNavigation, min_score: float = 0.8) -> bool:
    localized = bool(nav.is_localized())
    score = read_localization_score()
    print(f"is_localized: {localized}")
    if score is None:
        print(f"localization score: 读不到（看 {LOC_LOG}）")
        print("  也可另开终端: tail -f /userdata/log/localization_server/localization_server.INFO")
    else:
        print(f"localization score: {score:.4f}  （要求 ≥ {min_score}）")
    if localized:
        try:
            print("当前地图位姿:", format_pose(nav.get_current_pose()))
        except Exception as e:
            print(f"读位姿失败: {e}")
    ok = localized and (score is None or score >= min_score)
    if localized and score is not None and score < min_score:
        print("定位分数偏低：把机器人推回建图起点，朝向对齐，再用 engine_tools 发一次初始位姿。")
    if not localized:
        print("未定位：确认 localization_server 已启动，地图在 /var/maps/cur/，并发送初始位姿。")
    return ok


def wait_localized(
    nav: GalbotNavigation,
    min_score: float = 0.8,
    timeout_s: float = 30.0,
    init_pose: Optional[list[float]] = None,
) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if nav.is_localized():
            score = read_localization_score()
            if score is None or score >= min_score:
                return True
        if init_pose is not None and not nav.is_localized():
            try:
                nav.relocalize(init_pose)
            except Exception:
                pass
        time.sleep(0.5)
    return nav.is_localized()


def shutdown(robot: Optional[GalbotRobot]):
    if robot is None:
        return
    try:
        robot.request_shutdown()
        robot.wait_for_shutdown()
    except Exception:
        pass
    try:
        robot.destroy()
    except Exception:
        pass


def ensure_chassis_pose_ctrl(robot: GalbotRobot) -> bool:
    status = robot.switch_controller(G1ControllerName.CHASSIS_POSE_CTRL)
    print(f"switch_controller(CHASSIS_POSE_CTRL) -> {status}")
    return status == ControlStatus.SUCCESS


def release_chassis(robot: GalbotRobot) -> bool:
    try:
        status = robot.release_controller("chassis")
    except Exception as e:
        print(f"释放底盘失败: {e}")
        return False
    ok = status == ControlStatus.SUCCESS
    print(f"release_controller('chassis') -> {status} ({'成功' if ok else '失败'})")
    return ok


def load_points(path: str) -> dict:
    if not os.path.isfile(path):
        return {"points": {}}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("points", {})
    return data


def save_points(path: str, data: dict):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"已保存: {path}")


def record_named_pose(nav: GalbotNavigation, name: str) -> dict:
    pose = pose_list(nav.get_current_pose())
    yaw = yaw_from_quat(*pose[3:7])
    return {
        "name": name,
        "pose": pose,
        "yaw_deg": round(math.degrees(yaw), 3),
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
    }


def navigate_to_pose(
    nav: GalbotNavigation,
    goal_pose,
    timeout_s: float = 30.0,
    retry: int = 3,
    enable_collision_check: bool = False,
) -> bool:
    goal = pose_list(goal_pose)
    cur = pose_list(nav.get_current_pose())
    reachable = nav.check_path_reachability(goal, cur)
    print(f"reachable={reachable}")
    print("  当前:", format_pose(cur))
    print("  目标:", format_pose(goal))
    if not reachable:
        print("路径不可达，不发导航")
        return False
    status = nav.navigate_to_goal(
        goal_pose=goal,
        enable_collision_check=enable_collision_check,
        is_blocking=False,
        timeout=timeout_s,
    )
    print(f"已发出导航（非阻塞, collision_check={enable_collision_check}） status={status}")
    print("机器人会自己走，脚本不再等待到达。")
    return bool(status[0]) if isinstance(status, tuple) else bool(status)
