#!/usr/bin/env python3
"""把腰转正：leg_joint4 / leg_joint5 置 0，身高 j1/j2/j3 保持当前值。

不要和 infer_server / replay / tune_*_live 同时占用 SDK。
"""

import time

from galbot_sdk.g1 import ControlStatus, GalbotRobot

J4_LIMIT = (-1.5906, 1.5906)
J5_LIMIT = (-0.1645, 0.1645)
# 正对底盘前方。若只要稍微回正，改这两个数即可。
STRAIGHT_YAW = 0.0
STRAIGHT_ROLL = 0.0
MAX_SPEED = 0.15
TIMEOUT_S = 45


def clip(value, lo, hi):
    return min(max(float(value), lo), hi)


robot = GalbotRobot()
robot.init()
print("Initialization succeeded")
time.sleep(2)

pose = robot.get_joint_positions(["leg"], [])
if not pose or len(pose) < 5:
    raise RuntimeError(f"get_joint_positions(leg) failed: {pose}")

joint_pos = [
    float(pose[0]),
    float(pose[1]),
    float(pose[2]),
    clip(STRAIGHT_YAW, *J4_LIMIT),
    clip(STRAIGHT_ROLL, *J5_LIMIT),
]
print(
    "current leg "
    f"[{pose[0]:.4f}, {pose[1]:.4f}, {pose[2]:.4f}, {pose[3]:.4f}, {pose[4]:.4f}]"
)
print(
    "target  leg "
    f"[{joint_pos[0]:.4f}, {joint_pos[1]:.4f}, {joint_pos[2]:.4f}, "
    f"{joint_pos[3]:.4f}, {joint_pos[4]:.4f}]  (j4/j5 -> straight)"
)

status = robot.set_joint_positions(
    joint_pos, ["leg"], [], True, MAX_SPEED, TIMEOUT_S
)
print(f"set_joint_positions status={status}")
if status != ControlStatus.SUCCESS:
    print("Joint angle setting failed")
else:
    print("Body yaw set straight")

time.sleep(1)
robot.request_shutdown()
robot.wait_for_shutdown()
robot.destroy()
print("Resources released successfully")
