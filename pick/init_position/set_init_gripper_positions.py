"""Set both grippers to the training-data start position.

Grippers are not arm joints. ``set_joint_positions`` on a fully closed
target (0.0 m) often never reports arrival, so a blocking call times out
or returns failure. Use ``set_gripper_command`` like the Galbot SDK example.
"""

import time

from galbot_sdk.g1 import ControlStatus, G1JointGroup, GalbotRobot

# Physical width. Training observations start at model value 0 → 0.0 m.
TARGET_WIDTH_M = 0.0
VELOCITY_MPS = 0.1
EFFORT_N = 10.0
SETTLE_S = 2.0


robot = GalbotRobot()
if not robot.init():
    raise RuntimeError("GalbotRobot initialization failed")
print("Initialization succeeded")
time.sleep(2)

ok = True
for name in ("right_gripper", "left_gripper"):
    group = getattr(G1JointGroup, name)
    # Fully closed rarely reports "arrived"; do not block on encoder feedback.
    status = robot.set_gripper_command(
        group, TARGET_WIDTH_M, VELOCITY_MPS, EFFORT_N, False
    )
    print(f"set_gripper_command {name} width={TARGET_WIDTH_M} m status={status}")
    if status != ControlStatus.SUCCESS:
        ok = False

time.sleep(SETTLE_S)
if ok:
    print("Gripper position setting succeeded")
else:
    print("Gripper position setting failed")

time.sleep(1)
robot.request_shutdown()
robot.wait_for_shutdown()
robot.destroy()
print("Resources released successfully")
