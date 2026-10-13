"""Set both grippers to the training-data start position."""

import time

from galbot_sdk.g1 import ControlStatus, GalbotRobot


robot = GalbotRobot()
if not robot.init():
    raise RuntimeError("GalbotRobot initialization failed")
print("Initialization succeeded")
time.sleep(2)

# Physical gripper widths.  The training observations start at model value 0,
# which maps to 0.0 m with the inference client's 0..100 -> 0..0.12 mapping.
joint_pos = [0.0, 0.0]
joint_groups = ["right_gripper", "left_gripper"]

status = robot.set_joint_positions(
    joint_pos,
    joint_groups,
    [],
    True,
    0.1,
    45,
)
if status != ControlStatus.SUCCESS:
    print("Gripper position setting failed")
else:
    print("Gripper position setting succeeded")

time.sleep(1)
robot.request_shutdown()
robot.wait_for_shutdown()
robot.destroy()
print("Resources released successfully")
