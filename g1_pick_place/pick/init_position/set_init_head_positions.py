import time
from galbot_sdk.g1 import GalbotRobot
from galbot_sdk.g1 import ControlStatus

# Get and initialize the GalbotRobot singleton
robot = GalbotRobot()
robot.init()
print('Initialization succeeded')

# Program started, waiting for data
time.sleep(2)

# Set head joints to 0.2, 0.2, block and wait for motion to complete, max timeout 10s
# 训练数据中的头部起始姿态几乎固定。
joint_pos = [-0.074961, 0.429958]
# Set head joint group; if empty, defaults to whole body joints ["leg", "head", "left_arm", "right_arm"]
joint_groups = ["head"]
# Whether to block until joints reach target
is_blocking = True
# Limit joint max speed to 0.4 rad/s
max_speed = 0.4
# Maximum blocking wait time
timeout_s = 45

status = robot.set_joint_positions(
    joint_pos, joint_groups, [], is_blocking, max_speed, timeout_s
)

print(f"set_joint_positions status={status}")
if status != ControlStatus.SUCCESS:
    print("Joint angle setting failed")
else:
    print('Joint angle setting succeeded')

time.sleep(1)

# Use specific joint names for control; this parameter overrides joint_groups
# joint_names = ["head_joint1", "head_joint2"]
# joint_pos = [0.0, 0.0]

# status = robot.set_joint_positions(
#     joint_pos, [], joint_names, is_blocking, max_speed, timeout_s
#)

#if status != ControlStatus.SUCCESS:
#    print("Joint angle setting failed")
#else:
#    print('Joint angle setting succeeded')

# send SIGINT shutdown signal
robot.request_shutdown()
# Wait until entering shutdown state
robot.wait_for_shutdown()
# Perform SDK resource release
robot.destroy()
print('Resources released successfully')
