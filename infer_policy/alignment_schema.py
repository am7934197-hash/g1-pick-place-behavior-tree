"""Schemas shared by the mixed-generation pick/place behavior-tree policies."""

from __future__ import annotations

from typing import List

PICK_CHECKPOINT = (
    "/home/zhangyuqi/zhangyuqi/models/g1/pick_place_0911_clean179/"
    "checkpoints/040000/pretrained_model"
)
PLACE_CHECKPOINT = (
    "/home/zhangyuqi/zhangyuqi/G1-20260821/g1/place/"
    "checkpoints/040000/pretrained_model"
)
ALLOWED_CHECKPOINTS = (PICK_CHECKPOINT, PLACE_CHECKPOINT)
# Compatibility alias for older diagnostics importing this name.
PINNED_CHECKPOINT = PICK_CHECKPOINT
FORBIDDEN_CHECKPOINT_MARKERS = ("010000", "/last", "\\last", "checkpoints/last")

REQUIRED_CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_preprocessor_step_3_normalizer_processor.safetensors",
    "policy_postprocessor.json",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)

READY224_DIR = (
    "/home/zhangyuqi/zhangyuqi/G1-20260911/G1_data/pick_place_9.11/test/"
    "clean_ready_224"
)

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

# The new pick policy predicts the two arms and two grippers.  The legacy place
# policy still returns the full 23-D vector; RobotController accepts both and
# fills the missing leg/head targets from fresh measured state for the pick model.
ACTION_NAMES: List[str] = list(STATE_NAMES[:16])
ARM_JOINT_INDICES = tuple(i for i in range(16) if i not in (7, 15))
GRIPPER_INDICES = (7, 15)
CAMERA_KEYS = ("head_right", "left_arm", "right_arm")
TRAINING_FPS = 30.0
TRAINING_IMAGE_HW = (224, 224)
MODEL_IMAGE_HW = (224, 224)
TRAINING_LETTERBOX_CONTENT_HW = (168, 224)
TRAINING_LETTERBOX_PAD_TBLR = (28, 28, 0, 0)

# Start-state medians for task 0 in clean_ready_224.
INIT_JOINT_POSITIONS = {
    "right_arm": [-1.085722, 0.586364, 0.691154, 1.415937, 0.352408, -0.202317, -1.538150],
    "right_gripper": [0.0],
    "left_arm": [0.790863, -0.594513, -0.110183, -1.619236, -0.285080, 0.030295, 1.149216],
    "left_gripper": [0.00],
    "leg": [0.576609, 1.462627, 0.925278, 0.040399, 0.000090],
    "head": [-0.074925, 0.430006],
}
