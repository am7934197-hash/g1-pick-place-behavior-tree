"""Schema shared by the 040000 pick/place policies used by the behavior tree."""

from __future__ import annotations

from typing import List

PICK_CHECKPOINT = (
    "/home/zhangyuqi/zhangyuqi/G1-20260821/g1/pick/"
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
    "/home/zhangyuqi/zhangyuqi/G1-20260821/G1_data/pickandplace/"
    "pick_place_top1_actiondim_import/ready_224"
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

ACTION_NAMES: List[str] = list(STATE_NAMES)
ARM_JOINT_INDICES = tuple(i for i in range(16) if i not in (7, 15))
GRIPPER_INDICES = (7, 15)
CAMERA_KEYS = ("head_left", "head_right", "left_arm", "right_arm")
TRAINING_FPS = 30.0
TRAINING_IMAGE_HW = (480, 640)
MODEL_IMAGE_HW = (480, 640)
TRAINING_LETTERBOX_CONTENT_HW = (168, 224)
TRAINING_LETTERBOX_PAD_TBLR = (28, 28, 0, 0)

# Episode-0 medians claimed for ready_224 pickup episodes.
INIT_JOINT_POSITIONS = {
    "right_arm": [-1.074697, 0.587419, 0.683460, 1.414618, 0.352612, -0.195067, -1.538162],
    "right_gripper": [0.0],
    "left_arm": [0.767878, -0.583967, -0.088312, -1.620591, -0.284889, 0.026508, 1.193521],
    "left_gripper": [0.00],
    "leg": [0.576609, 1.462627, 0.925278, 0.040399, 0.000090],
    "head": [-0.074925, 0.430006],
}
