"""Schemas shared by the mixed-generation pick/place behavior-tree policies."""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Mapping

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
HARDWARE_CAMERA_KEYS = ("head_left", "head_right", "left_arm", "right_arm")
SENSOR_SET = [
    "HEAD_LEFT_CAMERA",
    "HEAD_RIGHT_CAMERA",
    "LEFT_ARM_CAMERA",
    "RIGHT_ARM_CAMERA",
]
OBSERVATION_CAMERA_MAP = {name: name for name in HARDWARE_CAMERA_KEYS}
TRAINING_FPS = 30.0
CONTROL_DT = 0.033
ACTIONS_PER_CHUNK = 50
TRAINING_IMAGE_HW = (224, 224)
MODEL_IMAGE_HW = (224, 224)
TRAINING_LETTERBOX_CONTENT_HW = (168, 224)
TRAINING_LETTERBOX_PAD_TBLR = (28, 28, 0, 0)
PNG_COMPRESS_LEVEL = 3

DEFAULT_JOINT_GROUPS = [
    {"name": "right_arm", "indices": [0, 1, 2, 3, 4, 5, 6], "enabled": True},
    {"name": "right_gripper", "indices": [7], "is_gripper": True, "enabled": True},
    {"name": "left_arm", "indices": [8, 9, 10, 11, 12, 13, 14], "enabled": True},
    {"name": "left_gripper", "indices": [15], "is_gripper": True, "enabled": True},
    {"name": "leg", "indices": [16, 17, 18, 19, 20], "enabled": False},
    {"name": "head", "indices": [21, 22], "enabled": False},
]

# Start-state medians for task 0 in clean_ready_224.
INIT_JOINT_POSITIONS = {
    "right_arm": [-1.085722, 0.586364, 0.691154, 1.415937, 0.352408, -0.202317, -1.538150],
    "right_gripper": [0.0],
    "left_arm": [0.790863, -0.594513, -0.110183, -1.619236, -0.285080, 0.030295, 1.149216],
    "left_gripper": [0.00],
    "leg": [0.576609, 1.462627, 0.925278, 0.040399, 0.000090],
    "head": [-0.074925, 0.430006],
}


def _image_feature(height: int = MODEL_IMAGE_HW[0], width: int = MODEL_IMAGE_HW[1]) -> Dict[str, Any]:
    return {
        "dtype": "image",
        "shape": [height, width, 3],
        "names": ["height", "width", "channels"],
    }


def make_observation_features(cameras: Iterable[str] | None = None) -> Dict[str, Any]:
    """Pick defaults to 3 cameras; pass 4 names for the legacy place model."""
    names = tuple(cameras) if cameras is not None else CAMERA_KEYS
    features: Dict[str, Any] = {
        f"observation.images.{camera}": _image_feature() for camera in names
    }
    features["observation.state"] = {
        "dtype": "float32",
        "shape": [len(STATE_NAMES)],
        "names": list(STATE_NAMES),
    }
    return features


def apply_vla_defaults(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Fill stable PI05/G1 schema fields so YAML only needs operator overrides."""
    cfg: Dict[str, Any] = copy.deepcopy(dict(config or {}))
    cfg.setdefault("policy_type", "pi05")
    cfg.setdefault("device", "cuda")
    cfg.setdefault("fps", int(TRAINING_FPS))
    cfg.setdefault("actions_per_chunk", ACTIONS_PER_CHUNK)

    transport = cfg.setdefault("observation_transport", {})
    transport.setdefault("image_encoding", "png")
    transport.setdefault("png_compress_level", PNG_COMPRESS_LEVEL)

    image_cfg = cfg.setdefault("image_preprocess", {})
    image_cfg.setdefault("enabled", True)
    image_cfg.setdefault("target_size", [MODEL_IMAGE_HW[0], MODEL_IMAGE_HW[1]])
    image_cfg.setdefault("keep_aspect_ratio", True)

    cameras = cfg.get("cameras")
    if cameras:
        cfg["observation_features"] = make_observation_features(cameras)
    elif not cfg.get("observation_features"):
        cfg["observation_features"] = make_observation_features()

    robot = cfg.setdefault("robot", {})
    robot.setdefault("sensor_set", list(SENSOR_SET))
    robot.setdefault("observation_camera_map", dict(OBSERVATION_CAMERA_MAP))
    robot.setdefault("state_feature_names", list(STATE_NAMES))
    robot.setdefault("vla_action_dim", len(ACTION_NAMES))
    if not robot.get("joint_groups"):
        robot["joint_groups"] = copy.deepcopy(DEFAULT_JOINT_GROUPS)

    control = robot.setdefault("control", {})
    control.setdefault("fps", int(TRAINING_FPS))
    control.setdefault("dt", CONTROL_DT)
    control.setdefault("action_semantics", "absolute")
    control.setdefault("set_joint_time_from_start", 0.0)
    control.setdefault("gripper_action_scale", 0.0012)
    interpolation = control.setdefault("interpolation", {})
    interpolation.setdefault("enabled", True)
    interpolation.setdefault("method", "linear")
    interpolation.setdefault("input_hz", int(TRAINING_FPS))
    interpolation.setdefault("output_hz", 250)

    cfg.setdefault("auto_setup_policy", True)
    cfg.setdefault("force_setup_policy", True)
    cfg.setdefault("execute_actions_per_chunk", 15)
    cfg.setdefault("obs_queue_timeout", 1.0)
    cfg.setdefault("action_response_timeout", 30.0)
    robot.setdefault("sensor_warmup_seconds", 15.0)
    robot.setdefault("camera_rgb_retries", 8)
    robot.setdefault("camera_rgb_retry_delay", 0.3)
    robot.setdefault("reference_frame", "base_link")
    robot.setdefault("end_effector_frame", "right_arm")
    verify = robot.setdefault("init_pose_verify", {})
    verify.setdefault("joint_tolerance_rad", 0.03)
    verify.setdefault("gripper_tolerance_model", 3.0)
    init_restore = robot.setdefault("init_pose_restore", {})
    init_restore.setdefault("timeout_seconds", 8.0)
    init_restore.setdefault("poll_interval_seconds", 0.1)
    init_restore.setdefault("stable_samples", 2)
    init_restore.setdefault("retry_delay_seconds", 0.2)
    init_restore.setdefault("retries", 2)
    recovery = robot.setdefault("recovery", {})
    recovery.setdefault("gripper_open_width", 0.12)
    recovery.setdefault("settle_seconds", 8.0)
    recovery.setdefault("retries", 2)

    server = cfg.setdefault("server", {})
    server.setdefault("host", "0.0.0.0")
    server.setdefault("port", 8006)
    log_cfg = cfg.setdefault("inference_log", {})
    log_cfg.setdefault("enabled", True)
    log_cfg.setdefault("log_dir", "logs")
    log_cfg.setdefault("save_images", True)
    log_cfg.setdefault("image_dir", "logs/images")
    log_cfg.setdefault("max_image_runs", 5)
    return cfg
