"""
Robot controller module for Galbot G1.
Encapsulates galbot SDK operations: initialization, navigation, camera, motion, gripper.
"""

try:
    import galbot_sdk.g1 as gm
    from galbot_sdk.g1 import GalbotNavigation
    from galbot_sdk.g1 import GalbotRobot
    from galbot_sdk.g1 import GalbotMotion
    from galbot_sdk.g1 import G1JointGroup, SensorType, JointCommand
except ImportError:
    gm = None
    GalbotNavigation = None
    GalbotRobot = None
    GalbotMotion = None
    G1JointGroup = None
    SensorType = None
    JointCommand = None

import time
import base64
import io
from typing import Sequence, List, Dict, Any, Optional

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None


def decode_compressed_image(compressed_image: dict, camera_info: dict = None) -> np.ndarray:
    """Decode compressed image from galbot SDK format."""
    if cv2 is None:
        raise ImportError("opencv-python is required for image decoding")

    image_data = compressed_image["data"]
    fmt = compressed_image["format"]
    if fmt == "rgb8":
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode RGB image")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif fmt == "16UC1":
        depth_img = np.frombuffer(image_data, dtype=np.uint16).copy()
        if not camera_info:
            depth_img = depth_img.reshape((720, 1280))
        else:
            depth_img = depth_img.reshape((camera_info["height"], camera_info["width"]))
        depth_scale = compressed_image.get("depth_scale", 1000.0)
        return depth_img.astype(np.float32) / depth_scale
    else:
        raise ValueError(f"Unsupported image format: {fmt}")


class RobotController:
    """High-level controller for Galbot G1 robot."""

    _VLA_GROUP_DOF = {
        "left_arm": 7,
        "right_arm": 7,
        "left_gripper": 1,
        "right_gripper": 1,
    }

    def __init__(self, config: dict):
        self.cfg = config
        self.robot: Optional[GalbotRobot] = None
        self.nav: Optional[GalbotNavigation] = None
        self.motion: Optional[GalbotMotion] = None
        if G1JointGroup is not None:
            self._joint_group_map = {
                "left_arm": G1JointGroup.left_arm,
                "right_arm": G1JointGroup.right_arm,
                "left_gripper": G1JointGroup.left_gripper,
                "right_gripper": G1JointGroup.right_gripper,
                "head": G1JointGroup.head,
            }
        else:
            self._joint_group_map = {}

    def is_available(self) -> bool:
        return gm is not None

    def init(self) -> bool:
        if not self.is_available():
            print("[RobotController] galbot_sdk not available, running in mock mode")
            return False

        self.robot = GalbotRobot()
        self.nav = GalbotNavigation()
        self.motion = GalbotMotion()

        # Build sensor set from config for robot.init()
        sensor_set_cfg = self.cfg.get("sensor_set", [])
        sensor_set = []
        for name in sensor_set_cfg:
            sensor = getattr(SensorType, name, None)
            if sensor is not None:
                sensor_set.append(sensor)
            else:
                print(f"[RobotController] Warning: unknown sensor '{name}'")

        ok_robot = self.robot.init(set(sensor_set))
        ok_motion = self.motion.init()
        ok_nav = self.nav.init()

        print(f"[RobotController] Robot init: {ok_robot}, Motion init: {ok_motion}, Nav init: {ok_nav}")
        if not ok_robot:
            return False

        # Warm up sensors. Instead of a plain sleep, we continuously poll the
        # RGB cameras in the main thread: the galbot SDK camera stream tends to
        # stop producing data if it is not accessed for a few seconds, and
        # background-thread polling does not keep it alive for this SDK version.
        warmup = self.cfg.get("sensor_warmup_seconds", 5.0)
        if warmup > 0:
            print(f"[RobotController] Warming up sensors for {warmup}s with camera pumping...")
            self._pump_cameras(duration=warmup)

        if not ok_motion:
            print("[RobotController] Warning: motion init failed, continuing without motion services")

        if not ok_nav:
            return False

        # Ensure localization
        if not self.nav.is_localized():
            print("[RobotController] Not localized, attempting relocalization...")
            self.nav.relocalize([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            for _ in range(20):
                time.sleep(0.5)
                if self.nav.is_localized():
                    break

        return ok_robot and ok_nav

    def _pump_cameras(self, duration: float, interval: float = 0.1):
        """Keep the camera stream alive by polling from the main thread.

        Some SDK versions require camera access from the same thread that
        initialized the robot; background threads do not keep the stream alive.
        Pumping continues until all cameras have produced at least one frame or
        the maximum duration is reached.
        """
        camera_map = {
            "head": SensorType.HEAD_RIGHT_CAMERA,
            "left": SensorType.LEFT_ARM_CAMERA,
            "right": SensorType.RIGHT_ARM_CAMERA,
        }
        end_time = time.perf_counter() + duration
        first_success = {name: False for name in camera_map}
        all_success = False
        while time.perf_counter() < end_time and not all_success:
            all_success = True
            for name, sensor in camera_map.items():
                if first_success[name]:
                    continue
                # For slow cameras, try a small burst before moving on.
                for _ in range(3):
                    try:
                        rgb_data = self.robot.get_rgb_data(sensor)
                        if rgb_data:
                            img = decode_compressed_image(rgb_data)
                            print(f"[RobotController] Camera '{name}' first frame: shape={img.shape}")
                            first_success[name] = True
                            break
                    except Exception as e:
                        print(f"[RobotController] Camera pump error for '{name}': {e}")
                if not first_success[name]:
                    all_success = False
            if not all_success:
                time.sleep(interval)

        failed = [name for name, ok in first_success.items() if not ok]
        if failed:
            print(f"[RobotController] Warning: cameras that did not produce data during warmup: {failed}")

    def shutdown(self):
        if self.robot is not None:
            try:
                self.robot.request_shutdown()
                self.robot.wait_for_shutdown()
                self.robot.destroy()
                print("[RobotController] Robot shutdown complete")
            except Exception as e:
                print(f"[RobotController] Shutdown error: {e}")
        self.robot = None
        self.nav = None
        self.motion = None

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def get_current_pose(self) -> Optional[List[float]]:
        if self.nav is None:
            return None
        try:
            return list(self.nav.get_current_pose())
        except Exception as e:
            print(f"[RobotController] get_current_pose error: {e}")
            return None

    def navigate_to_goal(self, goal_pose: Sequence[float], timeout: float = 30.0, retry: int = 3) -> bool:
        if self.nav is None:
            print("[RobotController] Navigation not available")
            return False
        try:
            cur_pose = self.get_current_pose()
            print(f"[RobotController] Current pose: {cur_pose}")
            if not self.nav.check_path_reachability(goal_pose, cur_pose):
                print("[RobotController] Path unreachable or unsafe")
                return False

            for attempt in range(retry, 0, -1):
                status = self.nav.navigate_to_goal(
                    goal_pose, enable_collision_check=True, is_blocking=False, timeout=timeout
                )
                start = time.time()
                reached = False
                while time.time() - start < timeout:
                    time.sleep(0.5)
                    if self.nav.check_goal_arrival():
                        reached = True
                        break
                if reached:
                    print("[RobotController] Navigation goal reached")
                    self.nav.stop_navigation()
                    return True
                print(f"[RobotController] Navigation attempt failed, retry left: {attempt - 1}")
                time.sleep(0.5)

            self.nav.stop_navigation()
            return False
        except Exception as e:
            print(f"[RobotController] Navigation error: {e}")
            return False

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def _camera_sensor_map(self):
        if SensorType is None:
            return {}
        return {
            "head": SensorType.HEAD_RIGHT_CAMERA,
            "left": SensorType.LEFT_ARM_CAMERA,
            "right": SensorType.RIGHT_ARM_CAMERA,
        }

    def get_rgb_image(self, camera_name: str = "left") -> Optional[np.ndarray]:
        if self.robot is None:
            return None
        try:
            sensor_type = self._camera_sensor_map().get(camera_name)
            if sensor_type is None:
                return None
            rgb_data = None
            retries = self.cfg.get("camera_rgb_retries", 3)
            retry_delay = self.cfg.get("camera_rgb_retry_delay", 0.2)
            for attempt in range(retries):
                rgb_data = self.robot.get_rgb_data(sensor_type)
                if rgb_data:
                    break
                if attempt < retries - 1:
                    print(f"[RobotController] get_rgb_image({camera_name}) empty, retrying {attempt + 1}/{retries}...")
                    time.sleep(retry_delay)
            if not rgb_data:
                print(f"[RobotController] get_rgb_image({camera_name}) failed after {retries} attempts")
                return None
            return decode_compressed_image(rgb_data)
        except Exception as e:
            print(f"[RobotController] get_rgb_image({camera_name}) error: {e}")
            return None

    def get_depth_image(self, camera_name: str = "left_arm_depth_camera") -> Optional[np.ndarray]:
        if self.robot is None:
            return None
        try:
            if SensorType is None:
                return None
            sensor_type = SensorType.LEFT_ARM_DEPTH_CAMERA if "left" in camera_name else SensorType.RIGHT_ARM_DEPTH_CAMERA
            depth_data = self.robot.get_depth_data(sensor_type)
            if not depth_data:
                return None
            return decode_compressed_image(depth_data)
        except Exception as e:
            print(f"[RobotController] get_depth_image error: {e}")
            return None

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _vla_group_names(self) -> List[str]:
        group_names = self.cfg.get("vla_joint_groups")
        if not group_names:
            group_names = ["left_arm", "right_arm", "left_gripper", "right_gripper"]
        return [str(name) for name in group_names]

    def _vla_joint_groups(self, group_names: Optional[Sequence[str]] = None):
        names = list(group_names or self._vla_group_names())
        groups = []
        for name in names:
            group = self._joint_group_map.get(name)
            if group is None:
                raise ValueError(f"Unsupported VLA joint group: {name}")
            groups.append(group)
        return groups

    def _vla_group_dim(self, group_names: Optional[Sequence[str]] = None) -> int:
        names = list(group_names or self._vla_group_names())
        unknown = [name for name in names if name not in self._VLA_GROUP_DOF]
        if unknown:
            raise ValueError(f"Unsupported VLA joint group(s): {unknown}")
        return sum(self._VLA_GROUP_DOF[name] for name in names)

    def _vla_action_dim(self, group_names: Optional[Sequence[str]] = None) -> int:
        cfg_dim = self.cfg.get("vla_action_dim")
        if cfg_dim is not None:
            return int(cfg_dim)
        return self._vla_group_dim(group_names)

    def _vla_gripper_indices(self, group_names: Optional[Sequence[str]] = None) -> List[int]:
        names = list(group_names or self._vla_group_names())
        indices = []
        offset = 0
        for name in names:
            dof = self._VLA_GROUP_DOF.get(name)
            if dof is None:
                raise ValueError(f"Unsupported VLA joint group: {name}")
            if name.endswith("_gripper"):
                indices.extend(range(offset, offset + dof))
            offset += dof
        return indices

    def get_joint_state(self) -> Optional[np.ndarray]:
        """Get current VLA joint positions in configured vla_joint_groups order."""
        if self.robot is None:
            return None
        try:
            group_names = self._vla_group_names()
            expected_dim = self._vla_action_dim(group_names)
            group_dim = self._vla_group_dim(group_names)
            if expected_dim != group_dim:
                print(
                    f"[RobotController] VLA dim mismatch: vla_action_dim={expected_dim}, "
                    f"groups {group_names} expand to {group_dim}"
                )
                return None

            joint_groups = self._vla_joint_groups(group_names)
            joint_states = self.robot.get_joint_states(joint_groups, [])
            if joint_states:
                positions = [js.position for js in joint_states]
                if len(positions) != expected_dim:
                    print(
                        f"[RobotController] VLA state dim mismatch: got {len(positions)}, "
                        f"expected {expected_dim}"
                    )
                    return None
                arr = np.array(positions, dtype=np.float32)
                for idx in self._vla_gripper_indices(group_names):
                    arr[idx] = self._gripper_state_to_vla_value(arr[idx])
                return arr
        except Exception as e:
            print(f"[RobotController] get_joint_states error: {e}")
        return None

    def get_vla_state(self) -> Optional[np.ndarray]:
        """Return VLA state vector in configured vla_joint_groups order."""
        return self.get_joint_state()

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    def set_end_effector_pose(self, pose: Sequence[float], timeout: float = 5.0) -> bool:
        if self.motion is None:
            return False
        try:
            status = self.motion.set_end_effector_pose(
                target_pose=list(pose),
                end_effector_frame=self.cfg.get("end_effector_frame", "left_arm"),
                reference_frame=self.cfg.get("reference_frame", "base_link"),
                enable_collision_check=self.cfg.get("motion_collision_check", False),
                is_blocking=True,
                timeout=timeout,
                params=gm.Parameter()
            )
            ok = status == gm.MotionStatus.SUCCESS
            print(f"[RobotController] set_end_effector_pose status={status}, ok={ok}")
            return ok
        except Exception as e:
            print(f"[RobotController] set_end_effector_pose error: {e}")
            return False

    # ------------------------------------------------------------------
    # Gripper
    # ------------------------------------------------------------------

    def _gripper_velocity(self, velocity: Optional[float] = None) -> float:
        return float(self.cfg.get("gripper_velocity", 0.15) if velocity is None else velocity)

    def _gripper_effort(self, effort: Optional[float] = None) -> float:
        return float(self.cfg.get("gripper_effort", 10.0) if effort is None else effort)

    def _gripper_width_command_scale(self) -> float:
        return float(self.cfg.get("gripper_width_command_scale", 1.0))

    def _control_cfg(self) -> Dict[str, Any]:
        return self.cfg.get("control", {})

    def _set_joint_time_from_start(self) -> float:
        return float(
            self._control_cfg().get(
                "set_joint_time_from_start",
                self.cfg.get("set_joint_time_from_start", 0.0),
            )
        )

    def _gripper_action_scale(self) -> float:
        return float(self._control_cfg().get("gripper_action_scale", 0.0012))

    def _arm_max_delta_per_step(self, fallback: float) -> float:
        return float(self._control_cfg().get("max_joint_delta_per_step", fallback))

    def _gripper_max_delta_per_step(self) -> float:
        return float(self._control_cfg().get("gripper_max_delta_per_step", 20.0))

    def _gripper_max_width(self) -> float:
        return float(self.cfg.get("gripper_max_width", 0.12))

    def _gripper_command_position(self, value: float, scale: float) -> float:
        position = float(value) * float(scale)
        min_width = float(self.cfg.get("gripper_min_width", 0.0))
        max_width = self._gripper_max_width()
        return float(np.clip(position, min_width, max_width))

    def _gripper_state_to_vla_value(self, position: float) -> float:
        """Return gripper state in the VLA 0~100 unit range."""
        position = float(position)
        max_width = self._gripper_max_width()
        if abs(position) <= max_width * 1.5:
            return float(np.clip(position / max_width * 100.0, 0.0, 100.0))
        return position

    def _gripper_state_to_command_position(self, position: float) -> float:
        """Return current gripper state in set_joint_commands 0~0.12 units."""
        position = float(position)
        max_width = self._gripper_max_width()
        if abs(position) > max_width * 1.5:
            return self._gripper_command_position(position, self._gripper_action_scale())
        return self._gripper_command_position(position, 1.0)

    def _full_command_groups(self):
        return [
            G1JointGroup.left_arm,
            G1JointGroup.right_arm,
            G1JointGroup.left_gripper,
            G1JointGroup.right_gripper,
        ]

    def _build_full_joint_commands_with_grippers(
        self,
        left_position: Optional[float],
        right_position: Optional[float],
        velocity: Optional[float] = None,
        effort: Optional[float] = None,
    ) -> Optional[List[JointCommand]]:
        if self.robot is None:
            return None

        joint_states = self.robot.get_joint_states(self._full_command_groups(), [])
        if not joint_states or len(joint_states) < 16:
            print("[RobotController] Failed to get current state for full gripper command")
            return None

        target_positions = [float(js.position) for js in joint_states[:16]]
        target_positions[14] = self._gripper_state_to_command_position(target_positions[14])
        target_positions[15] = self._gripper_state_to_command_position(target_positions[15])
        if left_position is not None:
            target_positions[14] = float(left_position)
        if right_position is not None:
            target_positions[15] = float(right_position)

        gripper_velocity = self._gripper_velocity(velocity)
        gripper_effort = self._gripper_effort(effort)

        joint_commands = []
        for i, position in enumerate(target_positions):
            cmd = JointCommand()
            cmd.position = float(position)
            if i >= 14:
                cmd.velocity = gripper_velocity
                cmd.effort = gripper_effort
            else:
                cmd.velocity = 0.0
                cmd.effort = 0.0
            joint_commands.append(cmd)
        return joint_commands

    def set_gripper_positions(
        self,
        left_width: float,
        right_width: float,
        velocity: Optional[float] = None,
        effort: Optional[float] = None,
    ) -> bool:
        """Set both grippers to target widths in a single call."""
        if self.robot is None:
            return False
        try:
            velocity = self._gripper_velocity(velocity)
            effort = self._gripper_effort(effort)
            left_position = self._gripper_command_position(left_width, self._gripper_width_command_scale())
            right_position = self._gripper_command_position(right_width, self._gripper_width_command_scale())
            joint_commands = self._build_full_joint_commands_with_grippers(
                left_position,
                right_position,
                velocity,
                effort,
            )
            if joint_commands is None:
                return False
            status = self.robot.set_joint_commands(joint_commands, self._full_command_groups(), [], 0.0)
            if status != gm.ControlStatus.SUCCESS:
                print(f"[RobotController] set_joint_commands grippers status={status}")
                return False
            print(
                f"[RobotController] Both grippers set: left={left_width}, "
                f"right={right_width}, command_positions="
                f"{[joint_commands[14].position, joint_commands[15].position]}, velocity={velocity}"
            )
            time.sleep(0.3)
            return True
        except Exception as e:
            print(f"[RobotController] set_gripper_positions error: {e}")
            return False

    def set_gripper(
        self,
        width: float,
        velocity: Optional[float] = None,
        effort: Optional[float] = None,
        is_moving: bool = False,
    ) -> bool:
        if self.robot is None:
            return False
        try:
            velocity = self._gripper_velocity(velocity)
            effort = self._gripper_effort(effort)
            group = self._joint_group_map.get(self.cfg.get("gripper_group", "left_gripper"))
            if group is None:
                group = G1JointGroup.left_gripper
            command_position = self._gripper_command_position(width, self._gripper_width_command_scale())
            left_position = command_position if group == G1JointGroup.left_gripper else None
            right_position = command_position if group == G1JointGroup.right_gripper else None
            joint_commands = self._build_full_joint_commands_with_grippers(
                left_position,
                right_position,
                velocity,
                effort,
            )
            if joint_commands is None:
                return False
            status = self.robot.set_joint_commands(joint_commands, self._full_command_groups(), [], 0.0)
            ok = status == gm.ControlStatus.SUCCESS
            print(
                f"[RobotController] set_gripper width={width}, "
                f"command_position={command_position}, velocity={velocity}, ok={ok}"
            )
            time.sleep(0.3)
            return ok
        except Exception as e:
            print(f"[RobotController] set_gripper error: {e}")
            return False

    def open_gripper(self):
        placement = self.cfg.get("placement", {})
        return self.set_gripper(placement.get("gripper_open_width", self.cfg.get("gripper_open_width", 0.1)))

    def close_gripper(self):
        placement = self.cfg.get("placement", {})
        return self.set_gripper(placement.get("gripper_close_width", self.cfg.get("gripper_close_width", 0.02)))

    # ------------------------------------------------------------------
    # Initial Pose
    # ------------------------------------------------------------------

    def set_arm_position(self, positions: Sequence[float], timeout: float = 5.0) -> bool:
        """Set arm joints to a target position (left 7 + right 7, 14-DOF).
        Grippers are left unchanged."""
        if self.robot is None or JointCommand is None:
            print("[RobotController] Robot not available, cannot set arm position")
            return False

        target = np.asarray(positions, dtype=np.float32)
        if target.shape[0] != 14:
            print(f"[RobotController] set_arm_position expects 14 values, got {target.shape[0]}")
            return False

        joint_groups = [G1JointGroup.left_arm, G1JointGroup.right_arm]
        joint_commands = [JointCommand() for _ in range(14)]
        for i, cmd in enumerate(joint_commands):
            cmd.position = float(target[i])

        time_from_start = self._set_joint_time_from_start()
        try:
            status = self.robot.set_joint_commands(
                joint_commands,
                joint_groups,
                [],
                time_from_start,
            )
            ok = status == gm.ControlStatus.SUCCESS
            print(f"[RobotController] set_arm_position status={status}, ok={ok}")
            if ok:
                time.sleep(timeout)
            return ok
        except Exception as e:
            print(f"[RobotController] set_arm_position error: {e}")
            return False

    # ------------------------------------------------------------------
    # Joint Commands
    # ------------------------------------------------------------------

    def execute_action_chunk(self, actions: List[List[float]], dt: float, max_delta: float = 0.3):
        """Execute a list of joint angle actions via set_joint_commands.
        Applies per-step delta limiting for safety because set_joint_commands
        does not interpolate from the current position.
        """
        if self.robot is None or JointCommand is None:
            print("[RobotController] Robot not available, skipping action execution")
            return

        group_names = self._vla_group_names()
        joint_groups = self._vla_joint_groups(group_names)
        joint_names = []
        action_dim = self._vla_action_dim(group_names)
        group_dim = self._vla_group_dim(group_names)
        if action_dim != group_dim:
            raise ValueError(
                f"VLA dim mismatch: vla_action_dim={action_dim}, "
                f"groups {group_names} expand to {group_dim}"
            )

        joint_commands = [JointCommand() for _ in range(action_dim)]
        gripper_indices = self._vla_gripper_indices(group_names)
        gripper_index_set = set(gripper_indices)
        arm_indices = [idx for idx in range(action_dim) if idx not in gripper_index_set]
        gripper_velocity = self._gripper_velocity()
        gripper_effort = self._gripper_effort()
        gripper_action_scale = self._gripper_action_scale()
        arm_max_delta = self._arm_max_delta_per_step(max_delta)
        gripper_max_delta = self._gripper_max_delta_per_step()

        # Get current joint positions for delta limiting on first step
        current_pos = self.get_joint_state()
        if current_pos is None:
            current_pos = np.zeros(action_dim, dtype=np.float32)
        else:
            current_pos = np.asarray(current_pos, dtype=np.float32)
        if current_pos.shape[0] != action_dim:
            raise ValueError(
                f"Current state dim mismatch: got {current_pos.shape[0]}, expected {action_dim}"
            )

        for i, action in enumerate(actions):
            action_arr = np.asarray(action, dtype=np.float32)
            if action_arr.shape[0] != action_dim:
                raise ValueError(
                    f"Action dim mismatch at step {i}: got {action_arr.shape[0]}, expected {action_dim}"
                )

            # Apply delta limiting for safety (set_joint_commands does not interpolate)
            delta = action_arr - current_pos
            clipped_delta = np.zeros_like(delta)
            if arm_indices:
                clipped_delta[arm_indices] = np.clip(
                    delta[arm_indices],
                    -arm_max_delta,
                    arm_max_delta,
                )
            if gripper_indices:
                clipped_delta[gripper_indices] = np.clip(
                    delta[gripper_indices],
                    -gripper_max_delta,
                    gripper_max_delta,
                )
            safe_action = current_pos + clipped_delta

            for j, cmd in enumerate(joint_commands):
                if j in gripper_index_set:
                    cmd.position = self._gripper_command_position(safe_action[j], gripper_action_scale)
                    cmd.velocity = gripper_velocity
                    cmd.effort = gripper_effort
                else:
                    cmd.position = float(safe_action[j])
                    cmd.velocity = 0.0
                    cmd.effort = 0.0

            time_from_start = self._set_joint_time_from_start()
            try:
                status = self.robot.set_joint_commands(
                    joint_commands,
                    joint_groups,
                    joint_names,
                    time_from_start
                )
                if status != gm.ControlStatus.SUCCESS:
                    print(f"[RobotController] set_joint_commands status={status} at step {i}")
            except Exception as e:
                print(f"[RobotController] set_joint_commands error at step {i}: {e}")

            current_pos = safe_action
            time.sleep(dt)

    # ------------------------------------------------------------------
    # Observation (for VLA)
    # ------------------------------------------------------------------

    def get_observation(self) -> Optional[Dict[str, Any]]:
        """Get a full observation dict for VLA inference.

        Returns a raw observation compatible with LeRobot's build_dataset_frame:
        - Three camera images as numpy arrays (head, left, right)
        - State expanded into individual joint keys
        """
        # 1. Capture all three cameras
        images = {}
        for cam_name, out_key in (
            ("head", "head_right"),
            ("left", "left_arm"),
            ("right", "right_arm"),
        ):
            img = self.get_rgb_image(cam_name)
            if img is None:
                print(f"[RobotController] Warning: failed to get image from camera '{cam_name}'")
                img = np.zeros((720, 1280, 3), dtype=np.uint8)
            images[out_key] = img

        # 2. Get configured VLA state
        action_dim = self._vla_action_dim()
        state = self.get_vla_state()
        if state is None:
            state = np.zeros(action_dim, dtype=np.float32)

        # 3. Build raw observation dict matching lerobot_features names
        obs: Dict[str, Any] = {**images}
        state_names = self.cfg.get("state_feature_names", [])
        if len(state_names) == len(state):
            for name, value in zip(state_names, state):
                obs[name] = float(value)
        else:
            # Fallback: pack entire state under a single key (won't work with
            # standard build_dataset_frame, but preserves backward compat)
            obs["observation.state"] = state

        return obs
