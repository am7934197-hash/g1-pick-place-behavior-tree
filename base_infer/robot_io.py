"""In-place Galbot G1 interface: observe, restore pose, and execute pick actions."""

import time
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

try:
    import galbot_sdk.g1 as gm
    from galbot_sdk.g1 import GalbotRobot, G1JointGroup, JointCommand, SensorType
except ImportError:  # Allows --help and unit tests on machines without the SDK.
    gm = GalbotRobot = G1JointGroup = JointCommand = SensorType = None


def decode_rgb(compressed: dict) -> np.ndarray:
    if not compressed or compressed.get("format") != "rgb8":
        raise ValueError(f"unsupported camera frame format: {compressed and compressed.get('format')}")
    encoded = np.frombuffer(compressed["data"], dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("failed to decode camera frame")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


FULL_STATE_DIM = 23
GRIPPER_INDICES = (7, 15)
EXECUTE_GROUPS = (
    ("right_arm", 7, False),
    ("right_gripper", 1, True),
    ("left_arm", 7, False),
    ("left_gripper", 1, True),
)


def expand_model_action(action: Sequence[float], current_state: np.ndarray, model_dim: int) -> np.ndarray:
    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
    state_arr = np.asarray(current_state, dtype=np.float32).reshape(-1)
    if action_arr.size != model_dim:
        raise ValueError(f"action dim {action_arr.size}, expected {model_dim}")
    if state_arr.size != FULL_STATE_DIM:
        raise ValueError(f"state dim {state_arr.size}, expected {FULL_STATE_DIM}")
    if model_dim == FULL_STATE_DIM:
        return action_arr.copy()
    expanded = state_arr.copy()
    expanded[:model_dim] = action_arr
    return expanded


def linear_interpolate(waypoints: np.ndarray, input_dt: float, output_dt: float) -> np.ndarray:
    waypoints = np.asarray(waypoints, dtype=np.float32)
    if len(waypoints) < 2:
        return waypoints
    t_in = np.arange(len(waypoints), dtype=np.float64) * float(input_dt)
    total = (len(waypoints) - 1) * float(input_dt)
    intervals = max(int(np.ceil(total / float(output_dt))), 1)
    t_out = np.linspace(0.0, total, intervals + 1, dtype=np.float64)
    result = np.empty((len(t_out), waypoints.shape[1]), dtype=np.float32)
    for joint in range(waypoints.shape[1]):
        result[:, joint] = np.interp(t_out, t_in, waypoints[:, joint].astype(np.float64))
    return result


def gripper_command_position(value: float, scale: float, min_width: float, max_width: float) -> float:
    return float(np.clip(float(value) * float(scale), min_width, max_width))


class StationaryRobot:
    """Stay in place: no navigation. Restore pose, then execute pick chunks."""

    CAMERA_SENSORS = {
        "head_right": "HEAD_RIGHT_CAMERA",
        "left_arm": "LEFT_ARM_CAMERA",
        "right_arm": "RIGHT_ARM_CAMERA",
    }
    GROUP_DOF = {
        "right_arm": 7,
        "right_gripper": 1,
        "left_arm": 7,
        "left_gripper": 1,
        "leg": 5,
        "head": 2,
    }

    def __init__(self, config: dict):
        self.config = config
        self.robot = None

    def initialize(self) -> None:
        if GalbotRobot is None:
            raise RuntimeError("galbot_sdk is not installed")
        sensors = set()
        for name in self.config.get("sensor_set", []):
            sensor = getattr(SensorType, str(name), None)
            if sensor is None:
                raise ValueError(f"unknown sensor: {name}")
            sensors.add(sensor)
        self.robot = GalbotRobot()
        if not self.robot.init(sensors):
            self.robot = None
            raise RuntimeError("GalbotRobot.init() failed")
        self._warm_up(float(self.config.get("sensor_warmup_seconds", 10.0)))

    def close(self) -> None:
        robot, self.robot = self.robot, None
        if robot is None:
            return
        try:
            robot.request_shutdown()
        finally:
            robot.destroy()

    def _warm_up(self, duration: float) -> None:
        deadline = time.monotonic() + max(0.0, duration)
        ready = set()
        while time.monotonic() < deadline and len(ready) < len(self.CAMERA_SENSORS):
            for camera in self.CAMERA_SENSORS:
                if camera in ready:
                    continue
                try:
                    image = self.capture_image(camera, retries=1)
                    if image is not None:
                        ready.add(camera)
                        print(f"[robot] {camera} ready: {image.shape}")
                except Exception:
                    pass
            if len(ready) < len(self.CAMERA_SENSORS):
                time.sleep(0.1)
        missing = sorted(set(self.CAMERA_SENSORS) - ready)
        if missing:
            raise RuntimeError(f"camera warm-up failed: {missing}")

    def capture_image(self, camera: str, retries: Optional[int] = None) -> Optional[np.ndarray]:
        if self.robot is None:
            raise RuntimeError("robot is not initialized")
        sensor_name = self.CAMERA_SENSORS.get(camera)
        if sensor_name is None:
            raise ValueError(f"unknown camera: {camera}")
        sensor = getattr(SensorType, sensor_name)
        attempts = int(retries or self.config.get("camera_retries", 5))
        delay = float(self.config.get("camera_retry_delay", 0.2))
        for attempt in range(attempts):
            frame = self.robot.get_rgb_data(sensor)
            if frame:
                return decode_rgb(frame)
            if attempt + 1 < attempts:
                time.sleep(delay)
        return None

    def read_state(self) -> np.ndarray:
        if self.robot is None:
            raise RuntimeError("robot is not initialized")
        group_names = list(self.config.get("joint_groups", ["right_arm", "right_gripper"]))
        groups = []
        expected = 0
        gripper_indices = []
        offset = 0
        for name in group_names:
            if name not in self.GROUP_DOF:
                raise ValueError(f"unsupported joint group: {name}")
            groups.append(getattr(G1JointGroup, name))
            dof = self.GROUP_DOF[name]
            if name.endswith("_gripper"):
                gripper_indices.extend(range(offset, offset + dof))
            expected += dof
            offset += dof
        states = self.robot.get_joint_states(groups, [])
        if not states or len(states) != expected:
            raise RuntimeError(f"joint state size {len(states) if states else 0}, expected {expected}")
        values = np.asarray([item.position for item in states], dtype=np.float32)
        max_width = float(self.config.get("gripper_max_width", 0.12))
        for index in gripper_indices:
            if abs(float(values[index])) <= max_width * 1.5:
                values[index] = np.clip(values[index] / max_width * 100.0, 0.0, 100.0)
        if not np.all(np.isfinite(values)):
            raise RuntimeError("joint state contains non-finite values")
        return values

    def set_initial_pose(self) -> None:
        """Restore and verify the configured training-start joint pose."""
        positions = self.config.get("init_joint_positions") or {}
        if not positions:
            return
        restore = self.config.get("init_pose_restore") or {}
        timeout = float(restore.get("timeout_seconds", 8.0))
        retries = max(1, int(restore.get("retries", 2)))
        interval = max(0.01, float(restore.get("poll_interval_seconds", 0.1)))
        stable_required = max(1, int(restore.get("stable_samples", 2)))
        retry_delay = max(0.0, float(restore.get("retry_delay_seconds", 0.2)))
        if timeout <= 0:
            raise ValueError("init_pose_restore.timeout_seconds must be positive")

        last_report = None
        for attempt in range(1, retries + 1):
            command_ok = self._send_initial_pose(positions)
            print(f"[robot] initial pose attempt={attempt}/{retries} command_ok={command_ok}")
            if command_ok:
                deadline = time.monotonic() + timeout
                stable = 0
                while time.monotonic() < deadline:
                    ok, last_report = self._verify_initial_pose(positions)
                    stable = stable + 1 if ok else 0
                    if stable >= stable_required:
                        print(f"[robot] initial pose reached: {last_report}")
                        return
                    time.sleep(min(interval, max(0.0, deadline - time.monotonic())))
            if attempt < retries:
                time.sleep(retry_delay)
        raise RuntimeError(f"failed to reach initial pose: {last_report}")

    def _send_initial_pose(self, positions: dict) -> bool:
        if self.robot is None or JointCommand is None:
            return False
        commands = []
        groups = []
        for name, values in positions.items():
            if name not in self.GROUP_DOF:
                raise ValueError(f"unsupported initial-pose group: {name}")
            if len(values) != self.GROUP_DOF[name]:
                raise ValueError(
                    f"initial pose {name} has {len(values)} values, "
                    f"expected {self.GROUP_DOF[name]}"
                )
            groups.append(getattr(G1JointGroup, name))
            is_gripper = name.endswith("_gripper")
            for value in values:
                command = JointCommand()
                command.position = float(value)
                command.velocity = (
                    float(self.config.get("gripper_velocity", 0.15)) if is_gripper else 0.0
                )
                command.effort = (
                    float(self.config.get("gripper_effort", 10.0)) if is_gripper else 0.0
                )
                commands.append(command)
        status = self.robot.set_joint_commands(
            commands,
            groups,
            [],
            float(self.config.get("set_joint_time_from_start", 0.0)),
        )
        return status == gm.ControlStatus.SUCCESS

    def _verify_initial_pose(self, positions: dict):
        measured = self.read_state()
        group_names = list(self.config.get("joint_groups", []))
        offsets = {}
        offset = 0
        for name in group_names:
            size = self.GROUP_DOF[name]
            offsets[name] = (offset, offset + size)
            offset += size

        verify = self.config.get("init_pose_verify") or {}
        joint_tolerance = float(verify.get("joint_tolerance_rad", 0.03))
        gripper_tolerance = float(verify.get("gripper_tolerance_model", 3.0))
        max_width = float(self.config.get("gripper_max_width", 0.12))
        report = {}
        all_ok = True
        for name, values in positions.items():
            if name not in offsets:
                raise ValueError(f"initial pose group {name} is absent from joint_groups")
            start, end = offsets[name]
            target = np.asarray(values, dtype=np.float32)
            tolerance = joint_tolerance
            if name.endswith("_gripper"):
                target = np.clip(target / max_width * 100.0, 0.0, 100.0)
                tolerance = gripper_tolerance
            error = float(np.max(np.abs(measured[start:end] - target)))
            ok = error <= tolerance
            report[name] = {"max_error": round(error, 6), "tolerance": tolerance, "ok": ok}
            all_ok = all_ok and ok
        return all_ok, report

    def capture_observation(self) -> Dict[str, object]:
        observation: Dict[str, object] = {}
        for camera in self.CAMERA_SENSORS:
            image = self.capture_image(camera)
            if image is None:
                raise RuntimeError(f"failed to capture {camera}")
            observation[camera] = image
        state = self.read_state()
        names = list(self.config.get("state_feature_names", []))
        if len(names) != len(state):
            raise ValueError(f"state_feature_names has {len(names)} items, state has {len(state)}")
        observation.update({name: float(value) for name, value in zip(names, state)})
        return observation

    def execute_action_chunk(self, actions: List[Sequence[float]]) -> bool:
        """Execute a pick-model chunk. Arms and grippers only; leg/head stay put."""
        if self.robot is None or JointCommand is None:
            raise RuntimeError("robot is not initialized")
        control = self.config.get("control") or {}
        model_dim = int(self.config.get("vla_action_dim", 16))
        current = self.read_state()
        arm_max_delta = float(control.get("max_joint_delta_per_step", 0.05))
        gripper_max_delta = float(control.get("gripper_max_delta_per_step", 20.0))
        scale = float(control.get("gripper_action_scale", 0.0012))
        min_width = float(self.config.get("gripper_min_width", 0.0))
        max_width = float(self.config.get("gripper_max_width", 0.12))
        limits = self._joint_limits()
        interpolation = control.get("interpolation") or {}
        interp_enabled = bool(interpolation.get("enabled", True))
        input_hz = float(interpolation.get("input_hz", 30))
        output_hz = float(interpolation.get("output_hz", 250))
        input_dt = 1.0 / input_hz
        output_dt = 1.0 / output_hz

        safe = []
        cursor = current.copy()
        for action in actions:
            target = expand_model_action(action, cursor, model_dim)
            if not np.all(np.isfinite(target)):
                raise ValueError("model action contains non-finite values")
            delta = target - cursor
            clipped = np.empty(FULL_STATE_DIM, dtype=np.float32)
            for index in range(FULL_STATE_DIM):
                limit = gripper_max_delta if index in GRIPPER_INDICES else arm_max_delta
                clipped[index] = np.clip(delta[index], -limit, limit)
            next_target = cursor + clipped
            if limits is not None:
                next_target = np.clip(next_target, limits[:, 0], limits[:, 1])
            safe.append(next_target)
            cursor = next_target
        if not safe:
            return False

        if interp_enabled:
            waypoints = np.vstack([current, np.stack(safe)])
            frames = linear_interpolate(waypoints, input_dt, output_dt)
            send_dt = (len(safe) * input_dt) / max(len(frames) - 1, 1)
        else:
            frames = np.stack(safe)
            send_dt = input_dt

        groups = [getattr(G1JointGroup, name) for name, _, _ in EXECUTE_GROUPS]
        commands = [JointCommand() for _ in range(sum(size for _, size, _ in EXECUTE_GROUPS))]
        gripper_velocity = float(self.config.get("gripper_velocity", 0.15))
        gripper_effort = float(self.config.get("gripper_effort", 10.0))
        time_from_start = float(self.config.get("set_joint_time_from_start", 0.0))

        for index, frame in enumerate(frames):
            if limits is not None:
                frame = np.clip(frame, limits[:, 0], limits[:, 1])
            offset = 0
            for name, size, is_gripper in EXECUTE_GROUPS:
                del name
                for local in range(size):
                    command = commands[offset + local]
                    value = float(frame[offset + local])
                    if is_gripper:
                        command.position = gripper_command_position(
                            value, scale, min_width, max_width
                        )
                        command.velocity = gripper_velocity
                        command.effort = gripper_effort
                    else:
                        command.position = value
                        command.velocity = 0.0
                        command.effort = 0.0
                offset += size
            status = self.robot.set_joint_commands(commands, groups, [], time_from_start)
            if status != gm.ControlStatus.SUCCESS:
                print(f"[robot] set_joint_commands status={status} frame={index}/{len(frames)}")
                if status == gm.ControlStatus.FAULT:
                    return False
            if send_dt > 0 and index + 1 < len(frames):
                time.sleep(send_dt)
        return True

    def _joint_limits(self) -> Optional[np.ndarray]:
        limits = (self.config.get("control") or {}).get("joint_limits")
        if not limits or len(limits) != FULL_STATE_DIM:
            return None
        return np.asarray(limits, dtype=np.float32)
