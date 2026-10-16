"""LeRobot async-inference wire data types used by the G1 client.

This is the lightweight, robot-side subset of LeRobot's helpers module.  It
keeps the objects at their canonical module path so pickle payloads are
accepted by ``lerobot.async_inference.policy_server`` without loading the
policy stack on the control computer.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TimedData:
    timestamp: float
    timestep: int

    def get_timestamp(self):
        return self.timestamp

    def get_timestep(self):
        return self.timestep


@dataclass
class TimedAction(TimedData):
    action: Any

    def get_action(self):
        return self.action


@dataclass
class TimedObservation(TimedData):
    observation: dict
    must_go: bool = False
    trace_id: str = ""
    previous_actions: Any = None
    previous_action_start_index: int = 0
    control_step: int = 0
    trajectory_step: int = 0
    fps: float = 0.0

    def get_observation(self):
        return self.observation


@dataclass
class RemotePolicyConfig:
    policy_type: str
    pretrained_name_or_path: str
    lerobot_features: dict
    actions_per_chunk: int
    device: str = "cpu"
    rename_map: dict = field(default_factory=dict)
    rtc_config: Optional[dict] = None

    # These fields are understood by the deployed LeRobot PolicyServer fork.
    # Keeping them here is harmless with upstream servers and preserves wire
    # compatibility with existing checkpoints.
    acp_enable: bool = False
    acp_use_cfg: bool = False
    acp_cfg_beta: float = 1.0
