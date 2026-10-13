"""
VLA gRPC client for direct communication with PolicyServer.
No lerobot dependency. Uses protoc-generated pb2 files and pickle-compatible
dataclasses matching lerobot internals.
"""

import os
import sys
import time
from typing import List, Dict, Any, Optional

import numpy as np
import grpc

# Add current dir to path so we can import generated pb2
sys.path.insert(0, os.path.dirname(__file__))
from lerobot.transport import services_pb2, services_pb2_grpc

# Internal trusted serialization: PolicyServer protocol requires pickle
# for TimedObservation and RemotePolicyConfig by design.
import pickle as _pickle  # nosec
import types as _types

CHUNK_SIZE = 2 * 1024 * 1024
TRANSFER_BEGIN = 1
TRANSFER_MIDDLE = 2
TRANSFER_END = 3


# ------------------------------------------------------------------
# Pickle-compatible dataclasses (must match lerobot internals exactly)
# ------------------------------------------------------------------

class TimedObservation:
    def __init__(self, timestamp: float, timestep: int, observation: dict, must_go: bool = False):
        self.timestamp = timestamp
        self.timestep = timestep
        self.observation = observation
        self.must_go = must_go

    def get_timestamp(self):
        return self.timestamp

    def get_timestep(self):
        return self.timestep

    def get_observation(self):
        return self.observation


class TimedAction:
    def __init__(self, timestamp: float, timestep: int, action):
        self.timestamp = timestamp
        self.timestep = timestep
        self.action = action

    def get_timestamp(self):
        return self.timestamp

    def get_timestep(self):
        return self.timestep

    def get_action(self):
        return self.action


class RemotePolicyConfig:
    def __init__(self, policy_type: str, pretrained_name_or_path: str, lerobot_features: dict,
                 actions_per_chunk: int, device: str = "cpu", rename_map: dict = None):
        self.policy_type = policy_type
        self.pretrained_name_or_path = pretrained_name_or_path
        self.lerobot_features = lerobot_features
        self.actions_per_chunk = actions_per_chunk
        self.device = device
        self.rename_map = rename_map or {}
        self.acp_enable = False
        self.acp_use_cfg = False
        self.acp_cfg_beta = 1.0


# Register a stub module so pickle believes these classes live in
# lerobot.async_inference.helpers, which exists on the remote PolicyServer.
# This avoids the need to install vla_client on the server.
for _mod_name in ("lerobot.async_inference.helpers", "lerobot.async_inference"):
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = _types.ModuleType(_mod_name)

_helpers_mod = sys.modules["lerobot.async_inference.helpers"]
_helpers_mod.TimedObservation = TimedObservation
_helpers_mod.TimedAction = TimedAction
_helpers_mod.RemotePolicyConfig = RemotePolicyConfig

TimedObservation.__module__ = "lerobot.async_inference.helpers"
TimedObservation.__qualname__ = "TimedObservation"
TimedAction.__module__ = "lerobot.async_inference.helpers"
TimedAction.__qualname__ = "TimedAction"
RemotePolicyConfig.__module__ = "lerobot.async_inference.helpers"
RemotePolicyConfig.__qualname__ = "RemotePolicyConfig"


# ------------------------------------------------------------------
# gRPC helper: send bytes in streaming chunks
# ------------------------------------------------------------------

def _send_observations(stub, timed_obs: TimedObservation):
    obs_bytes = _pickle.dumps(timed_obs)
    messages = []
    total = len(obs_bytes)
    sent = 0
    while sent < total:
        if sent + CHUNK_SIZE >= total:
            state = TRANSFER_END
        elif sent == 0:
            state = TRANSFER_BEGIN
        else:
            state = TRANSFER_MIDDLE
        chunk = obs_bytes[sent: sent + CHUNK_SIZE]
        messages.append(services_pb2.Observation(transfer_state=state, data=chunk))
        sent += len(chunk)
    stub.SendObservations(iter(messages))


def _get_actions(stub) -> list:
    actions_msg = stub.GetActions(services_pb2.Empty())
    if len(actions_msg.data) == 0:
        return []
    timed_actions = _pickle.loads(actions_msg.data)
    result = []
    for ta in timed_actions:
        action_list = ta.action.detach().cpu().numpy().tolist()
        if isinstance(action_list, list) and len(action_list) > 0 and isinstance(action_list[0], list):
            action_list = action_list[0]
        result.append({
            "timestep": ta.timestep,
            "timestamp": ta.timestamp,
            "action": action_list,
        })
    return result


# ------------------------------------------------------------------
# VLAClient
# ------------------------------------------------------------------

class VLAClient:
    def __init__(self, config: dict):
        self.cfg = config
        self.server_address = config.get("policy_server_address", "localhost:8080")
        self._channel = None
        self._stub = None

    def connect(self) -> bool:
        try:
            self._channel = grpc.insecure_channel(self.server_address)
            grpc.channel_ready_future(self._channel).result(timeout=10)
            self._stub = services_pb2_grpc.AsyncInferenceStub(self._channel)
            self._stub.Ready(services_pb2.Empty())
            print(f"[VLAClient] Connected to PolicyServer at {self.server_address}")
            return True
        except Exception as e:
            print(f"[VLAClient] Failed to connect: {e}")
            return False

    def disconnect(self):
        if self._channel:
            self._channel.close()
            self._channel = None
            self._stub = None
            print("[VLAClient] Disconnected")

    @property
    def connected(self) -> bool:
        return self._stub is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup_policy(self) -> bool:
        if not self.connected:
            print("[VLAClient] Not connected")
            return False
        try:
            remote_cfg = RemotePolicyConfig(
                policy_type=self.cfg.get("policy_type", "pi0"),
                pretrained_name_or_path=self.cfg.get("pretrained_name_or_path", ""),
                lerobot_features=self.cfg.get("observation_features", {}),
                actions_per_chunk=self.cfg.get("actions_per_chunk", 50),
                device=self.cfg.get("device", "cpu"),
            )
            cfg_bytes = _pickle.dumps(remote_cfg)
            self._stub.SendPolicyInstructions(services_pb2.PolicySetup(data=cfg_bytes))
            print("[VLAClient] Policy setup sent")
            return True
        except Exception as e:
            print(f"[VLAClient] setup_policy error: {e}")
            return False

    def update_instruction(self, instruction: str) -> bool:
        if not self.connected:
            return False
        try:
            instr_bytes = _pickle.dumps(instruction)
            self._stub.UpdateInstruction(services_pb2.InstructionUpdate(data=instr_bytes))
            print(f"[VLAClient] Instruction updated: {instruction}")
            return True
        except Exception as e:
            print(f"[VLAClient] update_instruction error: {e}")
            return False

    def reset(self) -> bool:
        if not self.connected:
            return False
        try:
            self._stub.ResetPolicy(services_pb2.Empty())
            print("[VLAClient] Policy reset")
            return True
        except Exception as e:
            print(f"[VLAClient] reset error: {e}")
            return False

    def infer(self, obs_dict: dict, instruction: str, timestep: int) -> Optional[List[dict]]:
        """Run inference and return list of action dicts.
        Each action: {"timestep": int, "action": List[float]}
        """
        if not self.connected:
            print("[VLAClient] Not connected")
            return None
        try:
            obs_dict["task"] = instruction
            timed_obs = TimedObservation(
                timestamp=time.time(),
                timestep=timestep,
                observation=obs_dict,
                must_go=True,
            )
            _send_observations(self._stub, timed_obs)
            actions = _get_actions(self._stub)
            print(f"[VLAClient] Received {len(actions)} actions")
            return actions
        except Exception as e:
            print(f"[VLAClient] infer error: {e}")
            return None
        except Exception as e:
            print(f"[VLAClient] infer error: {e}")
            return None
