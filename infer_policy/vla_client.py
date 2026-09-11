"""
VLA gRPC client for direct communication with PolicyServer.
No LeRobot dependency; uses the small wire messages required by its RPC API.
"""

import sys
import time
from io import BytesIO
from typing import List, Dict, Any, Optional

import numpy as np
import grpc

def _varint(value):
    result = bytearray()
    value = int(value)
    while value >= 128:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value)
    return bytes(result)


def _read_varint(payload, index):
    value = 0
    shift = 0
    while index < len(payload):
        byte = payload[index]
        index += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, index
        shift += 7
    raise ValueError("truncated protobuf varint")


def _encode_varint_field(field_number, value):
    tag = (int(field_number) << 3) | 0
    return _varint(tag) + _varint(value)


def _encode_bytes_field(field_number, data):
    data = data or b""
    if not isinstance(data, (bytes, bytearray)):
        data = bytes(data)
    tag = (int(field_number) << 3) | 2
    return _varint(tag) + _varint(len(data)) + data


def _decode_fields(payload):
    fields = {}
    index = 0
    payload = payload or b""
    while index < len(payload):
        tag, index = _read_varint(payload, index)
        field_number = tag >> 3
        wire_type = tag & 7
        if wire_type == 0:
            value, index = _read_varint(payload, index)
            fields[field_number] = value
        elif wire_type == 2:
            size, index = _read_varint(payload, index)
            fields[field_number] = payload[index:index + size]
            index += size
        else:
            raise ValueError(f"unsupported protobuf wire type {wire_type}")
    return fields


class _Empty:
    def __init__(self, data=b"", transfer_state=0):
        self.data = data or b""
        self.transfer_state = transfer_state

    def SerializeToString(self):
        return b""

    @classmethod
    def FromString(cls, payload):
        del payload
        return cls()


class _BytesField1:
    """proto3 messages whose only payload is `bytes data = 1`.

    PolicySetup, Actions and InstructionUpdate all use this layout. Encoding
    the pickle bytes as field 2 leaves the server with empty `data`, and
    pickle.loads(b'') raises 'Ran out of input'.
    """

    def __init__(self, data=b"", transfer_state=0):
        self.data = data or b""
        self.transfer_state = transfer_state

    def SerializeToString(self):
        return _encode_bytes_field(1, self.data)

    @classmethod
    def FromString(cls, payload):
        fields = _decode_fields(payload)
        return cls(data=fields.get(1, b""))


class _Observation:
    """Observation: TransferState transfer_state = 1; bytes data = 2."""

    def __init__(self, data=b"", transfer_state=0):
        self.data = data or b""
        self.transfer_state = transfer_state

    def SerializeToString(self):
        return _encode_varint_field(1, self.transfer_state) + _encode_bytes_field(2, self.data)

    @classmethod
    def FromString(cls, payload):
        fields = _decode_fields(payload)
        return cls(data=fields.get(2, b""), transfer_state=fields.get(1, 0))


class _Proto:
    Empty = _Empty
    Observation = _Observation
    Actions = _BytesField1
    PolicySetup = _BytesField1
    InstructionUpdate = _BytesField1


services_pb2 = _Proto()

# Internal trusted serialization: PolicyServer protocol requires pickle
# for TimedObservation and RemotePolicyConfig by design.
import pickle as _pickle  # nosec
import types as _types

CHUNK_SIZE = 2 * 1024 * 1024
TRANSFER_BEGIN = 1
TRANSFER_MIDDLE = 2
TRANSFER_END = 3


class _PolicyRpcClient:
    """Small version-independent client for the PolicyServer RPC methods.

    It avoids generated gRPC stub modules that vary across LeRobot versions.
    """

    def __init__(self, channel):
        prefix = "/transport.AsyncInference"
        self.SendObservations = channel.stream_unary(
            f"{prefix}/SendObservations",
            request_serializer=services_pb2.Observation.SerializeToString,
            response_deserializer=services_pb2.Empty.FromString,
        )
        self.GetActions = channel.unary_unary(
            f"{prefix}/GetActions",
            request_serializer=services_pb2.Empty.SerializeToString,
            response_deserializer=services_pb2.Actions.FromString,
        )
        self.SendPolicyInstructions = channel.unary_unary(
            f"{prefix}/SendPolicyInstructions",
            request_serializer=services_pb2.PolicySetup.SerializeToString,
            response_deserializer=services_pb2.Empty.FromString,
        )
        instruction_message = getattr(services_pb2, "InstructionUpdate", None)
        if instruction_message is not None:
            self.UpdateInstruction = channel.unary_unary(
                f"{prefix}/UpdateInstruction",
                request_serializer=instruction_message.SerializeToString,
                response_deserializer=services_pb2.Empty.FromString,
            )
        self.ResetPolicy = channel.unary_unary(
                f"{prefix}/ResetPolicy",
                request_serializer=services_pb2.Empty.SerializeToString,
                response_deserializer=services_pb2.Empty.FromString,
            )
        self.Ready = channel.unary_unary(
            f"{prefix}/Ready",
            request_serializer=services_pb2.Empty.SerializeToString,
            response_deserializer=services_pb2.Empty.FromString,
        )


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
    return total


def _get_actions(stub, timeout: float = 0.0) -> list:
    """Fetch exactly one result for the observation just sent.

    PolicyServer consumes an observation on the first GetActions call.  If
    preprocessing/inference fails it returns an empty payload; polling again
    cannot recover that consumed observation and only creates a misleading
    30/60 second wait.
    """
    rpc_timeout = timeout if timeout > 0 else None
    actions_msg = stub.GetActions(services_pb2.Empty(), timeout=rpc_timeout)
    if len(actions_msg.data) == 0:
        return []

    timed_actions = _pickle.loads(actions_msg.data)
    result = []
    for ta in timed_actions:
        action_list = ta.action.detach().cpu().numpy().tolist()
        if isinstance(action_list, list) and action_list and isinstance(action_list[0], list):
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
        self._policy_configured = False
        self._configured_checkpoint = None

    def connect(self) -> bool:
        try:
            self._channel = grpc.insecure_channel(self.server_address)
            grpc.channel_ready_future(self._channel).result(timeout=10)
            self._stub = _PolicyRpcClient(self._channel)
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
            self._policy_configured = False
            self._configured_checkpoint = None
            print("[VLAClient] Disconnected")

    @property
    def connected(self) -> bool:
        return self._stub is not None

    @property
    def policy_ready(self) -> bool:
        """Whether this client successfully configured a policy on this connection."""
        return self.connected and self._policy_configured

    @property
    def configured_checkpoint(self):
        """Exact path sent in the last successful setup RPC on this connection."""
        return self._configured_checkpoint

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_policy_ready(self, timeout: float = 3.0) -> bool:
        """Return False because this protocol has no policy-status RPC.

        GetActions is not a readiness probe: the server returns an empty payload
        both when no observation is queued and when inference/setup failed.  A
        false positive here can silently keep an incompatible model loaded, so
        callers must send explicit policy instructions instead.
        """
        del timeout
        if not self.connected:
            return False
        print("[VLAClient] Policy readiness cannot be probed by this protocol; setup required")
        return False

    def setup_policy(self) -> bool:
        if not self.connected:
            print("[VLAClient] Not connected")
            return False
        try:
            from checkpoint_guard import format_checkpoint_report, inspect_checkpoint_files, resolve_checkpoint_path
            resolved = resolve_checkpoint_path(self.cfg.get("pretrained_name_or_path", ""))
            file_info = inspect_checkpoint_files(resolved)
            print("[VLAClient] " + format_checkpoint_report(file_info).replace("\n", "\n[VLAClient] "))
            self.cfg["pretrained_name_or_path"] = resolved
            remote_cfg = RemotePolicyConfig(
                policy_type=self.cfg.get("policy_type", "pi0"),
                pretrained_name_or_path=resolved,
                lerobot_features=self.cfg.get("observation_features", {}),
                actions_per_chunk=self.cfg.get("actions_per_chunk", 50),
                device=self.cfg.get("device", "cpu"),
            )
            cfg_bytes = _pickle.dumps(remote_cfg)
            self._stub.SendPolicyInstructions(services_pb2.PolicySetup(data=cfg_bytes))
            self._policy_configured = True
            self._configured_checkpoint = resolved
            print(f"[VLAClient] Policy setup sent resolved={resolved} load_mode={file_info['load_mode']}")
            return True
        except Exception as e:
            self._policy_configured = False
            self._configured_checkpoint = None
            print(f"[VLAClient] setup_policy error: {e}")
            return False

    def update_instruction(self, instruction: str) -> bool:
        if not self.connected:
            return False
        if not hasattr(self._stub, "UpdateInstruction"):
            return False
        try:
            instr_bytes = _pickle.dumps(instruction)
            self._stub.UpdateInstruction(services_pb2.InstructionUpdate(data=instr_bytes))
            print(f"[VLAClient] Instruction updated: {instruction}")
            return True
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNIMPLEMENTED:
                print("[VLAClient] UpdateInstruction not supported on this PolicyServer, "
                      "instruction will be sent with each observation")
                return False
            print(f"[VLAClient] update_instruction error: {e}")
            return False
        except Exception as e:
            print(f"[VLAClient] update_instruction error: {e}")
            return False

    def reset(self) -> bool:
        if not self.connected:
            return False
        if not hasattr(self._stub, "ResetPolicy"):
            return True
        try:
            self._stub.ResetPolicy(services_pb2.Empty())
            print("[VLAClient] Policy reset")
            return True
        except Exception as e:
            print(f"[VLAClient] reset error: {e}")
            return False

    def _remap_observation(self, obs_dict: dict) -> dict:
        """观测 key 重映射。禁止用 observation.state 数组替换 23 个平键。

        PolicyServer.build_dataset_frame() 按 observation.state.names 从原始
        dict 读取 right_arm_joint1 等平键，因此必须保留这些 key。
        """
        mapping = self.cfg.get("observation_mapping") or {}
        if not mapping:
            return obs_dict
        print("[VLAClient] WARNING: observation_mapping is set; "
              "keeping original flat state keys for PolicyServer")
        result = dict(obs_dict)
        for target_key, source_key in mapping.items():
            if source_key == "!state":
                print("[VLAClient] skipping !state mapping; "
                      "do not replace flat joints with observation.state")
                continue
            val = obs_dict.get(source_key)
            if val is not None:
                result[target_key] = val
        return result

    def _pack_observation_for_policy(self, obs_dict: dict, instruction: str) -> dict:
        """Keep the raw flat keys consumed by PolicyServer.build_dataset_frame.

        Do NOT replace the 23 joint keys with a single observation.state array.
        PolicyServer.build_dataset_frame() looks up right_arm_joint1 etc. on the
        original observation dict using observation.state.names.  Images are
        likewise read from flat camera keys (head_left, left_arm, ...); adding
        dotted aliases duplicates every image in the pickle payload.
        """
        packed = self._encode_images_for_transport(obs_dict)
        mapping = self.cfg.get("observation_mapping") or {}
        if mapping:
            print("[VLAClient] WARNING: observation_mapping is set; "
                  "flat state keys must still be present for PolicyServer")
        packed["task"] = instruction
        return packed

    def _encode_images_for_transport(self, obs_dict: dict) -> dict:
        """Losslessly PNG-compress camera arrays for the gRPC transport.

        The PolicyServer decodes these values back to the original uint8 RGB
        arrays before running its unchanged training-time image preprocessing.
        """
        transport_cfg = self.cfg.get("observation_transport") or {}
        encoding = str(transport_cfg.get("image_encoding", "raw")).lower()
        if encoding == "raw":
            return dict(obs_dict)
        if encoding != "png":
            raise ValueError(f"Unsupported observation image encoding: {encoding}")

        from PIL import Image

        compress_level = int(transport_cfg.get("png_compress_level", 1))
        if not 0 <= compress_level <= 9:
            raise ValueError("png_compress_level must be between 0 and 9")

        packed = dict(obs_dict)
        features = self.cfg.get("observation_features") or {}
        for feature_key, feature in features.items():
            if feature.get("dtype") != "image":
                continue
            camera_key = feature_key.rsplit(".", 1)[-1]
            if camera_key not in packed:
                continue

            image = np.asarray(packed[camera_key])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(
                    f"PNG transport expects {camera_key} as HWC uint8 RGB, "
                    f"got shape={image.shape} dtype={image.dtype}"
                )
            if not getattr(self, "_logged_png_input", False):
                print(
                    f"[VLAClient] PNG input {camera_key}: shape={tuple(image.shape)} "
                    f"dtype={image.dtype} min={int(image.min())} max={int(image.max())} "
                    f"RGB uint8 0-255 (no client /255)"
                )
            buffer = BytesIO()
            Image.fromarray(np.ascontiguousarray(image), mode="RGB").save(
                buffer,
                format="PNG",
                compress_level=compress_level,
            )
            packed[camera_key] = {
                "__lerobot_image_encoding__": "png",
                "data": buffer.getvalue(),
            }
        self._logged_png_input = True
        return packed

    def _validate_packed_observation(self, obs_dict: dict) -> bool:
        """Last check before gRPC send: configured images + flat joint keys."""
        features = self.cfg.get("observation_features") or {}
        required = [
            key.rsplit(".", 1)[-1]
            for key, feature in features.items()
            if (feature or {}).get("dtype") == "image"
        ]
        state_names = (
            (features.get("observation.state") or {}).get("names")
            or (self.cfg.get("robot") or {}).get("state_feature_names")
            or []
        )
        required.extend(list(state_names))
        missing = [k for k in required if k not in obs_dict]
        if missing:
            print(f"[VLAClient] packed observation missing={missing}, "
                  f"obs.keys()={list(obs_dict.keys())}")
            return False
        if "observation.state" in obs_dict and not all(n in obs_dict for n in state_names):
            print("[VLAClient] observation.state present but flat keys missing; refusing send")
            return False
        return True

    def infer(self, obs_dict: dict, instruction: str, timestep: int) -> Optional[List[dict]]:
        """Run inference and return list of action dicts.
        Each action: {"timestep": int, "action": List[float]}
        """
        if not self.connected:
            print("[VLAClient] Not connected")
            return None
        try:
            obs_dict = self._remap_observation(obs_dict)
            obs_dict = self._pack_observation_for_policy(obs_dict, instruction)
            if not self._validate_packed_observation(obs_dict):
                print("[VLAClient] refusing to send invalid observation to PolicyServer")
                return None
            timed_obs = TimedObservation(
                timestamp=time.time(),
                timestep=timestep,
                observation=obs_dict,
                must_go=True,
            )
            send_start = time.perf_counter()
            payload_size = _send_observations(self._stub, timed_obs)
            send_ms = (time.perf_counter() - send_start) * 1000
            print(
                f"[VLAClient] Sent observation {payload_size / 1024 / 1024:.2f} MiB "
                f"in {send_ms:.0f}ms"
            )
            timeout = float(self.cfg.get("action_response_timeout", 30.0))
            t0 = time.time()
            actions = _get_actions(self._stub, timeout=timeout)
            print(f"[VLAClient] Received {len(actions) if actions else 0} actions "
                  f"in {(time.time() - t0) * 1000:.0f}ms "
                  f"action_dim={len(actions[0]['action']) if actions and 'action' in actions[0] else 0} "
                  f"(obs keys={list(obs_dict.keys())})")
            return actions
        except Exception as e:
            print(f"[VLAClient] infer error: {e}")
            return None
