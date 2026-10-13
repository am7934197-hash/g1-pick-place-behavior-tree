"""Small gRPC client compatible with LeRobot's AsyncInference PolicyServer."""

import pickle
import sys
import time
import types
from typing import Optional

import grpc


def _varint(value: int) -> bytes:
    out = bytearray()
    while value >= 128:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _bytes_field(number: int, value: bytes) -> bytes:
    return _varint((number << 3) | 2) + _varint(len(value)) + value


def _read_varint(data: bytes, index: int):
    result, shift = 0, 0
    while index < len(data):
        byte = data[index]
        index += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, index
        shift += 7
    raise ValueError("truncated protobuf varint")


def _field(data: bytes, wanted: int) -> bytes:
    index = 0
    while index < len(data):
        tag, index = _read_varint(data, index)
        number, wire = tag >> 3, tag & 7
        if wire == 0:
            _, index = _read_varint(data, index)
            continue
        if wire != 2:
            raise ValueError(f"unsupported protobuf wire type: {wire}")
        size, index = _read_varint(data, index)
        value = data[index:index + size]
        index += size
        if number == wanted:
            return value
    return b""


class Empty:
    def SerializeToString(self):
        return b""

    @classmethod
    def FromString(cls, _data):
        return cls()


class BytesMessage:
    def __init__(self, data=b""):
        self.data = data

    def SerializeToString(self):
        return _bytes_field(1, self.data)

    @classmethod
    def FromString(cls, data):
        return cls(_field(data, 1))


class ObservationMessage:
    def __init__(self, transfer_state=0, data=b""):
        self.transfer_state, self.data = transfer_state, data

    def SerializeToString(self):
        return _varint(8) + _varint(self.transfer_state) + _bytes_field(2, self.data)


class TimedObservation:
    def __init__(self, timestamp, timestep, observation, must_go=False):
        self.timestamp, self.timestep = timestamp, timestep
        self.observation, self.must_go = observation, must_go

    def get_timestamp(self): return self.timestamp
    def get_timestep(self): return self.timestep
    def get_observation(self): return self.observation


class TimedAction:
    def __init__(self, timestamp, timestep, action):
        self.timestamp, self.timestep, self.action = timestamp, timestep, action

    def get_timestamp(self): return self.timestamp
    def get_timestep(self): return self.timestep
    def get_action(self): return self.action


class RemotePolicyConfig:
    def __init__(self, policy_type, pretrained_name_or_path, lerobot_features,
                 actions_per_chunk, device="cpu", rename_map=None):
        self.policy_type = policy_type
        self.pretrained_name_or_path = pretrained_name_or_path
        self.lerobot_features = lerobot_features
        self.actions_per_chunk = actions_per_chunk
        self.device = device
        self.rename_map = rename_map or {}
        self.acp_enable = False
        self.acp_use_cfg = False
        self.acp_cfg_beta = 1.0


def _register_pickle_classes():
    for name, package in (("lerobot", True), ("lerobot.async_inference", True),
                          ("lerobot.async_inference.helpers", False)):
        if name not in sys.modules:
            module = types.ModuleType(name)
            if package:
                module.__path__ = []
            sys.modules[name] = module
    helpers = sys.modules["lerobot.async_inference.helpers"]
    for cls in (TimedObservation, TimedAction, RemotePolicyConfig):
        setattr(helpers, cls.__name__, cls)
        cls.__module__ = "lerobot.async_inference.helpers"
        cls.__qualname__ = cls.__name__


_register_pickle_classes()


class VLAClient:
    CHUNK_SIZE = 2 * 1024 * 1024

    def __init__(self, config: dict):
        self.config = config
        self.address = config["policy_server_address"]
        self.channel = None
        self.stub = None

    def connect(self) -> None:
        self.channel = grpc.insecure_channel(self.address)
        grpc.channel_ready_future(self.channel).result(
            timeout=float(self.config.get("connect_timeout", 10.0))
        )
        prefix = "/transport.AsyncInference"
        self.stub = types.SimpleNamespace(
            ready=self.channel.unary_unary(
                prefix + "/Ready", request_serializer=Empty.SerializeToString,
                response_deserializer=Empty.FromString),
            setup=self.channel.unary_unary(
                prefix + "/SendPolicyInstructions", request_serializer=BytesMessage.SerializeToString,
                response_deserializer=Empty.FromString),
            send=self.channel.stream_unary(
                prefix + "/SendObservations", request_serializer=ObservationMessage.SerializeToString,
                response_deserializer=Empty.FromString),
            actions=self.channel.unary_unary(
                prefix + "/GetActions", request_serializer=Empty.SerializeToString,
                response_deserializer=BytesMessage.FromString),
        )
        self.stub.ready(Empty())
        print(f"[vla] connected to {self.address}")

    def close(self) -> None:
        if self.channel is not None:
            self.channel.close()
        self.channel = self.stub = None

    def setup_policy(self) -> None:
        cfg = RemotePolicyConfig(
            policy_type=self.config.get("policy_type", "pi05"),
            pretrained_name_or_path=self.config["pretrained_name_or_path"],
            lerobot_features=self.config["observation_features"],
            actions_per_chunk=int(self.config.get("actions_per_chunk", 50)),
            device=self.config.get("device", "cuda"),
        )
        self.stub.setup(BytesMessage(pickle.dumps(cfg)))
        print(f"[vla] policy ready: {cfg.pretrained_name_or_path}")

    def infer(self, observation: dict, instruction: str, timestep: int):
        if self.stub is None:
            raise RuntimeError("VLA client is not connected")
        packed = dict(observation)
        packed["task"] = instruction
        timed = TimedObservation(time.time(), timestep, packed, must_go=True)
        payload = pickle.dumps(timed)

        def chunks():
            for offset in range(0, len(payload), self.CHUNK_SIZE):
                end = min(offset + self.CHUNK_SIZE, len(payload))
                state = 3 if end == len(payload) else (1 if offset == 0 else 2)
                yield ObservationMessage(state, payload[offset:end])

        started = time.perf_counter()
        self.stub.send(chunks())
        sent = time.perf_counter()
        response = self.stub.actions(
            Empty(), timeout=float(self.config.get("action_response_timeout", 30.0))
        )
        if not response.data:
            return [], {
                "payload_bytes": len(payload),
                "send_ms": (sent - started) * 1000,
                "wait_ms": (time.perf_counter() - sent) * 1000,
            }
        actions = []
        for item in pickle.loads(response.data):
            value = item.action.detach().cpu().numpy().tolist()
            if value and isinstance(value[0], list):
                value = value[0]
            actions.append({"timestep": item.timestep, "timestamp": item.timestamp, "action": value})
        timing = {
            "payload_bytes": len(payload),
            "send_ms": round((sent - started) * 1000, 2),
            "wait_ms": round((time.perf_counter() - sent) * 1000, 2),
        }
        return actions, timing
