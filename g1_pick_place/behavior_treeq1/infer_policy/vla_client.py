"""G1 adapter around LeRobot's asynchronous PolicyServer client protocol."""

import statistics
import threading
import time
import uuid
from io import BytesIO
from typing import List, Dict, Any, Optional

import numpy as np
import grpc
from lerobot.async_inference.helpers import (
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
)
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import (
    bytes_to_python_object,
    grpc_channel_options,
    python_object_to_bytes,
    send_bytes_in_chunks,
)

# PolicyServer's protocol intentionally serializes these canonical LeRobot
# data objects with pickle.  Only trusted client/server deployments may use it.


# ------------------------------------------------------------------
# gRPC helper: send bytes in streaming chunks
# ------------------------------------------------------------------

def _validate_rtc_previous_actions(previous_actions: Any, expected_dim: int) -> None:
    """Reject RTC guidance that does not match the policy action space."""
    if previous_actions is None:
        return
    matrix = np.asarray(previous_actions, dtype=np.float32)
    if matrix.size == 0:
        return
    if matrix.ndim != 2 or matrix.shape[1] != expected_dim:
        raise ValueError(
            "RTC previous_actions shape mismatch: "
            f"got {matrix.shape}, expected [T, {expected_dim}]"
        )

class _TimedRequestIterator:
    """Record when gRPC starts and finishes consuming a request iterator.

    Iterator consumption only means that Python handed the bytes to gRPC core;
    it is not proof that the peer has received them.  The remaining time until
    the unary ACK is tracked separately.
    """

    def __init__(self, iterator):
        self._iterator = iter(iterator)
        self.first_pull_at = None
        self.exhausted_at = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.first_pull_at is None:
            self.first_pull_at = time.perf_counter()
        try:
            return next(self._iterator)
        except StopIteration:
            self.exhausted_at = time.perf_counter()
            raise


def _send_observations(stub, timed_obs: TimedObservation, trace_id: str) -> dict:
    serialize_start = time.perf_counter()
    obs_bytes = python_object_to_bytes(timed_obs)
    serialize_end = time.perf_counter()
    send_start = time.perf_counter()
    request_iterator = _TimedRequestIterator(
        send_bytes_in_chunks(obs_bytes, services_pb2.Observation)
    )
    stub.SendObservations(
        request_iterator,
        metadata=(("x-lerobot-trace-id", trace_id),),
    )
    send_end = time.perf_counter()
    first_pull = request_iterator.first_pull_at or send_start
    exhausted = request_iterator.exhausted_at or send_end
    return {
        "payload_bytes": len(obs_bytes),
        "serialize_ms": (serialize_end - serialize_start) * 1000,
        "send_rpc_ms": (send_end - send_start) * 1000,
        "client_dispatch_ms": max(0.0, (first_pull - send_start) * 1000),
        "request_iterator_ms": max(0.0, (exhausted - first_pull) * 1000),
        "send_ack_wait_ms": max(0.0, (send_end - exhausted) * 1000),
    }


def _metadata_dict(metadata) -> dict:
    result = {}
    for item in metadata or ():
        key = getattr(item, "key", None)
        value = getattr(item, "value", None)
        if key is None and isinstance(item, tuple) and len(item) == 2:
            key, value = item
        if key is not None:
            result[str(key)] = value
    return result


def _metadata_float(metadata: dict, key: str) -> Optional[float]:
    value = metadata.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _server_timing_from_metadata(metadata: dict) -> dict:
    keys = {
        "server_receive_ms": "x-lerobot-server-receive-ms",
        "server_queue_ms": "x-lerobot-server-queue-ms",
        "server_preprocess_ms": "x-lerobot-server-preprocess-ms",
        "server_model_infer_ms": "x-lerobot-server-model-ms",
        "server_postprocess_ms": "x-lerobot-server-postprocess-ms",
        "server_serialize_ms": "x-lerobot-server-serialize-ms",
        "server_get_actions_ms": "x-lerobot-server-get-actions-ms",
    }
    result = {}
    for name, metadata_key in keys.items():
        value = _metadata_float(metadata, metadata_key)
        if value is not None:
            result[name] = value
    trace_id = metadata.get("x-lerobot-trace-id")
    if trace_id is not None:
        result["server_trace_id"] = str(trace_id)
    return result


def _get_actions(stub, timeout: float = 0.0, trace_id: str = "") -> tuple:
    """Fetch exactly one result for the observation just sent.

    PolicyServer consumes an observation on the first GetActions call.  If
    preprocessing/inference fails it returns an empty payload; polling again
    cannot recover that consumed observation and only creates a misleading
    30/60 second wait.
    """
    rpc_timeout = timeout if timeout > 0 else None
    request_start = time.perf_counter()
    metadata = (("x-lerobot-trace-id", trace_id),) if trace_id else None
    get_actions = stub.GetActions
    if hasattr(get_actions, "with_call"):
        actions_msg, call = get_actions.with_call(
            services_pb2.Empty(), timeout=rpc_timeout, metadata=metadata
        )
        trailing_metadata = _metadata_dict(call.trailing_metadata())
    else:
        # Test doubles and older generated stubs may not expose with_call.
        actions_msg = get_actions(services_pb2.Empty(), timeout=rpc_timeout)
        trailing_metadata = {}
    response_received = time.perf_counter()
    if len(actions_msg.data) == 0:
        return [], {
            "get_actions_rpc_ms": (response_received - request_start) * 1000,
            "response_decode_ms": 0.0,
            "server_timing": _server_timing_from_metadata(trailing_metadata),
        }

    decode_start = time.perf_counter()
    payload = bytes_to_python_object(actions_msg.data)
    response_meta = {}
    if isinstance(payload, dict) and "actions" in payload:
        timed_actions = payload["actions"]
        response_meta = {
            key: payload.get(key)
            for key in (
                "trace_id",
                "observation_timestep",
                "inference_delay_frames",
                "prefix_frames_used",
                "expired_prefix_frames",
                "previous_action_start_index",
            )
        }
    else:
        timed_actions = payload
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
    decode_end = time.perf_counter()
    return result, {
        "get_actions_rpc_ms": (response_received - request_start) * 1000,
        "response_decode_ms": (decode_end - decode_start) * 1000,
        "server_timing": _server_timing_from_metadata(trailing_metadata),
        "response_meta": response_meta,
    }


def summarize_infer_rpc_samples(
    samples: list,
    rtt_ms: float = 0.0,
    spike_factor: float = 3.0,
) -> dict:
    """Maintain the legacy aggregate estimate used by timing reports.

    ``send_ms`` is the synchronous SendObservations RPC through its server ACK,
    not pure socket upload. Detailed per-request logs use the unambiguous
    ``send_rpc_ms``/``send_ack_wait_ms`` names. GetActions is likewise a whole
    RPC; server timing metadata, when installed, provides the authoritative
    queue/model split.
    """
    waits = [float((sample or {}).get("wait_ms") or 0.0) for sample in samples]
    sends = [float((sample or {}).get("send_ms") or 0.0) for sample in samples]
    totals = [float((sample or {}).get("total_ms") or 0.0) for sample in samples]
    count = len(samples)
    send_ms = float(sum(sends))
    wait_ms = float(sum(waits))
    rpc_ms = float(sum(totals)) if any(totals) else send_ms + wait_ms
    rtt = max(0.0, float(rtt_ms or 0.0))
    detail_fields = (
        "client_wait_ms",
        "pack_ms",
        "serialize_ms",
        "send_rpc_ms",
        "client_dispatch_ms",
        "request_iterator_ms",
        "send_ack_wait_ms",
        "get_actions_rpc_ms",
        "response_decode_ms",
    )
    detail_totals = {
        key: float(sum(float((sample or {}).get(key) or 0.0) for sample in samples))
        for key in detail_fields
    }
    server_fields = (
        "server_receive_ms",
        "server_queue_ms",
        "server_preprocess_ms",
        "server_model_infer_ms",
        "server_postprocess_ms",
        "server_serialize_ms",
        "server_get_actions_ms",
    )
    detail_totals.update({
        key: float(sum(
            float(((sample or {}).get("server_timing") or {}).get(key) or 0.0)
            for sample in samples
        ))
        for key in server_fields
    })
    detail_totals["payload_bytes"] = int(sum(
        int((sample or {}).get("payload_bytes") or 0) for sample in samples
    ))

    if not waits:
        return {
            "count": 0,
            "send_ms": 0.0,
            "wait_ms": 0.0,
            "rpc_ms": 0.0,
            "median_wait_ms": 0.0,
            "compute_ms": 0.0,
            "network_ms": 0.0,
            "wait_spike_ms": 0.0,
            "rtt_ms": rtt,
            "rpc_detail": detail_totals,
        }
    min_wait = min(waits)
    healthy = [wait for wait in waits if wait <= max(1000.0, min_wait * 5.0)] or waits
    median_wait = float(statistics.median(healthy))
    cap = median_wait * float(spike_factor)
    compute_ms = 0.0
    wait_spike_ms = 0.0
    typical = max(0.0, median_wait - rtt)
    for wait in waits:
        if wait > cap:
            compute_ms += typical
            wait_spike_ms += wait - max(median_wait, rtt)
        else:
            compute_ms += max(0.0, wait - rtt)
    return {
        "count": count,
        "send_ms": send_ms,
        "wait_ms": wait_ms,
        "rpc_ms": rpc_ms,
        "median_wait_ms": median_wait,
        "compute_ms": compute_ms,
        "network_ms": send_ms + count * rtt + wait_spike_ms,
        "wait_spike_ms": wait_spike_ms,
        "rtt_ms": rtt,
        "rpc_detail": detail_totals,
    }


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
        self._baseline_rtt_ms = 0.0
        self._channel_generation = 0
        self.last_infer_timing: Optional[Dict[str, Any]] = None
        self._timing_local = threading.local()
        # A stale RTC worker may outlive its session.  gRPC channels are thread
        # safe, but this PolicyServer protocol uses one shared observation queue,
        # so overlapping SendObservations/GetActions pairs can consume each
        # other's work.  Serialize complete inference transactions per client.
        self._infer_lock = threading.Lock()

    @staticmethod
    def _trace(trace_id: str, event: str, **fields) -> None:
        suffix = " ".join(f"{key}={value}" for key, value in fields.items())
        print(
            f"[VLAClient] TRACE trace_id={trace_id} event={event} "
            f"perf_s={time.perf_counter():.6f}"
            + (f" {suffix}" if suffix else "")
        )

    def connect(self) -> bool:
        try:
            self._channel = grpc.insecure_channel(
                self.server_address,
                options=grpc_channel_options(),
            )
            grpc.channel_ready_future(self._channel).result(timeout=10)
            self._stub = services_pb2_grpc.AsyncInferenceStub(self._channel)
            ready_started = time.perf_counter()
            self._stub.Ready(services_pb2.Empty())
            self._baseline_rtt_ms = (time.perf_counter() - ready_started) * 1000
            self._channel_generation += 1
            print(
                f"[VLAClient] Connected to PolicyServer at {self.server_address} "
                f"channel_generation={self._channel_generation} "
                f"ready_rtt_ms={self._baseline_rtt_ms:.0f}; channel/stub will be reused"
            )
            return True
        except Exception as e:
            detail = str(e).strip() or repr(e)
            print(
                f"[VLAClient] Failed to connect to {self.server_address}: "
                f"{type(e).__name__}: {detail}"
            )
            return False

    def disconnect(self):
        if self._channel:
            self._channel.close()
            self._channel = None
            self._stub = None
            self._policy_configured = False
            self._configured_checkpoint = None
            self._baseline_rtt_ms = 0.0
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
            checkpoint_source = str(self.cfg.get("checkpoint_source", "auto")).strip().lower()
            if checkpoint_source == "remote":
                load_mode = "remote"
            else:
                file_info = inspect_checkpoint_files(resolved)
                load_mode = file_info["load_mode"]
                print("[VLAClient] " + format_checkpoint_report(file_info).replace("\n", "\n[VLAClient] "))
            self.cfg["pretrained_name_or_path"] = resolved
            remote_rtc_config = dict(self.cfg.get("rtc") or {})
            remote_rtc_config.pop("max_steps_per_observation", None)
            remote_cfg = RemotePolicyConfig(
                policy_type=self.cfg.get("policy_type", "pi0"),
                pretrained_name_or_path=resolved,
                lerobot_features=self.cfg.get("observation_features", {}),
                actions_per_chunk=self.cfg.get("actions_per_chunk", 50),
                device=self.cfg.get("device", "cpu"),
                rtc_config=remote_rtc_config,
            )
            cfg_bytes = python_object_to_bytes(remote_cfg)
            print(f"[VLAClient] Loading policy on remote PolicyServer: {resolved}")
            self._stub.SendPolicyInstructions(services_pb2.PolicySetup(data=cfg_bytes))
            self._policy_configured = True
            self._configured_checkpoint = resolved
            print(f"[VLAClient] Remote policy ready: {resolved} load_mode={load_mode}")
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
            instr_bytes = python_object_to_bytes(instruction)
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
        encoding = str(transport_cfg.get("image_encoding", "png")).lower()
        if encoding == "raw":
            return dict(obs_dict)
        if encoding != "png":
            raise ValueError(f"Unsupported observation image encoding: {encoding}")

        from PIL import Image

        compress_level = int(transport_cfg.get("png_compress_level", 3))
        if not 0 <= compress_level <= 9:
            raise ValueError("png_compress_level must be between 0 and 9")

        features = self.cfg.get("observation_features") or {}
        configured_cameras = {
            feature_key.rsplit(".", 1)[-1]
            for feature_key, feature in features.items()
            if (feature or {}).get("dtype") == "image"
        }
        # Never send an unconfigured camera. Apart from violating the training
        # schema, an ndarray left here bypasses PNG encoding and greatly inflates
        # the gRPC payload.
        known_cameras = {"head_left", "head_right", "left_arm", "right_arm"}
        packed = {
            key: value
            for key, value in obs_dict.items()
            if key not in known_cameras or key in configured_cameras
        }
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

    def probe_rtt_ms(self, timeout: float = 2.0) -> float:
        """Return the cached handshake RTT without invoking stateful Ready().

        LeRobot PolicyServer.Ready resets its observation queue.  Calling it as
        a per-run ping can race with an old RTC request, so RTT is measured only
        during connect(), when Ready is required by the protocol.
        """
        del timeout
        return self._baseline_rtt_ms if self.connected else 0.0

    def take_last_infer_timing(self) -> Dict[str, Any]:
        local_timing = getattr(self._timing_local, "last_infer_timing", None)
        timing = dict(local_timing or {})
        if not timing:
            timing = dict(self.last_infer_timing or {})
        self._timing_local.last_infer_timing = None
        if self.last_infer_timing is local_timing:
            self.last_infer_timing = None
        return timing

    def infer(
        self,
        obs_dict: dict,
        instruction: str,
        timestep: int,
        rtc_context: Optional[dict] = None,
    ) -> Optional[List[dict]]:
        """Run inference and return list of action dicts.
        Each action: {"timestep": int, "action": List[float]}
        """
        rtc_context = dict(rtc_context or {})
        if rtc_context:
            expected_action_dim = int(
                ((self.cfg.get("robot") or {}).get("vla_action_dim", 0)) or 0
            )
            if expected_action_dim > 0:
                _validate_rtc_previous_actions(
                    rtc_context.get("previous_actions"), expected_action_dim
                )
        trace_id = str(rtc_context.get("trace_id") or f"{timestep}-{uuid.uuid4().hex[:10]}")
        obs_ready_at = time.perf_counter()
        self._trace(trace_id, "CLIENT_OBS_READY", timestep=timestep)
        self._infer_lock.acquire()
        lock_acquired_at = time.perf_counter()
        timing = {
            "trace_id": trace_id,
            "client_wait_ms": (lock_acquired_at - obs_ready_at) * 1000,
            "pack_ms": 0.0,
            "serialize_ms": 0.0,
            "send_rpc_ms": 0.0,
            "client_dispatch_ms": 0.0,
            "request_iterator_ms": 0.0,
            "send_ack_wait_ms": 0.0,
            # Backward-compatible names consumed by aggregate reports.
            "send_ms": 0.0,
            "wait_ms": 0.0,
            "get_actions_rpc_ms": 0.0,
            "response_decode_ms": 0.0,
            "rpc_total_ms": 0.0,
            "total_ms": 0.0,
            "payload_bytes": 0,
            "server_timing": {},
            "ok": False,
        }
        self.last_infer_timing = timing
        self._timing_local.last_infer_timing = timing
        if timing["client_wait_ms"] >= 1.0:
            self._trace(
                trace_id, "CLIENT_INFER_LOCK_ACQUIRED",
                client_wait_ms=f"{timing['client_wait_ms']:.3f}",
            )
        infer_start = time.perf_counter()
        rpc_start = None
        try:
            if not self.connected:
                print("[VLAClient] Not connected")
                return None
            self._trace(trace_id, "CLIENT_PACK_START")
            t_pack = time.perf_counter()
            obs_dict = self._remap_observation(obs_dict)
            obs_dict = self._pack_observation_for_policy(obs_dict, instruction)
            pack_ms = (time.perf_counter() - t_pack) * 1000
            timing["pack_ms"] = pack_ms
            self._trace(trace_id, "CLIENT_PACK_END", pack_ms=f"{pack_ms:.3f}")
            if not self._validate_packed_observation(obs_dict):
                print("[VLAClient] refusing to send invalid observation to PolicyServer")
                return None
            timed_obs = TimedObservation(
                timestamp=time.time(),
                timestep=timestep,
                observation=obs_dict,
                must_go=True,
                trace_id=trace_id,
                previous_actions=rtc_context.get("previous_actions"),
                previous_action_start_index=int(
                    rtc_context.get("previous_action_start_index", 0) or 0
                ),
                control_step=int(rtc_context.get("control_step", 0) or 0),
                trajectory_step=int(rtc_context.get("trajectory_step", timestep) or 0),
                fps=float(rtc_context.get("fps", self.cfg.get("fps", 30)) or 30),
            )
            rpc_start = time.perf_counter()
            self._trace(trace_id, "CLIENT_RPC_START")
            self._trace(trace_id, "CLIENT_SEND_START")
            send_detail = _send_observations(self._stub, timed_obs, trace_id)
            timing.update(send_detail)
            timing["send_ms"] = timing["send_rpc_ms"]
            self._trace(
                trace_id, "CLIENT_SEND_END",
                send_rpc_ms=f"{timing['send_rpc_ms']:.3f}",
                serialize_ms=f"{timing['serialize_ms']:.3f}",
                client_dispatch_ms=f"{timing['client_dispatch_ms']:.3f}",
                request_iterator_ms=f"{timing['request_iterator_ms']:.3f}",
                send_ack_wait_ms=f"{timing['send_ack_wait_ms']:.3f}",
            )
            payload_size = int(timing["payload_bytes"])
            print(
                f"[VLAClient] Observation RPC ACK {payload_size / 1024 / 1024:.2f} MiB "
                f"in {timing['send_rpc_ms']:.0f}ms"
            )
            timeout = float(self.cfg.get("action_response_timeout", 30.0))
            actions, response_detail = _get_actions(
                self._stub, timeout=timeout, trace_id=trace_id
            )
            timing.update(response_detail)
            response_meta = dict(response_detail.get("response_meta") or {})
            timing["response_meta"] = response_meta
            if rtc_context:
                response_trace_id = str(response_meta.get("trace_id") or "")
                response_timestep = response_meta.get("observation_timestep")
                if response_trace_id != trace_id or response_timestep != timestep:
                    print(
                        "[VLAClient] stale RTC result discarded: "
                        f"request_trace={trace_id} response_trace={response_trace_id or '?'} "
                        f"request_timestep={timestep} response_timestep={response_timestep}"
                    )
                    return None
                timing["inference_delay_frames"] = int(
                    response_meta.get("inference_delay_frames", 0) or 0
                )
                timing["prefix_frames_used"] = int(
                    response_meta.get("prefix_frames_used", 0) or 0
                )
            timing["wait_ms"] = timing["get_actions_rpc_ms"]
            self._trace(
                trace_id, "CLIENT_RESPONSE_RECEIVED",
                get_actions_rpc_ms=f"{timing['get_actions_rpc_ms']:.3f}",
                response_decode_ms=f"{timing['response_decode_ms']:.3f}",
            )
            rpc_end = time.perf_counter()
            timing["rpc_total_ms"] = (rpc_end - rpc_start) * 1000
            timing["total_ms"] = (rpc_end - infer_start) * 1000
            timing["ok"] = bool(actions)
            self._trace(
                trace_id, "CLIENT_RPC_END",
                rpc_total_ms=f"{timing['rpc_total_ms']:.3f}",
                client_infer_total_ms=f"{timing['total_ms']:.3f}",
            )
            print(f"[VLAClient] Received {len(actions) if actions else 0} actions "
                  f"in {timing['get_actions_rpc_ms']:.0f}ms "
                  f"action_dim={len(actions[0]['action']) if actions and 'action' in actions[0] else 0} "
                  f"(obs keys={list(obs_dict.keys())})")
            return actions
        except Exception as e:
            now = time.perf_counter()
            timing["total_ms"] = (now - infer_start) * 1000
            if rpc_start is not None:
                timing["rpc_total_ms"] = (now - rpc_start) * 1000
            print(f"[VLAClient] infer error: {e}")
            return None
        finally:
            self._infer_lock.release()
