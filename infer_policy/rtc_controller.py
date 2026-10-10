"""
RTC (Real-Time Chunking) Controller — 异步推理 + 动作 buffer。

流程:
  1. 首次推理放满 buffer[50 帧]
  2. 主线程逐帧下发动作 (30Hz)
  3. 当 buffer 剩余 ≤ pre_infer_threshold 时，后台线程异步触发新推理
  4. 等待新 RTC 结果时，最多继续执行 max_steps_per_observation 帧旧动作
  5. 达到上限后保持最后目标，不再消费旧 buffer
  6. 将旧 chunk 剩余 prefix 随新 observation 发给远端 Pi0.5 RTC processor
  7. 新 chunk 到达后按实际执行延迟替换 buffer
"""

import threading
import time
import uuid
from typing import Optional

import numpy as np


class RTCController:
    """Robot-side queue; chunk continuity is generated inside Pi0.5 flow matching."""

    def __init__(self, config: dict):
        self.enabled = bool(config.get("enabled", False))
        self._pre_infer_threshold = int(config.get("pre_infer_threshold", 15))
        self._max_steps_per_observation = int(config.get("max_steps_per_observation", 0))
        if self._max_steps_per_observation < 0:
            raise ValueError("RTC max_steps_per_observation must be >= 0")

        self._action_buffer: list = []
        self._exec_ptr: int = 0
        self._lock = threading.Lock()
        self._new_chunk: Optional[list] = None
        self._new_chunk_ready = threading.Event()
        self._infer_in_progress = False
        self._control_step = 0
        self._trajectory_step = 0
        self._last_install_trajectory_step = 0
        self._continuation_start_trajectory_step: Optional[int] = None
        self._last_action: Optional[np.ndarray] = None
        self._new_chunk_meta: dict = {}
        self._session_id = uuid.uuid4().hex
        self._session_active = True
        self._infer_seq = 0

    def reset(self):
        """清空 buffer，每轮新推理开始前调用。"""
        with self._lock:
            self._action_buffer = []
            self._exec_ptr = 0
            self._new_chunk = None
            self._new_chunk_ready.clear()
            self._infer_in_progress = False
            self._control_step = 0
            self._trajectory_step = 0
            self._last_install_trajectory_step = 0
            self._continuation_start_trajectory_step = None
            self._last_action = None
            self._new_chunk_meta = {}
            self._session_id = uuid.uuid4().hex
            self._session_active = True
            self._infer_seq = 0

    def add_chunk(self, actions: list, skip_frames: int = 0) -> dict:
        """Replace the future queue, dropping frames executed since observation capture."""
        original_len = len(actions)
        actions = _normalize_actions(actions)
        skip_frames = max(0, min(int(skip_frames), len(actions)))
        actions = actions[skip_frames:]
        report = {
            "mode": "replace_rtc",
            "skipped": skip_frames,
            "start_index": skip_frames,
            "frames_elapsed": skip_frames,
            "original_len": original_len,
            "kept": len(actions),
        }

        with self._lock:
            self._action_buffer = actions
            self._exec_ptr = 0
            self._last_install_trajectory_step = self._trajectory_step
            # A fresh RTC result ends the old-action continuation window.
            self._continuation_start_trajectory_step = None
            return report

    def get_next_action(self) -> Optional[np.ndarray]:
        """取下一帧动作，buffer 空返回 None。"""
        with self._lock:
            if self._observation_budget_exhausted():
                return None
            if self._exec_ptr >= len(self._action_buffer):
                return None
            action = self._action_buffer[self._exec_ptr]
            self._exec_ptr += 1
            self._control_step += 1
            self._trajectory_step += 1
            self._last_action = action.copy()
            return action.copy()

    def record_hold_frame(self) -> None:
        with self._lock:
            self._control_step += 1

    def last_action(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._last_action is None else self._last_action.copy()

    @property
    def control_step(self) -> int:
        with self._lock:
            return self._control_step

    @property
    def trajectory_step(self) -> int:
        with self._lock:
            return self._trajectory_step

    def frames_since_last_chunk_install(self) -> int:
        with self._lock:
            return max(0, self._trajectory_step - self._last_install_trajectory_step)

    def buffer_remaining(self) -> int:
        """返回 buffer 中剩余未执行的帧数。"""
        with self._lock:
            return max(0, len(self._action_buffer) - self._exec_ptr)

    def should_trigger_inference(self) -> bool:
        """是否应该触发下一轮异步推理。"""
        with self._lock:
            remaining = max(0, len(self._action_buffer) - self._exec_ptr)
            return (
                remaining <= self._pre_infer_threshold
                and not self._infer_in_progress
                and not self._new_chunk_ready.is_set()
            )

    def _observation_budget_exhausted(self) -> bool:
        """Whether the in-flight RTC has consumed its old-action allowance.

        Caller holds ``_lock``. Hold frames increment only ``control_step``, so
        they never consume this trajectory-frame budget.
        """
        return (
            self._max_steps_per_observation > 0
            and self._continuation_start_trajectory_step is not None
            and self._trajectory_step - self._continuation_start_trajectory_step
            >= self._max_steps_per_observation
        )

    @property
    def infer_in_progress(self) -> bool:
        with self._lock:
            return self._infer_in_progress

    @property
    def new_chunk_ready(self) -> bool:
        return self._new_chunk_ready.is_set()

    @property
    def session_id(self) -> str:
        with self._lock:
            return self._session_id

    def is_session_active(self, session_id: Optional[str] = None) -> bool:
        with self._lock:
            return self._session_active and (
                session_id is None or session_id == self._session_id
            )

    def close_session(self) -> None:
        with self._lock:
            self._session_active = False
            self._infer_in_progress = False
            self._continuation_start_trajectory_step = None

    def mark_inference_started(self):
        """Latch one asynchronous inference and return its session metadata."""
        with self._lock:
            self._infer_in_progress = True
            self._infer_seq += 1
            # From this trigger onward, allow at most N additional old actions
            # while the new RTC request is in flight. Observation capture occurs
            # immediately afterward under the robot lock, so using the trigger
            # boundary is also safe if the capture thread is briefly delayed.
            self._continuation_start_trajectory_step = self._trajectory_step
            return {
                "session_id": self._session_id,
                "infer_seq": self._infer_seq,
                "infer_start_step": self._control_step,
                "infer_start_time": time.monotonic(),
            }

    def mark_observation_captured(self, session_id: str) -> Optional[dict]:
        with self._lock:
            if not self._session_active or session_id != self._session_id:
                return None
            return {
                "obs_capture_step": self._control_step,
                "obs_capture_control_step": self._control_step,
                "obs_capture_trajectory_step": self._trajectory_step,
                "obs_capture_time": time.monotonic(),
                "previous_actions": [
                    action.copy() for action in self._action_buffer[self._exec_ptr:]
                ],
                "previous_action_start_index": self._exec_ptr,
            }

    def mark_inference_done(self, new_actions: list, meta: Optional[dict] = None,
                            session_id: Optional[str] = None):
        """异步推理线程完成时调用。"""
        actions = _normalize_actions(new_actions)
        with self._lock:
            if not self._session_active or (
                session_id is not None and session_id != self._session_id
            ):
                return False
            self._new_chunk = actions
            self._new_chunk_meta = dict(meta or {})
            self._infer_in_progress = False
            self._new_chunk_ready.set()
            return True

    def wait_for_new_chunk(self, timeout: float) -> bool:
        return self._new_chunk_ready.wait(max(0.0, float(timeout)))

    def consume_new_chunk(self):
        """主线程消费异步推理结果，融合入 buffer。"""
        with self._lock:
            if not self._new_chunk_ready.is_set():
                return None
            self._new_chunk_ready.clear()
            chunk = self._new_chunk
            self._new_chunk = None
            meta = self._new_chunk_meta
            self._new_chunk_meta = {}

        if chunk:
            observation_step = int(meta.get("obs_capture_trajectory_step", self.trajectory_step))
            local_delay_frames = max(0, self.trajectory_step - observation_step)
            report = self.add_chunk(chunk, skip_frames=local_delay_frames)
            report["server_inference_delay_frames"] = int(
                meta.get("inference_delay_frames", 0) or 0
            )
            report["prefix_frames_used"] = int(meta.get("prefix_frames_used", 0) or 0)
            return report, meta
        with self._lock:
            # The caller will trigger a fresh inference for an empty result.
            # Give that retry its own old-action continuation window.
            self._continuation_start_trajectory_step = None
        return ({
            "mode": "empty",
            "skipped": 0,
            "start_index": 0,
            "frames_elapsed": 0,
            "original_len": 0,
            "kept": 0,
        }, meta)


def _normalize_actions(actions: list) -> list:
    """Convert a chunk to same-sized, one-dimensional float32 actions."""
    normalized = [np.asarray(action, dtype=np.float32).reshape(-1) for action in actions]
    if not normalized:
        return []
    dim = normalized[0].size
    for index, action in enumerate(normalized[1:], start=1):
        if action.size != dim:
            raise ValueError(
                "action dim mismatch inside RTC chunk: "
                f"frame 0 has {dim}, frame {index} has {action.size}"
            )
    return normalized
