"""
RTC (Real-Time Chunking) Controller — 异步推理 + 动作 buffer + 融合。

流程:
  1. 首次推理放满 buffer[50 帧]
  2. 主线程逐帧下发动作 (30Hz)
  3. 当 buffer 剩余 ≤ pre_infer_threshold 时，后台线程异步触发新推理
  4. 新 chunk 到达后，与旧 chunk 剩余部分加权/多项式融合
  5. 主线程感知不到推理延迟，机器人无停顿

用法:
    rtc = RTCController(config.get("rtc", {}))
    rtc.add_chunk(actions)          # 首次填充
    while True:
        action = rtc.get_next_action()
        if action is None:
            break
        # ... 下发 action ...
        if rtc.should_trigger_inference():
            trigger_async_infer()
        if rtc.new_chunk_ready:
            rtc.consume_new_chunk()
"""

import threading
from typing import Optional

import numpy as np


class RTCController:
    """异步动作 buffer 控制器，支持加权和多项式两种融合方式。"""

    def __init__(self, config: dict):
        self.enabled = bool(config.get("enabled", False))
        self._fusion_window = int(config.get("fusion_window", 10))
        self._fusion_method = str(config.get("fusion_method", "weighted"))
        self._pre_infer_threshold = int(config.get("pre_infer_threshold", 15))

        self._action_buffer: list = []     # list of np.ndarray (dim,)
        self._exec_ptr: int = 0
        self._lock = threading.Lock()
        self._new_chunk: Optional[list] = None
        self._new_chunk_meta: Optional[dict] = None
        self._new_chunk_ready = threading.Event()
        self._infer_in_progress = False
        self._infer_start_exec_ptr: int = 0  # exec_ptr 快照，用于计算推理延迟帧偏移
        self._infer_seq: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """清空 buffer，每轮新推理开始前调用。"""
        with self._lock:
            self._action_buffer = []
            self._exec_ptr = 0
            self._new_chunk = None
            self._new_chunk_meta = None
            self._new_chunk_ready.clear()
            self._infer_in_progress = False
            self._infer_start_exec_ptr = 0
            self._infer_seq = 0

    def add_chunk(self, actions: list) -> dict:
        """首次填充或融合新 chunk 到 buffer。

        根据推理延迟跳过新 chunk 中已过期的帧：
          frames_elapsed = exec_ptr - _infer_start_exec_ptr
          新 chunk 的前 frames_elapsed 帧对应推理期间已被旧 buffer 覆盖的时间段，丢弃。

        Returns a fusion report:
          mode: replace | fuse | drop_expired
          skipped: how many new-chunk frames were dropped as stale
          start_index: first kept new-chunk index (0 if none skipped)
          frames_elapsed: exec_ptr delta during inference
        """
        original_len = len(actions)
        actions = [np.asarray(a, dtype=np.float32) for a in actions]
        report = {
            "mode": "replace",
            "skipped": 0,
            "start_index": 0,
            "frames_elapsed": 0,
            "original_len": original_len,
            "kept": original_len,
        }
        with self._lock:
            if not self._action_buffer or self._exec_ptr >= len(self._action_buffer):
                self._action_buffer = actions
                self._exec_ptr = 0
                report["mode"] = "replace"
                return report

            remaining = self._action_buffer[self._exec_ptr:]
            frames_elapsed = self._exec_ptr - self._infer_start_exec_ptr
            report["frames_elapsed"] = int(frames_elapsed)
            if frames_elapsed > 0:
                if frames_elapsed >= len(actions):
                    report["mode"] = "drop_expired"
                    report["skipped"] = len(actions)
                    report["kept"] = 0
                    report["start_index"] = original_len
                    return report
                report["skipped"] = int(frames_elapsed)
                report["start_index"] = int(frames_elapsed)
                actions = actions[frames_elapsed:]

            if self._fusion_method == "polynomial":
                fused = self._fuse_quintic_hermite(remaining, actions)
            else:
                fused = self._fuse_weighted(remaining, actions)
            self._action_buffer = self._action_buffer[:self._exec_ptr] + fused
            report["mode"] = "fuse"
            report["kept"] = len(actions)
            return report

    def get_next_action(self) -> Optional[np.ndarray]:
        """取下一帧动作，buffer 空返回 None。"""
        with self._lock:
            if self._exec_ptr >= len(self._action_buffer):
                return None
            action = self._action_buffer[self._exec_ptr]
            self._exec_ptr += 1
            return action.copy()

    def buffer_remaining(self) -> int:
        """返回 buffer 中剩余未执行的帧数。"""
        with self._lock:
            return max(0, len(self._action_buffer) - self._exec_ptr)

    def should_trigger_inference(self) -> bool:
        """是否应该触发下一轮异步推理。"""
        return (self.buffer_remaining() <= self._pre_infer_threshold
                and not self._infer_in_progress)

    @property
    def infer_in_progress(self) -> bool:
        return self._infer_in_progress

    @property
    def new_chunk_ready(self) -> bool:
        return self._new_chunk_ready.is_set()

    def mark_inference_started(self):
        """异步推理线程开始时调用，记录当前 exec_ptr 用于计算延迟偏移。"""
        self._infer_in_progress = True
        with self._lock:
            self._infer_start_exec_ptr = self._exec_ptr

    def mark_inference_done(self, new_actions: list, meta: Optional[dict] = None):
        """异步推理线程完成时调用。"""
        self._new_chunk = [np.asarray(a, dtype=np.float32) for a in new_actions]
        self._new_chunk_meta = meta or {}
        self._new_chunk_ready.set()
        self._infer_in_progress = False

    def consume_new_chunk(self):
        """主线程消费异步推理结果，融合入 buffer。

        Returns (fusion_report, meta) or (None, None) if nothing is ready.
        """
        if not self._new_chunk_ready.is_set():
            return None, None
        self._new_chunk_ready.clear()
        chunk = self._new_chunk
        meta = self._new_chunk_meta or {}
        self._new_chunk = None
        self._new_chunk_meta = None
        # A failed/empty inference must not erase the still-valid tail of the
        # previous buffer. Report it and let the caller decide whether to stop.
        if chunk:
            return self.add_chunk(chunk), meta
        return {
            "mode": "empty",
            "skipped": 0,
            "start_index": 0,
            "original_len": 0,
            "kept": 0,
        }, meta

    # ------------------------------------------------------------------
    # Fusion methods
    # ------------------------------------------------------------------

    def _fuse_weighted(self, old_chunk, new_chunk):
        """加权线性过渡，权重从 old→new 线性变化。

        overlap = min(fusion_window, len(old), len(new))
        fused[i] = (1-t)*old[i] + t*new[i],  t = i/overlap
        """
        overlap = min(self._fusion_window, len(old_chunk), len(new_chunk))
        result = []
        for i in range(overlap):
            t = i / max(overlap, 1)
            w_old = 1.0 - t
            w_new = t
            result.append(w_old * old_chunk[i] + w_new * new_chunk[i])
        if len(new_chunk) > overlap:
            result.extend(new_chunk[overlap:])
        return result

    def _fuse_quintic_hermite(self, old_chunk, new_chunk):
        """五次 Hermite 样条融合，保证 C2 连续（位置/速度/加速度匹配）。

        基函数:
          h00(t) = 1 - 10t³ + 15t⁴ - 6t⁵   (h(0)=1, 其他零阶/一阶/二阶=0)
          h01(t) = 10t³ - 15t⁴ + 6t⁵       (h(1)=1, 其他零阶/一阶/二阶=0)

        fused[i] = h00(t) * old[i] + h01(t) * new[i]
        """
        overlap = min(self._fusion_window, len(old_chunk), len(new_chunk))
        old_arr = np.asarray(old_chunk, dtype=np.float64)
        new_arr = np.asarray(new_chunk, dtype=np.float64)
        dim = old_arr.shape[1]

        result = np.empty((max(overlap, len(new_chunk)), dim), dtype=np.float64)

        if overlap > 0:
            t_vals = np.linspace(0.0, 1.0, overlap, endpoint=False)  # 0, 1/N, 2/N, ...
            t2 = t_vals * t_vals
            t3 = t2 * t_vals
            t4 = t3 * t_vals
            t5 = t4 * t_vals
            h00 = 1.0 - 10.0 * t3 + 15.0 * t4 - 6.0 * t5   # (overlap,)
            h01 = 10.0 * t3 - 15.0 * t4 + 6.0 * t5          # (overlap,)
            for d in range(dim):
                result[:overlap, d] = h00 * old_arr[:overlap, d] + h01 * new_arr[:overlap, d]

        if len(new_chunk) > overlap:
            result[overlap:len(new_chunk)] = new_arr[overlap:]

        return [result[i].astype(np.float32) for i in range(len(new_chunk))]
