"""
动作滤波模块 — 在关节指令下发前平滑信号，防止机器人抖动。

用法:
    from action_filter import create_action_filter
    flt = create_action_filter(cfg, dim=23)
    filtered = flt(raw_action)

支持:
    - none:            直通，不做滤波
    - lowpass:         一阶指数平滑 (EMA)
    - moving_average:  滑动窗口均值
    - one_euro:        1€ 自适应滤波（速度高时响应快，静止时平滑强）
"""

from __future__ import annotations

import numpy as np
from typing import Optional


# ============================================================================
# 滤波器基类
# ============================================================================

class BaseFilter:
    dim: int

    def __init__(self, dim: int = 23):
        self.dim = dim

    def reset(self):
        """重置内部状态（新一轮推理开始时调用）。"""
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x


# ============================================================================
# 一阶低通滤波 (EMA)
# ============================================================================

class LowPassFilter(BaseFilter):
    """Exponential Moving Average — 一阶指数平滑。

    y[t] = alpha * x[t] + (1 - alpha) * y[t-1]

    alpha 越小 → 越平滑但响应慢；alpha 越大 → 响应快但平滑弱。
    """

    def __init__(self, dim: int = 23, alpha: float = 0.3):
        super().__init__(dim)
        self.alpha = float(alpha)
        self._state: Optional[np.ndarray] = None

    def reset(self):
        self._state = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self._state is None:
            self._state = x.astype(np.float32).copy()
            return self._state.copy()
        self._state = self.alpha * x + (1.0 - self.alpha) * self._state
        return self._state.copy()


# ============================================================================
# 滑动均值滤波
# ============================================================================

class MovingAverageFilter(BaseFilter):
    """滑动窗口均值 — 每个关节取最近 window_size 步的平均值。

    window_size 越大 → 越平滑但延迟越高。
    """

    def __init__(self, dim: int = 23, window_size: int = 5):
        super().__init__(dim)
        self.window_size = max(1, int(window_size))
        self._buffer = np.zeros((self.window_size, dim), dtype=np.float32)
        self._idx = 0
        self._count = 0

    def reset(self):
        self._buffer.fill(0)
        self._idx = 0
        self._count = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self._buffer[self._idx] = x
        self._idx = (self._idx + 1) % self.window_size
        if self._count < self.window_size:
            self._count += 1
        return self._buffer[:self._count].mean(axis=0).astype(np.float32)


# ============================================================================
# 1€ 自适应滤波
# ============================================================================

class OneEuroFilter(BaseFilter):
    """1€ Filter — 自适应截止频率低通滤波。

    参考文献: Casiez et al., "1€ Filter: A Simple Speed-based Low-pass
    Filter for Noisy Input in Interactive Systems", CHI 2012.

    原理:
        - 用信号变化速度（dx/dt）自适应调整截止频率
        - 静止/慢速时截止频率低 → 强去抖
        - 快速移动时截止频率高 → 低延迟响应

    参数:
        freq:       数据采样率 (Hz)，默认 30
        min_cutoff: 最小截止频率 (Hz)，静止时的平滑强度，越小越平滑，推荐 0.5~2
        beta:       速度增益，控制响应灵敏度，0 表示固定截止频率，推荐 0.005~0.05
        d_cutoff:   速度估计的截止频率 (Hz)，通常 1.0
    """

    def __init__(self, dim: int = 23, freq: float = 30.0,
                 min_cutoff: float = 1.0, beta: float = 0.01,
                 d_cutoff: float = 1.0):
        super().__init__(dim)
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        self._x_prev: Optional[np.ndarray] = None
        self._dx_prev: Optional[np.ndarray] = None
        self._first_time = True

    def reset(self):
        self._x_prev = None
        self._dx_prev = None
        self._first_time = True

    @staticmethod
    def _alpha(cutoff: float, freq: float) -> float:
        """Compute smoothing factor from cutoff frequency and sample rate."""
        tau = 1.0 / (2.0 * np.pi * cutoff)
        te = 1.0 / freq
        return float(te / (te + tau))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)

        if self._first_time:
            self._x_prev = x.copy()
            self._dx_prev = np.zeros(self.dim, dtype=np.float32)
            self._first_time = False
            return x.copy()

        # Estimate derivative (velocity)
        dx = (x - self._x_prev) * self.freq
        alpha_d = self._alpha(self.d_cutoff, self.freq)
        self._dx_prev = alpha_d * dx + (1.0 - alpha_d) * self._dx_prev

        # Adaptive cutoff based on signal speed
        speed = np.abs(self._dx_prev)
        cutoff = self.min_cutoff + self.beta * speed  # per-joint adaptive cutoff
        alpha = self._alpha(cutoff, self.freq)

        # Apply low-pass with adaptive alpha
        self._x_prev = alpha * x + (1.0 - alpha) * self._x_prev
        return self._x_prev.copy()


# ============================================================================
# 工厂函数
# ============================================================================

def create_action_filter(cfg: dict, dim: int = 23) -> BaseFilter:
    """从配置字典创建动作滤波器。

    cfg 格式 (对应 yaml 中 robot.control.action_filter):
        enabled: true
        type: "lowpass"          # none / lowpass / moving_average / one_euro
        lowpass:
          alpha: 0.3
        moving_average:
          window_size: 5
        one_euro:
          freq: 30.0
          min_cutoff: 1.0
          beta: 0.01
          d_cutoff: 1.0
    """
    enabled = cfg.get("enabled", False)
    filter_type = cfg.get("type", "none")

    if not enabled or filter_type == "none":
        return BaseFilter(dim)

    if filter_type == "lowpass":
        lp_cfg = cfg.get("lowpass", {})
        return LowPassFilter(dim=dim, alpha=lp_cfg.get("alpha", 0.3))

    if filter_type == "moving_average":
        ma_cfg = cfg.get("moving_average", {})
        return MovingAverageFilter(dim=dim, window_size=ma_cfg.get("window_size", 5))

    if filter_type == "one_euro":
        oe_cfg = cfg.get("one_euro", {})
        return OneEuroFilter(
            dim=dim,
            freq=oe_cfg.get("freq", 30.0),
            min_cutoff=oe_cfg.get("min_cutoff", 1.0),
            beta=oe_cfg.get("beta", 0.01),
            d_cutoff=oe_cfg.get("d_cutoff", 1.0),
        )

    raise ValueError(f"Unknown action filter type: {filter_type}")
