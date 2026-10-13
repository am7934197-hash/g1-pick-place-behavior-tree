"""
图片预处理：缩放图像以减小网络传输量。

pi05 模型内部会将任意尺寸输入 resize 到 224x224，
客户端提前缩放不影响推理精度，但可将单张 1280x960 (~3.7MB) 缩至 224x224 (~150KB)，
网络传输量降低约 25 倍，延迟从 ~1s 降至 ~50ms。
"""

import numpy as np
from typing import Optional

try:
    import cv2
except ImportError:
    cv2 = None


class ImagePreprocessor:
    """将观测 dict 中所有 ndarray 图像等比缩放 + 居中黑边填充到目标尺寸。

    Usage:
        pp = ImagePreprocessor({"enabled": True, "target_size": [224, 224]})
        obs = pp.process(obs)  # in-place: 图像 ndarray 被替换为缩小版
    """

    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        self.enabled = cfg.get("enabled", False)
        self._target_h, self._target_w = cfg.get("target_size", [224, 224])
        self._keep_ratio = cfg.get("keep_aspect_ratio", True)

    def process(self, obs: dict) -> dict:
        """处理 obs 中所有图像 ndarray。未启用时原样返回。"""
        if not obs or not self.enabled:
            return obs
        for k, v in obs.items():
            if isinstance(v, np.ndarray) and v.ndim == 3:
                obs[k] = self._resize(v)
        return obs

    def _resize(self, img: np.ndarray) -> np.ndarray:
        """等比缩放 + 居中黑边填充 → (target_h, target_w, C) uint8。

        输入为 observation_features 拉伸后的指定尺寸（如 480x640），
        此处再缩放 + 黑边填充到传输尺寸（如 224x224）。
        优先使用 OpenCV（比 PIL 快数倍），不可用时回退 PIL。
        """
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        h, w = img.shape[:2]

        if self._keep_ratio:
            ratio = max(w / self._target_w, h / self._target_h)
            new_h, new_w = int(h / ratio), int(w / ratio)
            if cv2 is not None:
                resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                from PIL import Image
                resized = np.array(
                    Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR),
                    dtype=np.uint8,
                )
            pad_h0 = (self._target_h - new_h) // 2
            pad_w0 = (self._target_w - new_w) // 2
            padded = np.zeros((self._target_h, self._target_w, 3), dtype=np.uint8)
            padded[pad_h0:pad_h0 + new_h, pad_w0:pad_w0 + new_w] = resized
            return padded
        else:
            if cv2 is not None:
                return cv2.resize(img, (self._target_w, self._target_h), interpolation=cv2.INTER_AREA)
            from PIL import Image
            return np.array(
                Image.fromarray(img).resize((self._target_w, self._target_h), Image.BILINEAR),
                dtype=np.uint8,
            )
