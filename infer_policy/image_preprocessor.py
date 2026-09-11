"""Client-side image preprocessing for per-checkpoint input geometry."""

from typing import Optional, Set, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from emulate_data_collection import emulate_fov, parse_camera_specs


TRAINING_IMAGE_HW = (480, 640)  # H, W; 4:3 source used to build the 224 videos


def letterbox_content_size(
    src_h: int, src_w: int, target_h: int = 224, target_w: int = 224
) -> Tuple[int, int]:
    """Content HxW after long-edge INTER_LINEAR scale into target, before padding."""
    scale = min(target_h / src_h, target_w / src_w)
    new_h = max(1, min(target_h, int(round(src_h * scale))))
    new_w = max(1, min(target_w, int(round(src_w * scale))))
    return new_h, new_w


def letterbox_pad_hw(
    src_h: int, src_w: int, target_h: int = 224, target_w: int = 224
) -> Tuple[int, int, int, int]:
    """Return (pad_top, pad_bottom, pad_left, pad_right) for a native HxW frame."""
    new_h, new_w = letterbox_content_size(src_h, src_w, target_h, target_w)
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    return pad_top, target_h - new_h - pad_top, pad_left, target_w - new_w - pad_left


def crop_to_training_aspect(
    img: np.ndarray,
    training_hw: Tuple[int, int] = TRAINING_IMAGE_HW,
    rtol: float = 0.02,
) -> np.ndarray:
    """Center-crop to the training 4:3 aspect. Does not stretch.

    Wrist cameras on this robot are 720×1280 (16:9). Training videos are
    480×640 (4:3). Cropping 1280→960 keeps height 720, then letterbox
    produces the same 168×224 content + 28px bars as 480×640.
    """
    h, w = img.shape[:2]
    if h <= 0 or w <= 0 or native_matches_training_aspect(h, w, training_hw, rtol):
        return img
    train_h, train_w = training_hw
    target_aspect = train_h / train_w
    native_aspect = h / w
    if native_aspect < target_aspect:
        new_w = max(1, min(w, int(round(h / target_aspect))))
        x0 = (w - new_w) // 2
        return img[:, x0:x0 + new_w]
    new_h = max(1, min(h, int(round(w * target_aspect))))
    y0 = (h - new_h) // 2
    return img[y0:y0 + new_h, :]


def native_matches_training_aspect(
    src_h: int, src_w: int, training_hw: Tuple[int, int] = TRAINING_IMAGE_HW, rtol: float = 0.02
) -> bool:
    """True when native aspect matches the 480x640 (4:3) training videos."""
    if src_h <= 0 or src_w <= 0:
        return False
    train_h, train_w = training_hw
    native_aspect = src_h / src_w
    train_aspect = train_h / train_w
    return abs(native_aspect - train_aspect) <= rtol * train_aspect


def describe_camera_geometry(
    native_img: np.ndarray,
    processed_img: Optional[np.ndarray] = None,
    training_hw: Tuple[int, int] = TRAINING_IMAGE_HW,
    crop_to_training: bool = False,
) -> dict:
    """Native HxW vs the 224 letterbox that PolicyServer will see."""
    h, w = int(native_img.shape[0]), int(native_img.shape[1])
    c = int(native_img.shape[2]) if native_img.ndim == 3 else 1
    cropped = crop_to_training_aspect(native_img, training_hw) if crop_to_training else native_img
    ch, cw = int(cropped.shape[0]), int(cropped.shape[1])
    pad_top, pad_bottom, pad_left, pad_right = letterbox_pad_hw(ch, cw)
    report = {
        "native_shape": [h, w, c],
        "native_dtype": str(native_img.dtype),
        "aspect_h_over_w": round(h / w, 4) if w else None,
        "matches_training_4_3": native_matches_training_aspect(h, w, training_hw),
        "crop_to_training_aspect": bool(crop_to_training),
        "cropped_shape": [ch, cw, c],
        "training_hw": list(training_hw),
        "expected_letterbox_pad": {
            "top": pad_top,
            "bottom": pad_bottom,
            "left": pad_left,
            "right": pad_right,
        },
    }
    if processed_img is not None and isinstance(processed_img, np.ndarray) and processed_img.ndim == 3:
        top_bar, bottom_bar = count_letterbox_black_bar_rows(processed_img)
        report["processed_shape"] = [int(x) for x in processed_img.shape]
        report["processed_black_bar_rows"] = {"top": top_bar, "bottom": bottom_bar}
    return report


def count_letterbox_black_bar_rows(img: np.ndarray, threshold: int = 2) -> Tuple[int, int]:
    """Count near-black rows from the top and bottom of an HWC uint8 image."""
    if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[0] == 0:
        return 0, 0
    row_max = img.max(axis=(1, 2))
    top = 0
    for value in row_max:
        if int(value) <= threshold:
            top += 1
        else:
            break
    bottom = 0
    for value in row_max[::-1]:
        if int(value) <= threshold:
            bottom += 1
        else:
            break
    return top, bottom


class ImagePreprocessor:
    """Letterbox to target_size before transport. PI05 路径下应 enabled=true。

    Usage:
        pp = ImagePreprocessor({"enabled": True, "target_size": [224, 224]})
        obs = pp.process(obs)
    """

    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        self.enabled = cfg.get("enabled", False)
        self._target_h, self._target_w = cfg.get("target_size", [224, 224])
        self._keep_ratio = cfg.get("keep_aspect_ratio", True)
        # This remains opt-in at the utility level so callers must state their
        # geometry contract. The production config pins it on: ready_224 was
        # built from 640x480 content and has 28px top/bottom bars at 224.
        self._crop_to_training = bool(cfg.get("crop_to_training_aspect", False))
        train_hw = cfg.get("training_hw", list(TRAINING_IMAGE_HW))
        self._training_hw = (int(train_hw[0]), int(train_hw[1]))
        emulate_cfg = cfg.get("emulate_data_collection") or {}
        self._emulate_dc = bool(emulate_cfg.get("enabled", False))
        self._emulate_specs = parse_camera_specs(emulate_cfg) if self._emulate_dc else {}
        self._emulate_cameras: Set[str] = set(self._emulate_specs)
        self._emulate_zoom = {
            name: float(spec["zoom_factor"]) for name, spec in self._emulate_specs.items()
        }
        self._emulate_out_wh = next(
            (spec["out_wh"] for spec in self._emulate_specs.values()),
            (640, 480),
        )
        self._emulate_scheme = next(
            (spec.get("scheme", "via_640") for spec in self._emulate_specs.values()),
            "via_640",
        )

    def process(self, obs: dict) -> dict:
        """处理 obs 中所有图像 ndarray。未启用时原样返回。"""
        if not obs or not self.enabled:
            return obs
        for k, v in obs.items():
            if isinstance(v, np.ndarray) and v.ndim == 3:
                img = v
                spec = self._emulate_specs.get(k) if self._emulate_dc else None
                if spec:
                    if spec.get("crop_to_training_aspect"):
                        img = crop_to_training_aspect(img, self._training_hw)
                    img = emulate_fov(
                        img,
                        spec["zoom_factor"],
                        spec["out_wh"],
                        spec.get("scheme", self._emulate_scheme),
                    )
                obs[k] = self._resize(img)
        return obs

    def _resize(self, img: np.ndarray) -> np.ndarray:
        """按长边等比缩进 target，居中黑边 → (target_h, target_w, C) uint8。"""
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            return np.zeros((self._target_h, self._target_w, 3), dtype=np.uint8)

        if self._crop_to_training:
            img = crop_to_training_aspect(img, self._training_hw)
            h, w = img.shape[:2]

        if self._keep_ratio:
            scale = min(self._target_h / h, self._target_w / w)
            new_h = max(1, min(self._target_h, int(round(h * scale))))
            new_w = max(1, min(self._target_w, int(round(w * scale))))
            resized = self._interp(img, new_w, new_h)
            pad_h0 = (self._target_h - new_h) // 2
            pad_w0 = (self._target_w - new_w) // 2
            padded = np.zeros((self._target_h, self._target_w, 3), dtype=np.uint8)
            padded[pad_h0:pad_h0 + new_h, pad_w0:pad_w0 + new_w] = resized
            return padded
        return self._interp(img, self._target_w, self._target_h)

    def _interp(self, img: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
        if cv2 is not None:
            return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        from PIL import Image
        return np.array(
            Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR),
            dtype=np.uint8,
        )
