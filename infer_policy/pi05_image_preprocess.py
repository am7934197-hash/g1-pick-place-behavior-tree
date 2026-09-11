"""Local replica of PI05 ``_preprocess_images`` (lerobot modeling_pi05.py).

Used to dump the tensor that SigLIP actually sees. The PolicyServer still runs
the real function; this replica must stay bit-compatible with:

    img = img.to(float32)           # uint8 0-255 becomes 0-255 float, NOT /255 here
    if HxW != 224: resize_with_pad
    img = img * 2 - 1

LeRobot dataset loaders usually already divide by 255 before the policy.
PolicyServer PNG decode restores uint8; if the server converts uint8->float32
without /255, *2-1 would be wrong. The replica therefore exposes both stages
so a run log can show which path was taken.

Official code (huggingface/lerobot modeling_pi05.py):
  - docstring: images typically [0,1], then *2-1 -> [-1,1]
  - resize_with_pad_torch on float32 clamps to [0,1]
  - if input is uint8 0-255 cast to float32 without /255, clamp would clip to 1

This module ALWAYS does one /255 on uint8 input, matching the training
dataloader + _preprocess_images contract, and records that it did so.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def count_bars(img: np.ndarray, threshold: float = 2.0) -> Tuple[int, int, int, int]:
    """Return (top, bottom, left, right) near-black margins."""
    if img.ndim == 3 and img.shape[0] == 3:
        hwc = np.transpose(img, (1, 2, 0))
    else:
        hwc = img
    if hwc.dtype != np.uint8:
        scale = 255.0 if float(np.nanmax(hwc)) <= 1.5 else 1.0
        vis = np.clip(np.asarray(hwc) * scale, 0, 255).astype(np.uint8)
    else:
        vis = hwc
    row_max = vis.max(axis=(1, 2))
    col_max = vis.max(axis=(0, 2))
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
    left = 0
    for value in col_max:
        if int(value) <= threshold:
            left += 1
        else:
            break
    right = 0
    for value in col_max[::-1]:
        if int(value) <= threshold:
            right += 1
        else:
            break
    return top, bottom, left, right


def _interp(img: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    if cv2 is not None:
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    from PIL import Image
    return np.array(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))


def resize_with_pad_uint8(img: np.ndarray, height: int = 224, width: int = 224) -> np.ndarray:
    """OpenPI/lerobot resize_with_pad geometry on HWC uint8."""
    h, w = int(img.shape[0]), int(img.shape[1])
    ratio = max(w / float(width), h / float(height))
    resized_h = max(1, int(h / ratio))
    resized_w = max(1, int(w / ratio))
    resized = _interp(img, resized_w, resized_h)
    pad_h0, rem_h = divmod(height - resized_h, 2)
    pad_h1 = pad_h0 + rem_h
    pad_w0, rem_w = divmod(width - resized_w, 2)
    pad_w1 = pad_w0 + rem_w
    padded = np.zeros((height, width, 3), dtype=np.uint8)
    padded[pad_h0:pad_h0 + resized_h, pad_w0:pad_w0 + resized_w] = resized
    return padded


def preprocess_images_pi05(images_hwc_uint8: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Run the documented PI05 image path once and return tensors + checks.

    Input: camera_key -> HWC uint8 RGB in 0-255.
    Output per camera:
      input_uint8, after_div255, after_resize, after_pm1 CHW float32
    """
    report: Dict[str, Any] = {"cameras": {}, "div255_applied": True, "pm1_applied": True}
    for key, image in images_hwc_uint8.items():
        arr = np.asarray(image)
        if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(
                f"{key}: expected HWC uint8 RGB, got shape={arr.shape} dtype={arr.dtype}"
            )
        in_min = int(arr.min())
        in_max = int(arr.max())
        if in_min < 0 or in_max > 255:
            raise ValueError(f"{key}: uint8 range {in_min}-{in_max} is invalid")

        as_float = arr.astype(np.float32) / 255.0
        h, w = arr.shape[:2]
        resized_again = False
        if (h, w) != (224, 224):
            padded_u8 = resize_with_pad_uint8(arr, 224, 224)
            as_float = padded_u8.astype(np.float32) / 255.0
            resized_again = True
        else:
            padded_u8 = arr
        pm1 = as_float * 2.0 - 1.0
        chw = np.transpose(pm1, (2, 0, 1))
        top, bottom, left, right = count_bars(padded_u8)
        # RGB check: channel 0 is red. A red-dominant patch in the non-bar region.
        content = padded_u8[top:224 - bottom if bottom else 224, left:224 - right if right else 224]
        rgb_means = [float(content[:, :, c].mean()) for c in range(3)] if content.size else [0, 0, 0]
        report["cameras"][key] = {
            "input_shape": list(arr.shape),
            "input_dtype": str(arr.dtype),
            "input_min": in_min,
            "input_max": in_max,
            "div255_min": float(as_float.min()),
            "div255_max": float(as_float.max()),
            "tensor_shape": list(chw.shape),
            "tensor_dtype": str(chw.dtype),
            "tensor_min": float(chw.min()),
            "tensor_max": float(chw.max()),
            "resized_again": resized_again,
            "black_bars_tblr": [top, bottom, left, right],
            "content_hw": [int(content.shape[0]), int(content.shape[1])] if content.size else [0, 0],
            "content_channel_means_rgb": rgb_means,
            "layout": "CHW",
            "color": "RGB",
        }
    return report
