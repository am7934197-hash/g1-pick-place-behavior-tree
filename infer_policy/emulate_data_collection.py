#!/usr/bin/env python3
"""Emulate data-collection FOV on working-mode frames.

Two schemes (keep 4:3, never stretch to a square):

- via_640: crop, then INTER_LINEAR to 640x480, then the caller letterboxes to 224.
- crop_only: crop only, then the caller letterboxes to 224 (one downscale).

Heads: working 1280x960 is already rectified. Vendor data-collection
zoom_factor=3 applies to RAW fisheye; do not crop 1/3 of working frames.
Head emulate uses zoom=1 (keep 4:3). Wrists crop 16:9 to 4:3 first.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


DEFAULT_ZOOM_FACTOR = 1.0  # on already-rectified working frames; vendor RAW zoom=3 is not this
DEFAULT_OUT_WH = (640, 480)  # color_width, color_height
DEFAULT_SCHEME = "via_640"
VALID_SCHEMES = ("via_640", "crop_only")
HEAD_CAMERAS = ("head_left", "head_right")
WRIST_CAMERAS = ("left_arm", "right_arm")


def _interp(img: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    if cv2 is not None:
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    from PIL import Image
    return np.array(
        Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR),
        dtype=np.uint8,
    )


def normalize_scheme(value: Any) -> str:
    scheme = str(value or DEFAULT_SCHEME).strip()
    if scheme not in VALID_SCHEMES:
        raise ValueError(f"emulate_data_collection.scheme must be one of {VALID_SCHEMES}, got {value!r}")
    return scheme


def center_crop_zoom(
    img: np.ndarray,
    zoom_factor: float = DEFAULT_ZOOM_FACTOR,
) -> np.ndarray:
    """Center-crop by zoom_factor. Does not resize. zoom=1 returns img."""
    if img is None or not isinstance(img, np.ndarray) or img.ndim < 2:
        raise ValueError("img must be an HWC ndarray")
    zoom = float(zoom_factor)
    if zoom <= 0:
        raise ValueError(f"zoom_factor must be > 0, got {zoom_factor}")
    if abs(zoom - 1.0) < 1e-9:
        return img
    h, w = int(img.shape[0]), int(img.shape[1])
    crop_w = max(1, min(w, int(round(w / zoom))))
    crop_h = max(1, min(h, int(round(h / zoom))))
    x0 = (w - crop_w) // 2
    y0 = (h - crop_h) // 2
    return img[y0:y0 + crop_h, x0:x0 + crop_w]


def resize_to_wh(img: np.ndarray, out_wh: Tuple[int, int] = DEFAULT_OUT_WH) -> np.ndarray:
    h, w = int(img.shape[0]), int(img.shape[1])
    out_w, out_h = int(out_wh[0]), int(out_wh[1])
    if out_w <= 0 or out_h <= 0:
        raise ValueError(f"out_wh must be positive, got {out_wh}")
    if (h, w) == (out_h, out_w):
        return img
    return _interp(img, out_w, out_h)


def zoom_crop_to_size(
    img: np.ndarray,
    zoom_factor: float = DEFAULT_ZOOM_FACTOR,
    out_wh: Tuple[int, int] = DEFAULT_OUT_WH,
) -> np.ndarray:
    """Center-crop by zoom_factor then resize to ``out_wh`` (W, H)."""
    if img is None or not isinstance(img, np.ndarray) or img.ndim < 2:
        raise ValueError("img must be an HWC ndarray")
    h, w = int(img.shape[0]), int(img.shape[1])
    out_w, out_h = int(out_wh[0]), int(out_wh[1])
    if out_w <= 0 or out_h <= 0:
        raise ValueError(f"out_wh must be positive, got {out_wh}")
    if (h, w) == (out_h, out_w):
        return img
    cropped = center_crop_zoom(img, zoom_factor)
    return resize_to_wh(cropped, out_wh)


def emulate_fov(
    img: np.ndarray,
    zoom_factor: float,
    out_wh: Tuple[int, int],
    scheme: str = DEFAULT_SCHEME,
) -> np.ndarray:
    """Crop to data-collection FOV. via_640 also resizes to out_wh; crop_only does not."""
    scheme = normalize_scheme(scheme)
    h, w = int(img.shape[0]), int(img.shape[1])
    out_w, out_h = int(out_wh[0]), int(out_wh[1])
    if (h, w) == (out_h, out_w):
        return img
    cropped = center_crop_zoom(img, zoom_factor)
    if scheme == "crop_only":
        return cropped
    return resize_to_wh(cropped, out_wh)


def parse_camera_specs(emulate_cfg: Dict[str, Any] | None) -> Dict[str, Dict[str, Any]]:
    """Per-camera zoom / crop settings.

    Heads use zoom=1 on already-rectified working frames. Wrists crop 16:9 to
    4:3 then optionally resize to 640x480. A plain camera list keeps the global
    zoom_factor for those names only.
    """
    cfg = emulate_cfg or {}
    out_wh = cfg.get("output_wh", list(DEFAULT_OUT_WH))
    out_wh = (int(out_wh[0]), int(out_wh[1]))
    global_zoom = float(cfg.get("zoom_factor", DEFAULT_ZOOM_FACTOR))
    scheme = normalize_scheme(cfg.get("scheme", DEFAULT_SCHEME))
    cameras = cfg.get("cameras")
    specs: Dict[str, Dict[str, Any]] = {}

    def head_spec(zoom: float = DEFAULT_ZOOM_FACTOR) -> Dict[str, Any]:
        return {
            "zoom_factor": float(zoom),
            "out_wh": out_wh,
            "crop_to_training_aspect": False,
            "undistort_type": int(cfg.get("undistort_type", 1)),
            "scheme": scheme,
        }

    def wrist_spec() -> Dict[str, Any]:
        return {
            "zoom_factor": 1.0,
            "out_wh": out_wh,
            "crop_to_training_aspect": True,
            "undistort_type": 0,
            "scheme": scheme,
        }

    if cameras is None:
        for name in HEAD_CAMERAS:
            specs[name] = head_spec(global_zoom)
        for name in WRIST_CAMERAS:
            specs[name] = wrist_spec()
        return specs

    if isinstance(cameras, dict):
        for name, raw in cameras.items():
            key = str(name)
            item = dict(raw) if isinstance(raw, dict) else {}
            spec = wrist_spec() if key in WRIST_CAMERAS else head_spec(global_zoom)
            if "zoom_factor" in item:
                spec["zoom_factor"] = float(item["zoom_factor"])
            if "crop_to_training_aspect" in item:
                spec["crop_to_training_aspect"] = bool(item["crop_to_training_aspect"])
            if "undistort_type" in item:
                spec["undistort_type"] = int(item["undistort_type"])
            if "output_wh" in item:
                wh = item["output_wh"]
                spec["out_wh"] = (int(wh[0]), int(wh[1]))
            if "scheme" in item:
                spec["scheme"] = normalize_scheme(item["scheme"])
            specs[key] = spec
        return specs

    for name in cameras:
        key = str(name)
        specs[key] = head_spec(global_zoom) if key in HEAD_CAMERAS else wrist_spec()
        if key not in HEAD_CAMERAS and key not in WRIST_CAMERAS:
            specs[key] = head_spec(global_zoom)
    return specs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="working-mode image path")
    parser.add_argument("output", help="640x480 output path")
    parser.add_argument("--zoom-factor", type=float, default=DEFAULT_ZOOM_FACTOR)
    parser.add_argument("--out-w", type=int, default=DEFAULT_OUT_WH[0])
    parser.add_argument("--out-h", type=int, default=DEFAULT_OUT_WH[1])
    args = parser.parse_args()
    if cv2 is None:
        raise SystemExit("opencv is required for the CLI")
    src = cv2.imread(args.input)
    if src is None:
        raise SystemExit(f"could not read {args.input}")
    out = zoom_crop_to_size(src, args.zoom_factor, (args.out_w, args.out_h))
    if not cv2.imwrite(args.output, out):
        raise SystemExit(f"could not write {args.output}")
    print(f"{src.shape[1]}x{src.shape[0]} -> {out.shape[1]}x{out.shape[0]} wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
