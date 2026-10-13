"""Image preprocessing for data-collection camera frames."""

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


class ImagePreprocessor:
    """Letterbox native 4:3 frames to the model target size.

    Data-collection capture already outputs 640x480. Do not center-crop
    working-mode 16:9 wrists; that FOV is applied by the camera driver.
    """

    def __init__(self, config: dict):
        self.enabled = bool(config.get("enabled", True))
        target = config.get("target_size", [224, 224])
        self.target_h, self.target_w = int(target[0]), int(target[1])
        training = config.get("training_size", [480, 640])
        self.training_h, self.training_w = int(training[0]), int(training[1])
        self.crop_to_training_aspect = bool(
            config.get("crop_to_training_aspect", False)
        )
        expected = config.get("expected_native_size")
        self.expected_native_hw: Optional[Tuple[int, int]] = None
        if expected:
            self.expected_native_hw = (int(expected[0]), int(expected[1]))

    def process(self, observation: dict) -> dict:
        if not self.enabled:
            return observation
        result = dict(observation)
        for key, value in observation.items():
            if isinstance(value, np.ndarray) and value.ndim == 3:
                result[key] = self.process_image(value)
        return result

    def process_image(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected HWC 3-channel image, got {image.shape}")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        self._check_native_size(image)
        if self.crop_to_training_aspect:
            image = self._center_crop(image)

        src_h, src_w = image.shape[:2]
        scale = min(self.target_h / src_h, self.target_w / src_w)
        out_h = max(1, min(self.target_h, int(round(src_h * scale))))
        out_w = max(1, min(self.target_w, int(round(src_w * scale))))
        resized = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((self.target_h, self.target_w, 3), dtype=np.uint8)
        y0 = (self.target_h - out_h) // 2
        x0 = (self.target_w - out_w) // 2
        canvas[y0:y0 + out_h, x0:x0 + out_w] = resized
        return canvas

    def _check_native_size(self, image: np.ndarray) -> None:
        if self.expected_native_hw is None:
            return
        src_h, src_w = image.shape[:2]
        expected_h, expected_w = self.expected_native_hw
        if (src_h, src_w) != (expected_h, expected_w):
            raise ValueError(
                f"native image {src_w}x{src_h}, expected data-collection "
                f"{expected_w}x{expected_h}"
            )

    def _center_crop(self, image: np.ndarray) -> np.ndarray:
        src_h, src_w = image.shape[:2]
        target_ratio = self.training_w / self.training_h
        src_ratio = src_w / src_h
        if abs(src_ratio - target_ratio) < 1e-6:
            return image
        if src_ratio > target_ratio:
            crop_w = max(1, int(round(src_h * target_ratio)))
            x0 = (src_w - crop_w) // 2
            return image[:, x0:x0 + crop_w]
        crop_h = max(1, int(round(src_w / target_ratio)))
        y0 = (src_h - crop_h) // 2
        return image[y0:y0 + crop_h, :]


def validate_image_shape(image: np.ndarray, expected: Sequence[int]) -> None:
    expected_shape = tuple(int(v) for v in expected)
    if image.shape != expected_shape:
        raise ValueError(f"image shape {image.shape}, expected {expected_shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"image dtype {image.dtype}, expected uint8")
    if image.size == 0 or not np.any(image):
        raise ValueError("image is empty or all-zero")
