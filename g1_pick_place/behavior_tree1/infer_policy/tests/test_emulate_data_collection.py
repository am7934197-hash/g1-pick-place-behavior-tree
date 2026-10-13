import unittest

import numpy as np

from emulate_data_collection import (
    center_crop_zoom,
    emulate_fov,
    parse_camera_specs,
    zoom_crop_to_size,
)


class ZoomCropToSizeTest(unittest.TestCase):
    def test_1280x960_zoom3_becomes_640x480(self):
        src = np.zeros((960, 1280, 3), dtype=np.uint8)
        crop_w = int(round(1280 / 3.0))
        crop_h = int(round(960 / 3.0))
        x0 = (1280 - crop_w) // 2
        y0 = (960 - crop_h) // 2
        src[y0:y0 + crop_h, x0:x0 + crop_w] = (10, 20, 30)
        out = zoom_crop_to_size(src, zoom_factor=3.0, out_wh=(640, 480))
        self.assertEqual(out.shape, (480, 640, 3))
        self.assertGreater(int(out.max()), 0)
        self.assertTrue(np.all(out[..., 0] >= 8))

    def test_already_640x480_is_skipped(self):
        src = np.arange(480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3)
        out = zoom_crop_to_size(src, zoom_factor=3.0, out_wh=(640, 480))
        self.assertIs(out, src)
        self.assertEqual(out.shape, (480, 640, 3))

    def test_crop_uses_image_center(self):
        src = np.zeros((960, 1280, 3), dtype=np.uint8)
        src[:, :, 0] = 1
        src[470:490, 630:650, 1] = 200
        out = zoom_crop_to_size(src, zoom_factor=3.0, out_wh=(640, 480))
        self.assertEqual(out.shape, (480, 640, 3))
        self.assertGreater(int(out[:, :, 1].max()), 100)


class ParseCameraSpecsTest(unittest.TestCase):
    def test_default_includes_both_heads_and_wrists(self):
        specs = parse_camera_specs({"enabled": True})
        self.assertEqual(specs["head_left"]["zoom_factor"], 1.0)
        self.assertEqual(specs["head_right"]["zoom_factor"], 1.0)
        self.assertFalse(specs["head_left"]["crop_to_training_aspect"])
        self.assertEqual(specs["left_arm"]["zoom_factor"], 1.0)
        self.assertTrue(specs["left_arm"]["crop_to_training_aspect"])
        self.assertTrue(specs["right_arm"]["crop_to_training_aspect"])

    def test_dict_config_keeps_head_zoom_and_wrist_crop(self):
        specs = parse_camera_specs({
            "output_wh": [640, 480],
            "cameras": {
                "head_left": {"undistort_type": 1, "zoom_factor": 3.0},
                "head_right": {"undistort_type": 1, "zoom_factor": 3.0},
                "left_arm": {"zoom_factor": 1.0, "crop_to_training_aspect": True},
                "right_arm": {"zoom_factor": 1.0, "crop_to_training_aspect": True},
            },
        })
        self.assertEqual(set(specs), {"head_left", "head_right", "left_arm", "right_arm"})
        self.assertEqual(specs["head_right"]["zoom_factor"], 3.0)
        self.assertTrue(specs["right_arm"]["crop_to_training_aspect"])
        self.assertEqual(specs["head_left"]["scheme"], "via_640")

    def test_scheme_crop_only_is_recorded(self):
        specs = parse_camera_specs({"scheme": "crop_only"})
        self.assertEqual(specs["head_left"]["scheme"], "crop_only")
        self.assertEqual(specs["left_arm"]["scheme"], "crop_only")


class DualSchemeTest(unittest.TestCase):
    def test_center_crop_zoom_does_not_resize(self):
        src = np.zeros((960, 1280, 3), dtype=np.uint8)
        cropped = center_crop_zoom(src, 3.0)
        self.assertEqual(cropped.shape[:2], (320, 427))

    def test_via_640_resizes_crop_only_does_not(self):
        src = np.full((960, 1280, 3), 180, dtype=np.uint8)
        via = emulate_fov(src, 3.0, (640, 480), "via_640")
        only = emulate_fov(src, 3.0, (640, 480), "crop_only")
        self.assertEqual(via.shape[:2], (480, 640))
        self.assertEqual(only.shape[:2], (320, 427))


if __name__ == "__main__":
    unittest.main()
