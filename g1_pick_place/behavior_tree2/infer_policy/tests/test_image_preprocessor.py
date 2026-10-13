import unittest
from unittest.mock import patch

import numpy as np

import image_preprocessor
from image_preprocessor import ImagePreprocessor, count_letterbox_black_bar_rows


class HeadTrainingCropTest(unittest.TestCase):
    @staticmethod
    def _processor(enabled: bool = True, target_size=None) -> ImagePreprocessor:
        return ImagePreprocessor({
            "enabled": True,
            "target_size": target_size or [224, 224],
            "keep_aspect_ratio": True,
            "head_training_crop": {"enabled": enabled},
        })

    def test_head_crop_matches_training_coordinates_and_lanczos(self):
        yy, xx = np.indices((224, 224), dtype=np.uint8)
        src = np.stack((xx, yy, (xx.astype(np.uint16) + yy) % 256), axis=2).astype(np.uint8)

        actual = self._processor().process({"head_right": src.copy()})["head_right"]
        resized = image_preprocessor.cv2.resize(
            src[126:214, 52:172],
            (224, 164),
            interpolation=image_preprocessor.cv2.INTER_LANCZOS4,
        )
        expected = image_preprocessor.cv2.copyMakeBorder(
            resized,
            top=30,
            bottom=30,
            left=0,
            right=0,
            borderType=image_preprocessor.cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

        self.assertEqual(actual.shape, (224, 224, 3))
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(np.all(actual[:30] == 0))
        self.assertTrue(np.all(actual[194:] == 0))

    def test_head_crop_preserves_rgb_channel_order(self):
        rgb = np.empty((224, 224, 3), dtype=np.uint8)
        rgb[:] = (240, 80, 10)

        actual = self._processor().process({"head_left": rgb})["head_left"]

        np.testing.assert_array_equal(actual[112, 112], np.array([240, 80, 10], dtype=np.uint8))

    def test_disabled_crop_leaves_ready_head_unchanged(self):
        src = np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3)

        actual = self._processor(enabled=False).process({"head_right": src.copy()})["head_right"]

        np.testing.assert_array_equal(actual, src)

    def test_enabled_crop_does_not_change_wrists(self):
        wrist = np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3)
        actual = self._processor().process({
            "left_arm": wrist.copy(),
            "right_arm": wrist.copy(),
        })

        np.testing.assert_array_equal(actual["left_arm"], wrist)
        np.testing.assert_array_equal(actual["right_arm"], wrist)

    def test_crop_rejects_non_224_preprocessed_head(self):
        with self.assertRaisesRegex(ValueError, "expects head_right at 224x224"):
            self._processor(target_size=[200, 200]).process({
                "head_right": np.ones((224, 224, 3), dtype=np.uint8),
            })


class EmulateDataCollectionPreprocessTest(unittest.TestCase):
    def setUp(self):
        self.pp = ImagePreprocessor({
            "enabled": True,
            "target_size": [224, 224],
            "keep_aspect_ratio": True,
            "crop_to_training_aspect": True,
            "training_hw": [480, 640],
            "emulate_data_collection": {
                "enabled": True,
                "output_wh": [640, 480],
                "cameras": {
                    "head_left": {"undistort_type": 1, "zoom_factor": 1.0},
                    "head_right": {"undistort_type": 1, "zoom_factor": 1.0},
                    "left_arm": {"zoom_factor": 1.0, "crop_to_training_aspect": True},
                    "right_arm": {"zoom_factor": 1.0, "crop_to_training_aspect": True},
                },
            },
        })

    def test_working_head_letterbox_keeps_28px_bars(self):
        src = np.full((960, 1280, 3), 180, dtype=np.uint8)
        out = self.pp.process({"head_left": src})
        img = out["head_left"]
        self.assertEqual(img.shape, (224, 224, 3))
        top, bottom = count_letterbox_black_bar_rows(img)
        self.assertEqual(top, 28)
        self.assertEqual(bottom, 28)
        self.assertTrue(np.all(img[:28] == 0))
        self.assertTrue(np.all(img[196:] == 0))
        self.assertGreater(int(img[28:196].max()), 0)

    def test_already_640x480_head_is_not_zoomed_again(self):
        src = np.full((480, 640, 3), 180, dtype=np.uint8)
        src[0, 0] = (9, 8, 7)
        out = self.pp.process({"head_left": src.copy()})
        img = out["head_left"]
        self.assertEqual(img.shape, (224, 224, 3))
        top, bottom = count_letterbox_black_bar_rows(img)
        self.assertEqual(top, 28)
        self.assertEqual(bottom, 28)

    def test_head_right_matches_head_left_geometry(self):
        src = np.full((960, 1280, 3), 180, dtype=np.uint8)
        out = self.pp.process({"head_right": src})
        img = out["head_right"]
        self.assertEqual(img.shape, (224, 224, 3))
        top, bottom = count_letterbox_black_bar_rows(img)
        self.assertEqual(top, 28)
        self.assertEqual(bottom, 28)

    def test_wrist_is_not_zoom_cropped(self):
        wrist = np.zeros((720, 1280, 3), dtype=np.uint8)
        # Visible after 16:9->4:3 crop (x=160..1120); outside zoom-3 crop (x=426..853).
        wrist[:, 160:200] = (0, 255, 0)
        out = self.pp.process({"left_arm": wrist})
        content = out["left_arm"][28:196]
        self.assertEqual(out["left_arm"].shape, (224, 224, 3))
        self.assertGreater(int(content[:, :24, 1].max()), 200)

    def test_ready_480x640_wrists_skip_legacy_fov_emulation(self):
        wrist = np.full((480, 640, 3), 160, dtype=np.uint8)
        with patch("image_preprocessor.emulate_fov", side_effect=AssertionError):
            out = self.pp.process({
                "left_arm": wrist.copy(),
                "right_arm": wrist.copy(),
            })

        for camera in ("left_arm", "right_arm"):
            self.assertEqual(out[camera].shape, (224, 224, 3))
            self.assertEqual(count_letterbox_black_bar_rows(out[camera]), (28, 28))

    def _pp(self, scheme: str) -> ImagePreprocessor:
        cfg = {
            "enabled": True,
            "target_size": [224, 224],
            "keep_aspect_ratio": True,
            "crop_to_training_aspect": True,
            "training_hw": [480, 640],
            "emulate_data_collection": {
                "enabled": True,
                "scheme": scheme,
                "output_wh": [640, 480],
                "cameras": {
                    "head_left": {"undistort_type": 1, "zoom_factor": 1.0},
                    "left_arm": {"zoom_factor": 1.0, "crop_to_training_aspect": True},
                },
            },
        }
        return ImagePreprocessor(cfg)

    def test_via_640_head_has_28px_bars(self):
        src = np.full((960, 1280, 3), 180, dtype=np.uint8)
        img = self._pp("via_640").process({"head_left": src})["head_left"]
        self.assertEqual(img.shape, (224, 224, 3))
        self.assertEqual(count_letterbox_black_bar_rows(img), (28, 28))

    def test_crop_only_head_letterboxes_without_640(self):
        src = np.full((960, 1280, 3), 180, dtype=np.uint8)
        img = self._pp("crop_only").process({"head_left": src})["head_left"]
        self.assertEqual(img.shape, (224, 224, 3))
        top, bottom = count_letterbox_black_bar_rows(img)
        self.assertLessEqual(abs(top - 28), 4)
        self.assertLessEqual(abs(bottom - 28), 4)

    def test_wrist_both_schemes_keep_training_bars(self):
        wrist = np.full((720, 1280, 3), 160, dtype=np.uint8)
        for scheme in ("via_640", "crop_only"):
            img = self._pp(scheme).process({"left_arm": wrist})["left_arm"]
            top, bottom = count_letterbox_black_bar_rows(img)
            self.assertEqual(img.shape, (224, 224, 3), scheme)
            self.assertLessEqual(abs(top - 28), 4, scheme)
            self.assertLessEqual(abs(bottom - 28), 4, scheme)


class DirectLetterboxPreprocessTest(unittest.TestCase):
    def setUp(self):
        self.pp = ImagePreprocessor({
            "enabled": True,
            "target_size": [224, 224],
            "keep_aspect_ratio": True,
            "crop_to_training_aspect": False,
            "emulate_data_collection": {"enabled": False},
        })

    def test_head_letterboxes_native_1280x960(self):
        src = np.full((960, 1280, 3), 180, dtype=np.uint8)
        img = self.pp.process({"head_left": src})["head_left"]
        self.assertEqual(img.shape, (224, 224, 3))
        self.assertEqual(count_letterbox_black_bar_rows(img), (28, 28))

    def test_wrist_letterboxes_native_1280x720(self):
        src = np.full((720, 1280, 3), 180, dtype=np.uint8)
        img = self.pp.process({"left_arm": src})["left_arm"]
        self.assertEqual(img.shape, (224, 224, 3))
        top, bottom = count_letterbox_black_bar_rows(img)
        self.assertEqual((top, bottom), (49, 49))

    def test_rectangular_pick_target_uses_full_480x640_canvas(self):
        pp = ImagePreprocessor({
            "enabled": True,
            "target_size": [480, 640],
            "keep_aspect_ratio": True,
            "crop_to_training_aspect": False,
        })
        head = np.full((960, 1280, 3), 180, dtype=np.uint8)
        wrist = np.full((720, 1280, 3), 180, dtype=np.uint8)
        result = pp.process({"head_left": head, "left_arm": wrist})
        self.assertEqual(result["head_left"].shape, (480, 640, 3))
        self.assertEqual(count_letterbox_black_bar_rows(result["head_left"]), (0, 0))
        self.assertEqual(result["left_arm"].shape, (480, 640, 3))
        self.assertEqual(count_letterbox_black_bar_rows(result["left_arm"]), (60, 60))



if __name__ == "__main__":
    unittest.main()
