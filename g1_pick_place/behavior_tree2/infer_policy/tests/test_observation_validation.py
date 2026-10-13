import copy
import unittest

import numpy as np

import server
from image_preprocessor import ImagePreprocessor


class HeadTrainingCropValidationTest(unittest.TestCase):
    def setUp(self):
        self._old_cfg = server.CFG
        self._old_required_cameras = server.REQUIRED_CAMERAS
        self._old_required_state_names = server.REQUIRED_STATE_NAMES
        self._old_preprocessor = server.image_preprocessor

        server.CFG = copy.deepcopy(server.CFG)
        server.CFG["image_preprocess"] = {
            **(server.CFG.get("image_preprocess") or {}),
            "head_training_crop": {"enabled": True},
            "enforce_training_geometry": True,
        }
        server.REQUIRED_CAMERAS = ["head_right", "left_arm"]
        server.REQUIRED_STATE_NAMES = list(server.STATE_NAMES)
        server.image_preprocessor = ImagePreprocessor(server.CFG["image_preprocess"])

    def tearDown(self):
        server.CFG = self._old_cfg
        server.REQUIRED_CAMERAS = self._old_required_cameras
        server.REQUIRED_STATE_NAMES = self._old_required_state_names
        server.image_preprocessor = self._old_preprocessor

    def _observation(self, wrist_has_training_bars: bool) -> dict:
        head = np.full((224, 224, 3), 120, dtype=np.uint8)
        wrist = np.full((224, 224, 3), 120, dtype=np.uint8)
        if wrist_has_training_bars:
            wrist[:28] = 0
            wrist[196:] = 0
        obs = {"head_right": head, "left_arm": wrist}
        obs.update({name: 0.1 for name in server.REQUIRED_STATE_NAMES})
        return obs

    def test_cropped_head_skips_letterbox_check_but_wrist_still_passes_it(self):
        self.assertTrue(server.validate_observation(self._observation(True)))

    def test_wrist_without_training_bars_is_still_rejected(self):
        self.assertFalse(server.validate_observation(self._observation(False)))


if __name__ == "__main__":
    unittest.main()
