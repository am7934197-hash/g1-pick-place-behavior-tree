import unittest

import numpy as np

from image_preprocessor import ImagePreprocessor


class ImagePreprocessorTest(unittest.TestCase):
    def setUp(self):
        self.processor = ImagePreprocessor({
            "enabled": True,
            "training_size": [480, 640],
            "target_size": [224, 224],
            "expected_native_size": [480, 640],
            "crop_to_training_aspect": False,
        })

    def test_data_collection_letterboxes_640x480(self):
        image = np.full((480, 640, 3), 127, dtype=np.uint8)
        result = self.processor.process_image(image)
        self.assertEqual(result.shape, (224, 224, 3))
        self.assertTrue(np.all(result[:28] == 0))
        self.assertTrue(np.all(result[196:] == 0))
        self.assertTrue(np.all(result[28:196] == 127))

    def test_rejects_working_mode_wrist_size(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "1280x720"):
            self.processor.process_image(image)

    def test_optional_crop_still_handles_16_9(self):
        processor = ImagePreprocessor({
            "enabled": True,
            "training_size": [480, 640],
            "target_size": [224, 224],
            "crop_to_training_aspect": True,
        })
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        image[:, 160:200, 1] = 255
        result = processor.process_image(image)
        self.assertEqual(result.shape, (224, 224, 3))
        self.assertGreater(int(result[28:196, :12, 1].max()), 200)


if __name__ == "__main__":
    unittest.main()
