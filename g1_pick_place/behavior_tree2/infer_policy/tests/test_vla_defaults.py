import unittest

from alignment_schema import (
    ACTION_NAMES,
    CAMERA_KEYS,
    HARDWARE_CAMERA_KEYS,
    STATE_NAMES,
    apply_vla_defaults,
    make_observation_features,
)


class VLADefaultsTest(unittest.TestCase):
    def test_empty_config_gets_pick_schema(self):
        cfg = apply_vla_defaults({})
        self.assertEqual(cfg["policy_type"], "pi05")
        self.assertEqual(cfg["device"], "cuda")
        self.assertEqual(cfg["fps"], 30)
        self.assertEqual(cfg["actions_per_chunk"], 50)
        self.assertEqual(cfg["observation_transport"]["image_encoding"], "png")
        self.assertEqual(cfg["image_preprocess"]["target_size"], [224, 224])
        self.assertTrue(cfg["image_preprocess"]["enabled"])
        self.assertEqual(cfg["robot"]["control"]["dt"], 0.033)
        self.assertEqual(cfg["robot"]["state_feature_names"], list(STATE_NAMES))
        self.assertEqual(cfg["robot"]["vla_action_dim"], len(ACTION_NAMES))
        self.assertEqual(
            [group["name"] for group in cfg["robot"]["joint_groups"]],
            ["right_arm", "right_gripper", "left_arm", "left_gripper", "leg", "head"],
        )
        self.assertEqual(
            set(key.rsplit(".", 1)[-1] for key, feature in cfg["observation_features"].items()
                if feature.get("dtype") == "image"),
            set(CAMERA_KEYS),
        )
        self.assertTrue(cfg["force_setup_policy"])
        self.assertTrue(cfg["robot"]["control"]["interpolation"]["enabled"])
        self.assertEqual(cfg["server"]["port"], 8006)

    def test_cameras_override_rebuilds_features(self):
        cfg = apply_vla_defaults({
            "observation_features": make_observation_features(CAMERA_KEYS),
            "cameras": list(HARDWARE_CAMERA_KEYS),
        })
        self.assertIn("observation.images.head_left", cfg["observation_features"])
        self.assertEqual(cfg["observation_features"]["observation.state"]["shape"], [23])

    def test_yaml_overrides_are_kept(self):
        cfg = apply_vla_defaults({"fps": 15, "device": "cpu", "actions_per_chunk": 10})
        self.assertEqual(cfg["fps"], 15)
        self.assertEqual(cfg["device"], "cpu")
        self.assertEqual(cfg["actions_per_chunk"], 10)


if __name__ == "__main__":
    unittest.main()
