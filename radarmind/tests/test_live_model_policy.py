from __future__ import annotations

import unittest

from radarmind.agent.carla_dynamic_demo import radar_corridor_summary, select_closed_loop_action
from radarmind.agent.live_model_policy import conservative_fusion


class LiveModelPolicyTest(unittest.TestCase):
    def test_hybrid_uses_more_conservative_model_action(self) -> None:
        result = conservative_fusion(
            {"type": "slow_down", "reason": "vulnerable road user"},
            {"type": "keep_speed", "reason": "safe TTC"},
        )
        self.assertEqual(result["type"], "slow_down")
        self.assertEqual(result["source"], "radarmind")

    def test_hybrid_safety_shield_overrides_model(self) -> None:
        result = conservative_fusion(
            {"type": "keep_speed", "reason": "model clear"},
            {"type": "brake", "reason": "short headway"},
        )
        self.assertEqual(result["type"], "brake")
        self.assertEqual(result["source"], "safety_shield")

    def test_radarmind_mode_has_emergency_override(self) -> None:
        result = select_closed_loop_action(
            "radarmind",
            {"fresh": True, "action": {"type": "keep_speed", "reason": "model clear"}},
            {"type": "emergency_brake", "reason": "TTC below limit"},
        )
        self.assertEqual(result["type"], "emergency_brake")
        self.assertEqual(result["source"], "emergency_safety_override")

    def test_radar_height_includes_sensor_pitch(self) -> None:
        points = [
            {
                "depth": 20.0,
                "azimuth": 0.0,
                "altitude": -0.034906585,
                "velocity": 0.0,
            }
        ]
        summary = radar_corridor_summary(
            points,
            corridor_half_width=2.4,
            radar_height=1.0,
            radar_pitch_deg=2.0,
            ground_clearance=0.35,
        )
        self.assertEqual(summary["corridor_points"], 1)
        self.assertEqual(summary["ground_filtered_points"], 0)
        self.assertAlmostEqual(summary["nearest_detection"]["estimated_height_m"], 1.0, places=2)

    def test_stale_model_falls_back_to_safety(self) -> None:
        result = select_closed_loop_action(
            "hybrid",
            {"fresh": False, "action": {"type": "slow_down", "reason": "old"}},
            {"type": "keep_speed", "reason": "safe"},
        )
        self.assertEqual(result["type"], "keep_speed")
        self.assertEqual(result["source"], "safety_fallback")


if __name__ == "__main__":
    unittest.main()
