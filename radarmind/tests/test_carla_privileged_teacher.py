import unittest
from types import SimpleNamespace

from radarmind.agent.carla_privileged_teacher import build_privileged_teacher, project_actor
from radarmind.datasets.build_carla_fusion_sft import balance_records


def vector(x=0.0, y=0.0, z=0.0): return SimpleNamespace(x=x, y=y, z=z)


class FakeActor:
    def __init__(self, actor_id, x, y, vx=0.0, role="radarmind_npc", type_id="vehicle.test"):
        self.id, self.is_alive, self.type_id = actor_id, True, type_id
        self.attributes = {"role_name": role}
        self._transform = SimpleNamespace(location=vector(x, y, 0.0), rotation=SimpleNamespace(yaw=0.0))
        self._velocity = vector(vx, 0.0, 0.0)
    def get_transform(self): return self._transform
    def get_velocity(self): return self._velocity


class CarlaPrivilegedTeacherTest(unittest.TestCase):
    def setUp(self): self.ego = FakeActor(1, 0.0, 0.0, vx=10.0, role="radarmind_ego")
    def test_projection_and_ttc(self):
        projected = project_actor(self.ego, FakeActor(2, 20.0, 1.0, vx=5.0))
        self.assertIsNotNone(projected); self.assertTrue(projected["in_path"])
        self.assertAlmostEqual(projected["closing_speed_mps"], 5.0, places=1)
        self.assertAlmostEqual(projected["ttc_s"], 4.0, places=1)
    def test_behind_actor_is_excluded(self):
        self.assertIsNone(project_actor(self.ego, FakeActor(2, -5.0, 0.0)))
    def test_balanced_records_oversamples_actions_equally(self):
        rows = []
        for action, count in (("keep_speed", 3), ("brake", 1)):
            for index in range(count):
                rows.append({"sample_id": f"{action}-{index}", "radar_scene": {"recommended_action": {"type": action}}})
        balanced = balance_records(rows, seed=7)
        counts = {action: sum(row["radar_scene"]["recommended_action"]["type"] == action for row in balanced)
                  for action in ("keep_speed", "brake")}
        self.assertEqual(counts, {"keep_speed": 3, "brake": 3})

    def test_vru_semantics_raise_action_severity(self):
        pedestrian = FakeActor(3, 12.0, 0.5, role="radarmind_pedestrian", type_id="walker.pedestrian.0001")
        teacher = build_privileged_teacher(self.ego, [pedestrian], {"type": "keep_speed", "reason": "clear"})
        self.assertEqual(teacher["objects"][0]["class"], "pedestrian")
        self.assertEqual(teacher["recommended_action"]["type"], "emergency_brake")
        self.assertFalse(teacher["privileged_input_online"])


if __name__ == "__main__": unittest.main()
