from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DynamicStartupTests(unittest.TestCase):
    def test_action_never_uses_paused_world_steps(self):
        source = (ROOT / "actions" / "grasp_cube.py").read_text(encoding="utf-8")
        self.assertNotIn('f"multi_step:', source)
        self.assertNotIn("self._step(", source)

    def test_gazebo_starts_running(self):
        source = (
            ROOT
            / "src"
            / "robomaster_pick_place_sim"
            / "launch"
            / "static_model.launch.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"ign", "gazebo",\n                "-r",', source)

    def test_objects_spawn_after_controller_ready_barrier(self):
        source = (ROOT / "run_vision_sorting.sh").read_text(encoding="utf-8")
        action = source.index('actions/grasp_bottle_tennis.py')
        barrier = source.index('until [[ -f "${CONTROLLER_READY_FILE}" ]]')
        scene = source.index('actions/spawn_sorting_scene.py')
        self.assertLess(action, barrier)
        self.assertLess(barrier, scene)
        self.assertIn('touch "${SCENE_READY_FILE}"', source[scene:])


if __name__ == "__main__":
    unittest.main()
