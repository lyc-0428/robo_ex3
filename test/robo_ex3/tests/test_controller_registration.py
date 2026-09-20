"""Exercise parameter-service failure paths without a ROS installation."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


SOURCE = Path(__file__).resolve().parents[1] / "actions" / "grasp_cube.py"


class ControllerRegistrationTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == "GraspCubeAction")
        method = next(n for n in cls.body
                      if getattr(n, "name", "") == "_register_runtime_controller_type")
        self.clock = Mock()
        self.clock.monotonic.return_value = 0.0
        self.parameter = Mock()
        scope = dict(
            re=__import__("re"), time=self.clock,
            rclpy=SimpleNamespace(ok=lambda: True, spin_once=Mock()),
            Parameter=self.parameter,
            SetParameters=SimpleNamespace(Request=SimpleNamespace),
            FORWARD_CONTROLLER_TYPE="forward_command_controller/ForwardCommandController",
            GraspActionError=RuntimeError,
        )
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), scope)
        self.register = scope[method.name]
        self.node = Mock(controller_timeout=45.0)
        self.node._run.return_value.stdout = scope["FORWARD_CONTROLLER_TYPE"]
        self.client = self.node.create_client.return_value
        self.client.wait_for_service.return_value = True
        self.future = self.client.call_async.return_value
        self.future.done.return_value = True
        self.future.result.return_value = SimpleNamespace(
            results=[SimpleNamespace(successful=True, reason="")])

    def test_success_uses_configured_timeout_and_typed_parameter(self):
        self.register(self.node, "hold")
        self.client.wait_for_service.assert_called_once_with(timeout_sec=45.0)
        self.parameter.assert_called_once_with(
            "hold.type", value="forward_command_controller/ForwardCommandController")
        self.node.destroy_client.assert_called_once_with(self.client)

    def test_unavailable_service_does_not_send_request(self):
        self.client.wait_for_service.return_value = False
        with self.assertRaisesRegex(RuntimeError, "不可用"):
            self.register(self.node, "hold")
        self.client.call_async.assert_not_called()
        self.node.destroy_client.assert_called_once_with(self.client)

    def test_request_timeout_cancels_future_and_cleans_up(self):
        self.future.done.return_value = False
        self.clock.monotonic.side_effect = [0.0, 46.0]
        with self.assertRaisesRegex(RuntimeError, "未返回"):
            self.register(self.node, "hold")
        self.future.cancel.assert_called_once()
        self.node.destroy_client.assert_called_once_with(self.client)

    def test_rejection_preserves_service_reason(self):
        self.future.result.return_value.results[0] = SimpleNamespace(
            successful=False, reason="read-only parameter")
        with self.assertRaisesRegex(RuntimeError, "read-only parameter"):
            self.register(self.node, "hold")
        self.node.destroy_client.assert_called_once_with(self.client)


if __name__ == "__main__":
    unittest.main()
