import rclpy
import pytest
from rclpy.parameter import Parameter
from vla_auto_recover.state_manager_node import StateManagerNode
from vla_auto_recover.processing.config.system_settings import State
from vla_interfaces.msg import DetectionOutput


# ============ fixture ============
@pytest.fixture
def state_manager_node():
    """Fixture to create and yield the StateManagerNode."""
    rclpy.init()
    node = StateManagerNode()
    node.set_parameters([])
    yield node
    node.destroy_node()
    rclpy.shutdown()


def test_cb_state_transition(state_manager_node: StateManagerNode):
    """Test the state transition callback."""
    import time

    assert state_manager_node.state_manager.state == State.RUNNING.value

    # タイムスタンプを含むDetectionOutputメッセージを作成
    current_timestamp = int(time.time_ns())

    state_manager_node._cb_state_transition(
        DetectionOutput(
            detection_result="ANOMALY",
            action_id=1,
            reason="赤のお皿の上に青いお皿が乗っている",
            timestamp=current_timestamp,
        )
    )
    assert state_manager_node.state_manager.state == State.RECOVERY.value
    assert state_manager_node.latest_update_timestamp == current_timestamp
