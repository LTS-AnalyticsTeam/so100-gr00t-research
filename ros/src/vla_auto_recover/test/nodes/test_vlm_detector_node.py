import pytest
import rclpy
import launch
import launch_pytest
import time
from unittest.mock import patch, MagicMock
from launch_ros.actions import Node
from vla_auto_recover.processing.utils.get_sample_image import get_sample_image
from vla_auto_recover.processing.config.system_settings import State, CB_OutputIF, ADR
from vla_auto_recover.vlm_detector_node import VLMDetectorNode
from vla_interfaces.msg import ImagePair
from std_msgs.msg import String
from rclpy.parameter import Parameter


# ============ fixture ============
@pytest.fixture
def vlm_detector_node():
    """Fixture to create and yield the VLMDetectorNode."""
    # Check if rclpy is already initialized
    if not rclpy.ok():
        rclpy.init()
    
    with patch('vla_auto_recover.vlm_detector_node.VLMDetector') as mock_vlm_detector_class:
        # Mock the VLMDetector class
        mock_vlm_detector = MagicMock()
        mock_vlm_detector_class.return_value = mock_vlm_detector
        
        node = VLMDetectorNode()
        node.set_parameters(
            [
                Parameter("fps", value=5.0),
                Parameter("worker_num", value=4),
            ]
        )
        yield node
        
        try:
            node.destroy_node()
        except Exception:
            pass  # Ignore cleanup errors in tests
    
    # Only shutdown if we initialized
    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass


@launch_pytest.fixture
def launch_description():
    """Launch description for all nodes."""
    vlm_detector = Node(
        package="vla_auto_recover",
        executable="vlm_detector_node",
        name="vlm_detector",
        output="screen",
    )

    return launch.LaunchDescription(
        [
            vlm_detector,
            launch_pytest.actions.ReadyToTest(),
        ]
    )


@pytest.fixture
def image_pair_msg():
    return ImagePair(
        center_cam=get_sample_image("normal", "start", "center_cam"),
        right_cam=get_sample_image("normal", "start", "right_cam"),
    )


# ============ Pythonクラスの実行による（ホワイトボックステスト） ============
def test_cb_put_queue(vlm_detector_node: VLMDetectorNode, image_pair_msg: ImagePair):
    """Test that the VLMDetectorNode initializes correctly."""
    vlm_detector_node._cb_sub_put_queue(image_pair_msg)
    assert image_pair_msg == vlm_detector_node.q_image_pair.get_nowait()


def test_cb_change_state(vlm_detector_node):
    """Test the _cb_change_state callback."""
    vlm_detector_node._cb_sub_change_state(String(data="RECOVERY"))
    assert vlm_detector_node.state == State.RECOVERY


def test_detection_worker(vlm_detector_node: VLMDetectorNode, image_pair_msg: ImagePair):
    """Test detection worker processing."""
    # Mock the VLMDetector's call_CB method to return a valid output
    mock_output = CB_OutputIF(
        detection_result=ADR.NORMAL,
        action_id=1,
        reason="test reason"
    )
    vlm_detector_node.vlm_detector.call_CB.return_value = mock_output
    
    assert vlm_detector_node.q_image_pair.empty()
    vlm_detector_node.q_image_pair.put(image_pair_msg)
    vlm_detector_node._detection_worker()
    assert vlm_detector_node.q_detection_output.qsize() == 1
