import pytest
import rclpy
import launch
import launch_pytest
import time
import queue
from unittest.mock import patch, MagicMock
from launch_ros.actions import Node
from vla_auto_recover.vla_controller_node import VLAControllerNode
from vla_auto_recover.processing.utils.get_sample_image import get_sample_image
from vla_interfaces.msg import ImagePair
from std_msgs.msg import Int32
from rclpy.parameter import Parameter


# ============ fixture ============
@pytest.fixture
def vla_controller_node():
    """Fixture to create and yield the VLAControllerNode."""
    # Check if rclpy is already initialized
    if not rclpy.ok():
        rclpy.init()
    
    with patch('vla_auto_recover.vla_controller_node.GR00TExecuter') as mock_executer:
        # Mock GR00TExecuter to avoid hardware dependencies
        mock_instance = MagicMock()
        mock_executer.return_value = mock_instance
        
        node = VLAControllerNode()
        # Initialize action_id attribute that tests expect
        node.action_id = 0
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
    """Launch description for VLA controller node."""
    vla_controller = Node(
        package="vla_auto_recover",
        executable="vla_controller_node",
        name="vla_controller",
        output="screen",
    )

    return launch.LaunchDescription(
        [
            vla_controller,
            launch_pytest.actions.ReadyToTest(),
        ]
    )


@pytest.fixture
def image_pair_msg():
    """Fixture to create ImagePair message."""
    return ImagePair(
        center_cam=get_sample_image("normal", "start", "center_cam"),
        right_cam=get_sample_image("normal", "start", "right_cam"),
    )


# ============ Pythonクラスの実行による（ホワイトボックステスト） ============
def test_vla_controller_node_initialization(vla_controller_node: VLAControllerNode):
    """Test that the VLAControllerNode initializes correctly."""
    assert vla_controller_node.get_name() == "vla_controller"
    assert vla_controller_node.gr00t_executer is not None
    assert vla_controller_node.image_pair_queue is not None
    assert vla_controller_node.image_sub is not None
    assert vla_controller_node.action_id_sub is not None
    assert vla_controller_node.timer_exec_action is not None
    
    # Check queue configuration
    assert vla_controller_node.image_pair_queue.maxsize == 1


def test_cb_timer_save_image_empty_queue(vla_controller_node: VLAControllerNode, image_pair_msg: ImagePair):
    """Test saving image to empty queue."""
    assert vla_controller_node.image_pair_queue.empty()
    
    vla_controller_node._cb_timer_save_image(image_pair_msg)
    
    assert not vla_controller_node.image_pair_queue.empty()
    assert vla_controller_node.image_pair_queue.qsize() == 1
    
    # Verify the stored message
    stored_msg = vla_controller_node.image_pair_queue.get_nowait()
    assert stored_msg == image_pair_msg


def test_cb_timer_save_image_full_queue(vla_controller_node: VLAControllerNode, image_pair_msg: ImagePair):
    """Test saving image to full queue (should replace existing image)."""
    # Fill the queue first
    first_msg = ImagePair(
        center_cam=get_sample_image("normal", "start", "center_cam"),
        right_cam=get_sample_image("normal", "start", "right_cam"),
    )
    vla_controller_node.image_pair_queue.put_nowait(first_msg)
    assert vla_controller_node.image_pair_queue.qsize() == 1
    
    # Add new message (should replace the old one)
    vla_controller_node._cb_timer_save_image(image_pair_msg)
    
    assert vla_controller_node.image_pair_queue.qsize() == 1
    
    # Verify the new message replaced the old one
    stored_msg = vla_controller_node.image_pair_queue.get_nowait()
    assert stored_msg == image_pair_msg


def test_cb_change_action(vla_controller_node: VLAControllerNode):
    """Test action change callback."""
    # Mock timer methods
    vla_controller_node.timer_exec_action.cancel = MagicMock()
    vla_controller_node.timer_exec_action.reset = MagicMock()
    
    action_msg = Int32(data=5)
    vla_controller_node._cb_change_action(action_msg)
    
    # Verify action_id is set
    assert vla_controller_node.action_id == 5
    
    # Verify timer operations
    vla_controller_node.timer_exec_action.cancel.assert_called_once()
    vla_controller_node.timer_exec_action.reset.assert_called_once()
    
    # Verify gr00t_executer.go_back_start_position was called
    vla_controller_node.gr00t_executer.go_back_start_position.assert_called_once()


def test_cb_timer_exec_action_with_image(vla_controller_node: VLAControllerNode, image_pair_msg: ImagePair):
    """Test timer execution with available image."""
    # Put image in queue
    vla_controller_node.image_pair_queue.put_nowait(image_pair_msg)
    
    with patch('vla_auto_recover.vla_controller_node.imgmsg_to_ndarray') as mock_convert:
        mock_convert.return_value = MagicMock()
        
        result = vla_controller_node._timer_exec_action()
        
        # Verify queue is empty after processing
        assert vla_controller_node.image_pair_queue.empty()
        
        # Verify gr00t_executer.act was called
        vla_controller_node.gr00t_executer.act.assert_called_once()
        
        # Verify conversion function was called for both cameras
        assert mock_convert.call_count == 2
        
        # Verify return value
        assert result is None


def test_cb_timer_exec_action_without_image(vla_controller_node: VLAControllerNode):
    """Test timer execution without available image."""
    # Ensure queue is empty
    assert vla_controller_node.image_pair_queue.empty()
    
    result = vla_controller_node._timer_exec_action()
    
    # Verify gr00t_executer.act was not called
    vla_controller_node.gr00t_executer.act.assert_not_called()
    
    # Verify return value
    assert result is None


def test_queue_behavior_maxsize(vla_controller_node: VLAControllerNode):
    """Test queue behavior with maxsize=1."""
    # Create multiple image messages
    msg1 = ImagePair(center_cam=get_sample_image("normal", "start", "center_cam"),
                     right_cam=get_sample_image("normal", "start", "right_cam"))
    msg2 = ImagePair(center_cam=get_sample_image("normal", "start", "center_cam"),
                     right_cam=get_sample_image("normal", "start", "right_cam"))
    msg3 = ImagePair(center_cam=get_sample_image("normal", "start", "center_cam"),
                     right_cam=get_sample_image("normal", "start", "right_cam"))
    
    # Add messages sequentially
    vla_controller_node._cb_timer_save_image(msg1)
    vla_controller_node._cb_timer_save_image(msg2)
    vla_controller_node._cb_timer_save_image(msg3)
    
    # Queue should only contain the latest message
    assert vla_controller_node.image_pair_queue.qsize() == 1
    stored_msg = vla_controller_node.image_pair_queue.get_nowait()
    assert stored_msg == msg3


def test_destroy_node_cleanup(vla_controller_node: VLAControllerNode):
    """Test proper cleanup in destroy_node."""
    # Mock timer methods
    vla_controller_node.timer_exec_action.cancel = MagicMock()
    
    # Mock robot connection
    vla_controller_node.gr00t_executer.robot = MagicMock()
    vla_controller_node.gr00t_executer.robot.disconnect = MagicMock()
    
    # Call destroy_node
    vla_controller_node.destroy_node()
    
    # Verify cleanup operations
    vla_controller_node.timer_exec_action.cancel.assert_called_once()
    vla_controller_node.gr00t_executer.go_back_start_position.assert_called_once()
    vla_controller_node.gr00t_executer.robot.disconnect.assert_called_once()


def test_destroy_node_cleanup_with_exceptions():
    """Test destroy_node cleanup handles exceptions gracefully."""
    # Check if rclpy is already initialized
    if not rclpy.ok():
        rclpy.init()
    
    with patch('vla_auto_recover.vla_controller_node.GR00TExecuter') as mock_executer:
        mock_instance = MagicMock()
        mock_executer.return_value = mock_instance
        
        node = VLAControllerNode()
        
        # Mock timer to raise exception
        node.timer_exec_action.cancel = MagicMock(side_effect=Exception("Timer error"))
        
        # Mock robot to raise exception
        node.gr00t_executer.robot = MagicMock()
        node.gr00t_executer.robot.disconnect = MagicMock(side_effect=Exception("Disconnect error"))
        node.gr00t_executer.go_back_start_position = MagicMock(side_effect=Exception("Position error"))
        
        # Should not raise exception despite internal errors
        node.destroy_node()
        
        # Verify cleanup attempts were made
        node.timer_exec_action.cancel.assert_called_once()
        node.gr00t_executer.go_back_start_position.assert_called_once()
        node.gr00t_executer.robot.disconnect.assert_called_once()
    
    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass


# ============ エラーハンドリングのテスト ============
def test_timer_exec_action_exception_handling(vla_controller_node: VLAControllerNode, image_pair_msg: ImagePair):
    """Test exception handling in timer execution."""
    # Put image in queue
    vla_controller_node.image_pair_queue.put_nowait(image_pair_msg)
    
    with patch('vla_auto_recover.vla_controller_node.imgmsg_to_ndarray', side_effect=Exception("Conversion error")):
        # Should not raise exception
        result = vla_controller_node._timer_exec_action()
        
        # Queue should still be processed (image removed)
        assert vla_controller_node.image_pair_queue.empty()


def test_image_conversion_and_execution(vla_controller_node: VLAControllerNode, image_pair_msg: ImagePair):
    """Test image conversion and execution flow."""
    import numpy as np
    
    # Put image in queue
    vla_controller_node.image_pair_queue.put_nowait(image_pair_msg)
    
    with patch('vla_auto_recover.vla_controller_node.imgmsg_to_ndarray') as mock_convert:
        # Mock return values for image conversion
        center_array = np.zeros((480, 640, 3), dtype=np.uint8)
        right_array = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_convert.side_effect = [center_array, right_array]
        
        vla_controller_node._timer_exec_action()
        
        # Verify gr00t_executer.act was called with correct structure
        vla_controller_node.gr00t_executer.act.assert_called_once()
        call_args = vla_controller_node.gr00t_executer.act.call_args[0][0]
        
        assert "center_cam" in call_args
        assert "right_cam" in call_args
        assert np.array_equal(call_args["center_cam"], center_array)
        assert np.array_equal(call_args["right_cam"], right_array)