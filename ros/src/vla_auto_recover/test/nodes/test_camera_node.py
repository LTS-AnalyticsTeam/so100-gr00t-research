import pytest
import rclpy
import launch
import launch_pytest
import time
import cv2
import numpy as np
from unittest.mock import patch, MagicMock, Mock
from launch_ros.actions import Node
from vla_auto_recover.camera_node import CameraNode
from vla_interfaces.msg import ImagePair
from sensor_msgs.msg import Image
from rclpy.parameter import Parameter


# ============ fixture ============
@pytest.fixture
def mock_camera_node():
    """Fixture to create and yield the CameraNode with mocked cameras."""
    if not rclpy.ok():
        rclpy.init()
    
    with patch('cv2.VideoCapture') as mock_video_capture:
        # Mock camera initialization
        mock_cap_center = MagicMock()
        mock_cap_right = MagicMock()
        
        # Configure mocks to simulate successful camera initialization
        mock_cap_center.isOpened.return_value = True
        mock_cap_center.get.return_value = 30.0  # FPS
        mock_cap_center.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        
        mock_cap_right.isOpened.return_value = True
        mock_cap_right.get.return_value = 30.0  # FPS
        mock_cap_right.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        
        # Return different mock objects for different camera IDs
        def side_effect(camera_id):
            if camera_id == 0:
                return mock_cap_center
            elif camera_id == 2:
                return mock_cap_right
            return MagicMock()
        
        mock_video_capture.side_effect = side_effect
        
        node = CameraNode()
        # Wait a bit for initialization
        time.sleep(0.1)
        
        yield node, mock_cap_center, mock_cap_right
        
        try:
            node.destroy_node()
        except Exception:
            pass
    
    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass


@pytest.fixture
def sample_frame():
    """Create a sample camera frame for testing."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :] = [100, 150, 200]  # BGR color
    return frame


@launch_pytest.fixture
def launch_description():
    """Launch description for camera node."""
    camera_node = Node(
        package="vla_auto_recover",
        executable="camera_node",
        name="camera",
        output="screen",
    )

    return launch.LaunchDescription(
        [
            camera_node,
            launch_pytest.actions.ReadyToTest(),
        ]
    )


# ============ Pythonクラスの実行による（ホワイトボックステスト） ============
def test_camera_node_initialization(mock_camera_node):
    """Test that the CameraNode initializes correctly."""
    node, mock_cap_center, mock_cap_right = mock_camera_node
    
    assert node.get_name() == "camera"
    assert node.image_vlm_pub is not None
    assert node.image_vla_pub is not None
    assert node.timer_vlm_camera is not None
    assert node.timer_vla_camera is not None
    assert node.frame_lock is not None
    assert hasattr(node, 'cameras')
    assert hasattr(node, 'capture_running')
    assert hasattr(node, 'capture_thread')


def test_camera_initialization_success(mock_camera_node):
    """Test successful camera initialization."""
    node, mock_cap_center, mock_cap_right = mock_camera_node
    
    # Verify cameras were configured
    mock_cap_center.set.assert_called()
    mock_cap_right.set.assert_called()
    
    # Check that cameras are stored
    assert 'center_cam' in node.cameras
    assert 'right_cam' in node.cameras


@patch('cv2.VideoCapture')
def test_camera_initialization_failure(mock_video_capture):
    """Test camera initialization failure."""
    # Mock failed camera initialization
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    mock_video_capture.return_value = mock_cap
    
    if not rclpy.ok():
        rclpy.init()
    
    try:
        node = CameraNode()
        # The node should handle initialization failure gracefully
        assert hasattr(node, 'cameras')
    finally:
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


def test_get_latest_image_pair_success(mock_camera_node, sample_frame):
    """Test successful image pair retrieval."""
    node, mock_cap_center, mock_cap_right = mock_camera_node
    
    # Set up frames in the node
    with node.frame_lock:
        node.latest_center_frame = sample_frame.copy()
        node.latest_right_frame = sample_frame.copy()
    
    center_img, right_img = node._get_latest_image_pair()
    
    assert center_img is not None
    assert right_img is not None
    assert center_img.shape == (480, 640, 3)
    assert right_img.shape == (480, 640, 3)
    assert np.array_equal(center_img, sample_frame)
    assert np.array_equal(right_img, sample_frame)


def test_get_latest_image_pair_no_frames(mock_camera_node):
    """Test image pair retrieval when no frames are available."""
    node, mock_cap_center, mock_cap_right = mock_camera_node
    
    # Ensure no frames are available
    with node.frame_lock:
        node.latest_center_frame = None
        node.latest_right_frame = None
    
    center_img, right_img = node._get_latest_image_pair()
    
    assert center_img is None
    assert right_img is None


def test_cb_publish_image_vlm_success(mock_camera_node, sample_frame):
    """Test successful VLM image publishing."""
    node, mock_cap_center, mock_cap_right = mock_camera_node
    
    # Mock the publisher
    node.image_vlm_pub.publish = MagicMock()
    
    # Set up frames
    with node.frame_lock:
        node.latest_center_frame = sample_frame.copy()
        node.latest_right_frame = sample_frame.copy()
    
    node._cb_publish_image_vlm()
    
    # Verify that publish was called
    node.image_vlm_pub.publish.assert_called_once()
    
    # Get the published message
    call_args = node.image_vlm_pub.publish.call_args[0]
    published_msg = call_args[0]
    
    assert isinstance(published_msg, ImagePair)
    assert isinstance(published_msg.center_cam, Image)
    assert isinstance(published_msg.right_cam, Image)
    assert published_msg.center_cam.height == 480
    assert published_msg.center_cam.width == 640
    assert published_msg.center_cam.encoding == "bgr8"


def test_cb_publish_image_vla_success(mock_camera_node, sample_frame):
    """Test successful VLA image publishing."""
    node, mock_cap_center, mock_cap_right = mock_camera_node
    
    # Mock the publisher
    node.image_vla_pub.publish = MagicMock()
    
    # Set up frames
    with node.frame_lock:
        node.latest_center_frame = sample_frame.copy()
        node.latest_right_frame = sample_frame.copy()
    
    node._cb_publish_image_vla()
    
    # Verify that publish was called
    node.image_vla_pub.publish.assert_called_once()
    
    # Get the published message
    call_args = node.image_vla_pub.publish.call_args[0]
    published_msg = call_args[0]
    
    assert isinstance(published_msg, ImagePair)
    assert isinstance(published_msg.center_cam, Image)
    assert isinstance(published_msg.right_cam, Image)


def test_cb_publish_image_vlm_no_images(mock_camera_node):
    """Test VLM image publishing when no images are available."""
    node, mock_cap_center, mock_cap_right = mock_camera_node
    
    # Mock the publisher
    node.image_vlm_pub.publish = MagicMock()
    
    # Ensure no frames are available
    with node.frame_lock:
        node.latest_center_frame = None
        node.latest_right_frame = None
    
    node._cb_publish_image_vlm()
    
    # Verify that publish was not called when no images are available
    node.image_vlm_pub.publish.assert_not_called()


def test_cb_publish_image_vla_no_images(mock_camera_node):
    """Test VLA image publishing when no images are available."""
    node, mock_cap_center, mock_cap_right = mock_camera_node
    
    # Mock the publisher
    node.image_vla_pub.publish = MagicMock()
    
    # Ensure no frames are available
    with node.frame_lock:
        node.latest_center_frame = None
        node.latest_right_frame = None
    
    node._cb_publish_image_vla()
    
    # Verify that publish was not called when no images are available
    node.image_vla_pub.publish.assert_not_called()


def test_thread_safety(mock_camera_node, sample_frame):
    """Test thread safety of image retrieval."""
    import threading
    
    node, mock_cap_center, mock_cap_right = mock_camera_node
    results = []
    
    def worker():
        # Set up frames
        with node.frame_lock:
            node.latest_center_frame = sample_frame.copy()
            node.latest_right_frame = sample_frame.copy()
        
        result = node._get_latest_image_pair()
        results.append(result)
    
    # Run multiple threads simultaneously
    threads = []
    for _ in range(5):
        thread = threading.Thread(target=worker)
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # All results should be valid
    for center_img, right_img in results:
        if center_img is not None and right_img is not None:
            assert center_img.shape == (480, 640, 3)
            assert right_img.shape == (480, 640, 3)


def test_capture_loop_functionality(mock_camera_node, sample_frame):
    """Test camera capture loop functionality."""
    node, mock_cap_center, mock_cap_right = mock_camera_node
    
    # Configure mock cameras to return sample frames
    mock_cap_center.read.return_value = (True, sample_frame.copy())
    mock_cap_right.read.return_value = (True, sample_frame.copy())
    
    # Let the capture loop run for a short time
    time.sleep(0.2)
    
    # Check that frames are being captured
    with node.frame_lock:
        assert node.latest_center_frame is not None
        assert node.latest_right_frame is not None


def test_camera_node_destroy(mock_camera_node):
    """Test proper cleanup in destroy_node."""
    node, mock_cap_center, mock_cap_right = mock_camera_node
    
    # Verify cameras are released during cleanup
    node.destroy_node()
    
    assert not node.capture_running
    mock_cap_center.release.assert_called()
    mock_cap_right.release.assert_called()


# ============ エラーハンドリングのテスト ============
def test_cb_publish_image_vlm_exception_handling(mock_camera_node):
    """Test exception handling in VLM image publishing."""
    node, mock_cap_center, mock_cap_right = mock_camera_node
    
    node.image_vlm_pub.publish = MagicMock()
    
    with patch.object(node, '_get_latest_image_pair', side_effect=Exception("Test exception")):
        # Should not raise an exception
        node._cb_publish_image_vlm()
        
        # Verify that publish was not called due to exception
        node.image_vlm_pub.publish.assert_not_called()


def test_cb_publish_image_vla_exception_handling(mock_camera_node):
    """Test exception handling in VLA image publishing."""
    node, mock_cap_center, mock_cap_right = mock_camera_node
    
    node.image_vla_pub.publish = MagicMock()
    
    with patch.object(node, '_get_latest_image_pair', side_effect=Exception("Test exception")):
        # Should not raise an exception
        node._cb_publish_image_vla()
        
        # Verify that publish was not called due to exception
        node.image_vla_pub.publish.assert_not_called()


def test_capture_loop_camera_failure(mock_camera_node):
    """Test capture loop handling camera read failures."""
    node, mock_cap_center, mock_cap_right = mock_camera_node
    
    # Configure cameras to fail reading
    mock_cap_center.read.return_value = (False, None)
    mock_cap_right.read.return_value = (False, None)
    
    # Let the capture loop run with failures
    time.sleep(0.1)
    
    # The loop should handle failures gracefully without crashing
    assert node.capture_running


def test_parameter_configuration():
    """Test camera node parameter configuration."""
    if not rclpy.ok():
        rclpy.init()
    
    try:
        with patch('cv2.VideoCapture') as mock_video_capture:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 30.0
            mock_video_capture.return_value = mock_cap
            
            node = CameraNode()
            
            # Test default parameters
            assert node.get_parameter("center_cam_id").value == 0
            assert node.get_parameter("right_cam_id").value == 2
            assert node.get_parameter("fps").value == 30
            
            node.destroy_node()
    finally:
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass