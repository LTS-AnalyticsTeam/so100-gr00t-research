import pytest
import rclpy
import launch
import launch_pytest
import time
import os
import tempfile
import shutil
import cv2
import numpy as np
from unittest.mock import patch, MagicMock
from launch_ros.actions import Node
from vla_auto_recover.camera_node import CameraNode
from vla_interfaces.msg import ImagePair
from sensor_msgs.msg import Image
from rclpy.parameter import Parameter


# ============ fixture ============
@pytest.fixture
def camera_node():
    """Fixture to create and yield the CameraNode."""
    rclpy.init()
    node = CameraNode()
    yield node
    node.destroy_node()
    rclpy.shutdown()


@pytest.fixture
def temp_camera_dir():
    """Fixture to create temporary camera directories with sample images."""
    temp_dir = tempfile.mkdtemp()
    center_cam_dir = os.path.join(temp_dir, "center_cam")
    right_cam_dir = os.path.join(temp_dir, "right_cam")
    
    os.makedirs(center_cam_dir, exist_ok=True)
    os.makedirs(right_cam_dir, exist_ok=True)
    
    # Create sample images
    sample_image = np.zeros((480, 640, 3), dtype=np.uint8)
    sample_image[:, :] = [100, 150, 200]  # BGR color
    
    # Save sample images with timestamp-like names
    cv2.imwrite(os.path.join(center_cam_dir, "20231201_120000_000.jpg"), sample_image)
    cv2.imwrite(os.path.join(center_cam_dir, "20231201_120001_000.jpg"), sample_image)
    cv2.imwrite(os.path.join(right_cam_dir, "20231201_120000_000.jpg"), sample_image)
    cv2.imwrite(os.path.join(right_cam_dir, "20231201_120001_000.jpg"), sample_image)
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


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
def test_camera_node_initialization(camera_node: CameraNode):
    """Test that the CameraNode initializes correctly."""
    assert camera_node.get_name() == "camera"
    assert camera_node.image_vlm_pub is not None
    assert camera_node.image_vla_pub is not None
    assert camera_node.timer_vlm_camera is not None
    assert camera_node.timer_vla_camera is not None
    assert camera_node.file_lock is not None


@patch('vla_auto_recover.camera_node.CAMRERA_DIR')
def test_get_latest_image_pair_success(mock_camera_dir, camera_node: CameraNode, temp_camera_dir):
    """Test successful image pair retrieval."""
    mock_camera_dir.__str__ = lambda: temp_camera_dir
    mock_camera_dir.__fspath__ = lambda: temp_camera_dir
    
    with patch('vla_auto_recover.camera_node.CAMRERA_DIR', temp_camera_dir):
        center_img, right_img = camera_node._get_latest_image_pair()
        
        assert center_img is not None
        assert right_img is not None
        assert center_img.shape == (480, 640, 3)
        assert right_img.shape == (480, 640, 3)


@patch('vla_auto_recover.camera_node.CAMRERA_DIR')
def test_get_latest_image_pair_no_directories(mock_camera_dir, camera_node: CameraNode):
    """Test image pair retrieval when directories don't exist."""
    mock_camera_dir.__str__ = lambda: "/nonexistent/path"
    mock_camera_dir.__fspath__ = lambda: "/nonexistent/path"
    
    with patch('vla_auto_recover.camera_node.CAMRERA_DIR', "/nonexistent/path"):
        center_img, right_img = camera_node._get_latest_image_pair()
        
        assert center_img is None
        assert right_img is None


def test_get_latest_image_pair_no_files(camera_node: CameraNode):
    """Test image pair retrieval when no image files exist."""
    temp_dir = tempfile.mkdtemp()
    center_cam_dir = os.path.join(temp_dir, "center_cam")
    right_cam_dir = os.path.join(temp_dir, "right_cam")
    
    os.makedirs(center_cam_dir, exist_ok=True)
    os.makedirs(right_cam_dir, exist_ok=True)
    
    try:
        with patch('vla_auto_recover.camera_node.CAMRERA_DIR', temp_dir):
            center_img, right_img = camera_node._get_latest_image_pair()
            
            assert center_img is None
            assert right_img is None
    finally:
        shutil.rmtree(temp_dir)


@patch('vla_auto_recover.camera_node.CAMRERA_DIR')
def test_cb_publish_image_vlm_success(mock_camera_dir, camera_node: CameraNode, temp_camera_dir):
    """Test successful VLM image publishing."""
    mock_camera_dir.__str__ = lambda: temp_camera_dir
    mock_camera_dir.__fspath__ = lambda: temp_camera_dir
    
    # Mock the publisher
    camera_node.image_vlm_pub.publish = MagicMock()
    
    with patch('vla_auto_recover.camera_node.CAMRERA_DIR', temp_camera_dir):
        camera_node._cb_publish_image_vlm()
        
        # Verify that publish was called
        camera_node.image_vlm_pub.publish.assert_called_once()
        
        # Get the published message
        call_args = camera_node.image_vlm_pub.publish.call_args[0]
        published_msg = call_args[0]
        
        assert isinstance(published_msg, ImagePair)
        assert isinstance(published_msg.center_cam, Image)
        assert isinstance(published_msg.right_cam, Image)


@patch('vla_auto_recover.camera_node.CAMRERA_DIR')
def test_cb_publish_image_vla_success(mock_camera_dir, camera_node: CameraNode, temp_camera_dir):
    """Test successful VLA image publishing."""
    mock_camera_dir.__str__ = lambda: temp_camera_dir
    mock_camera_dir.__fspath__ = lambda: temp_camera_dir
    
    # Mock the publisher
    camera_node.image_vla_pub.publish = MagicMock()
    
    with patch('vla_auto_recover.camera_node.CAMRERA_DIR', temp_camera_dir):
        camera_node._cb_publish_image_vla()
        
        # Verify that publish was called
        camera_node.image_vla_pub.publish.assert_called_once()
        
        # Get the published message
        call_args = camera_node.image_vla_pub.publish.call_args[0]
        published_msg = call_args[0]
        
        assert isinstance(published_msg, ImagePair)
        assert isinstance(published_msg.center_cam, Image)
        assert isinstance(published_msg.right_cam, Image)


def test_cb_publish_image_vlm_no_images(camera_node: CameraNode):
    """Test VLM image publishing when no images are available."""
    # Mock the publisher
    camera_node.image_vlm_pub.publish = MagicMock()
    
    with patch.object(camera_node, '_get_latest_image_pair', return_value=(None, None)):
        camera_node._cb_publish_image_vlm()
        
        # Verify that publish was not called when no images are available
        camera_node.image_vlm_pub.publish.assert_not_called()


def test_cb_publish_image_vla_no_images(camera_node: CameraNode):
    """Test VLA image publishing when no images are available."""
    # Mock the publisher
    camera_node.image_vla_pub.publish = MagicMock()
    
    with patch.object(camera_node, '_get_latest_image_pair', return_value=(None, None)):
        camera_node._cb_publish_image_vla()
        
        # Verify that publish was not called when no images are available
        camera_node.image_vla_pub.publish.assert_not_called()


def test_thread_safety(camera_node: CameraNode, temp_camera_dir):
    """Test thread safety of image retrieval."""
    import threading
    import time
    
    results = []
    
    def worker():
        with patch('vla_auto_recover.camera_node.CAMRERA_DIR', temp_camera_dir):
            result = camera_node._get_latest_image_pair()
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
    
    # All results should be valid (not None, None)
    for center_img, right_img in results:
        if center_img is not None and right_img is not None:
            assert center_img.shape == (480, 640, 3)
            assert right_img.shape == (480, 640, 3)


# ============ エラーハンドリングのテスト ============
def test_cb_publish_image_vlm_exception_handling(camera_node: CameraNode):
    """Test exception handling in VLM image publishing."""
    camera_node.image_vlm_pub.publish = MagicMock()
    
    with patch.object(camera_node, '_get_latest_image_pair', side_effect=Exception("Test exception")):
        # Should not raise an exception
        camera_node._cb_publish_image_vlm()
        
        # Verify that publish was not called due to exception
        camera_node.image_vlm_pub.publish.assert_not_called()


def test_cb_publish_image_vla_exception_handling(camera_node: CameraNode):
    """Test exception handling in VLA image publishing."""
    camera_node.image_vla_pub.publish = MagicMock()
    
    with patch.object(camera_node, '_get_latest_image_pair', side_effect=Exception("Test exception")):
        # Should not raise an exception
        camera_node._cb_publish_image_vla()
        
        # Verify that publish was not called due to exception
        camera_node.image_vla_pub.publish.assert_not_called()