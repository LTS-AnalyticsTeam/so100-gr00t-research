import pytest
import rclpy
import launch
import launch_pytest
from launch_ros.actions import Node
from sensor_msgs.msg import Image
from vla_auto_recover.processing.utils.get_sample_image import get_sample_image


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


@pytest.mark.launch(fixture=launch_description)
def test_all_nodes_startup(setup, launch_context):
    """Test that all nodes start up successfully."""

    # sbscribe to the vlm_detector node
    rclpy.init()
    node = rclpy.create_node("test_injector_node")

    # image publisher
    image_vlm_center_pub = node.create_publisher("/image/vlm/center")
    image_vlm_right_pub = node.create_publisher("/image/vlm/right")

    image_vlm_center_pub.publish(get_sample_image("normal", "start", "center"))
    image_vlm_right_pub.publish(get_sample_image("normal", "start", "right"))

    # state_change publisher
    state_change_pub = node.create_publisher("/state_change")
    state_change_pub.publish()

    import time

    time.sleep(1)


@pytest.mark.launch(fixture=launch_description, shutdown=True)
def test_all_nodes_shutdown(setup, launch_context):
    """Test that all nodes shut down successfully."""
    import time

    time.sleep(1)


# tests/test_internal.py
import rclpy
from vla_auto_recover.vlm_detector_node import VLMDetectorNode


@pytest.fixture
def vlm_detector_node():
    rclpy.init()
    node = VLMDetectorNode()  # 直接生成
    yield node
    node.destroy_node()
    rclpy.shutdown()


def test_node_image_sub_cb(vlm_detector_node):
    test_node = rclpy.create_node("test_node")
    image_vlm_center_pub = test_node.create_publisher("/image/vlm/center")
    image_vlm_right_pub = test_node.create_publisher("/image/vlm/right")

    # Test the callback function with a sample image
    center_image = get_sample_image("normal", "start", "center")
    right_image = get_sample_image("normal", "start", "right")

    # Check if the state is still RUNNING after processing the images
    assert node.state == "RUNNING", f"Expected state RUNNING, got {node.state}"
