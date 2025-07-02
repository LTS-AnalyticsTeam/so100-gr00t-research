import pytest
import rclpy
import launch
import launch_pytest
import time
from launch_ros.actions import Node
from vla_auto_recover.processing.utils.get_sample_image import get_sample_image
from vla_auto_recover.processing.config.system_settings import State
from vla_auto_recover.vlm_detector_node import VLMDetectorNode
from vla_interfaces.msg import ImagePair
from std_msgs.msg import String
from rclpy.parameter import Parameter


# ============ fixture ============
@pytest.fixture
def vlm_detector_node():
    """Fixture to create and yield the VLMDetectorNode."""
    rclpy.init()
    node = VLMDetectorNode()
    node.set_parameters(
        [
            Parameter("fps", value=5.0),
            Parameter("worker_num", value=4),
        ]
    )
    yield node
    node.destroy_node()
    rclpy.shutdown()


@pytest.fixture
def image_pair_msg():
    return ImagePair(
        center_cam=get_sample_image("normal", "start", "center_cam"),
        right_cam=get_sample_image("normal", "start", "right_cam"),
    )


def test_cb_():
    """"""


def test_cb_():
    """"""


def test_cb_():
    """"""
