from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            # Camera Node
            Node(
                package="vla_auto_recover",
                executable="camera_node",
                name="camera",
                output="own_log",
            ),
            # VLM Detector Node
            Node(
                package="vla_auto_recover",
                executable="vlm_detector_node",
                name="vlm_detector",
                output="own_log",
            ),
            # State Manager Node
            Node(
                package="vla_auto_recover",
                executable="state_manager_node",
                name="state_manager",
                output="own_log",
            ),
            # VLA Controller Node
            Node(
                package="vla_auto_recover",
                executable="vla_controller_node",
                name="vla_controller",
                output="own_log",
            ),
        ]
    )
