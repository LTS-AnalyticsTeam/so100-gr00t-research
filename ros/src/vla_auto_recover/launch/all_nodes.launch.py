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
                output="screen",
            ),
            # VLA Controller Node
            Node(
                package="vla_auto_recover",
                executable="vla_controller_node",
                name="vla_controller",
                output="screen",
            ),
            # VLM Monitor Node
            Node(
                package="vla_auto_recover",
                executable="vlm_monitor_node",
                name="vlm_monitor",
                output="screen",
            ),
        ]
    )
