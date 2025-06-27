from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # GR00T Controller Node
        Node(
            package='vla_auto_recover',
            executable='gr00t_controller_node',
            name='gr00t_controller',
            output='screen'
        ),
        
        # State Manager Node
        Node(
            package='vla_auto_recover',
            executable='state_manager_node',
            name='state_manager',
            output='screen'
        ),
        
        # VLM Watcher Node
        Node(
            package='vla_auto_recover',
            executable='vlm_watcher_node',
            name='vlm_watcher',
            output='screen'
        ),
    ])