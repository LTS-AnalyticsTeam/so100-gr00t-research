from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1. 常時異常監視ノード
        Node(
            package='vlm_watcher',
            executable='vlm_watcher_node',
            arguments=[
                '--fps', '5.0',
                '--prompt', 'Detect semantic anomaly and select one recovery action.',
                '--action-list', '/config/action_list.jsonl'
            ],
            output='screen'
        ),

        # 2. VLA 実行ノード（gr00t_controller）は既存のまま起動
        Node(
            package='gr00t_controller',
            executable='gr00t_controller_node',
            output='screen'
        ),
    ])
