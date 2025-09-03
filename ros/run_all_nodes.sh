# #!/bin/bash

cd /workspace/ros
colcon build
source install/setup.bash
ros2 launch vla_auto_recover all_nodes.launch.py