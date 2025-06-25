
以降`/workspace`をベースディレクトリとして考える。

# Nodeの実行方法

```
# ビルド
colcon build
source install/setup.bash

# nodeの実行（1nodeずつ実行）
ros2 run vla_auto_recover gr00t_controller_node
ros2 run vla_auto_recover vlm_watcher_node
ros2 run vla_auto_recover state_manager_node

# nodeの実行（複数nodeを同時に実行）
ros2 launch vla_auto_recover all_nodes.launch.py
```



