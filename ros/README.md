
以降`/workspace/ros`をベースディレクトリとして考える。

# Nodeの実行方法

```
# ビルド
colcon build
source install/setup.bash

# nodeの実行（1nodeずつ実行）
ros2 run vla_auto_recover vla_controller_node
ros2 run vla_auto_recover vlm_watcher_node
ros2 run vla_auto_recover state_manager_node

# nodeの実行（複数nodeを同時に実行）
ros2 launch vla_auto_recover all_nodes.launch.py
```


```
# .env ファイル作成
cat > /workspace/ros/config/.env << 'EOF'
# Azure OpenAI設定
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
AZURE_OPENAI_API_VERSION=2024-02-15-preview
USE_AZURE_OPENAI=true
EOF
```