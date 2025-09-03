

# GR00T Serverの立ち上げ
Isaac-GR00Tのリポジトリからサーバーを立ち上げる。
https://github.com/NVIDIA/Isaac-GR00T

# Dockerによる環境構築
```
docker-compose build vla_auto_recover-cli
docker-compose up -d vla_auto_recover-cli
```
以降、Dockerコンテナ内で作業を行う。

# Lerobotのインストール
lerobotの`1ee2ca5c2627eab05940452472d876d0d4e73d1f`のコミットを用いて実装を行った。
`pip install -e .`と`-e`オプションをつけると、ROSをビルドするときにlerobotのパッケージが認識されないので注意。

```
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout 1ee2ca5c2627eab05940452472d876d0d4e73d1f
pip install .
```

# ROS2の構成
Nodeの構成とTopic通信は以下の通り。
![Node構成](docs/nodes_graph.svg)


VLMで状態を観測し、観測した状態をStateManagerNodeで状態を管理している。
<!-- ![状態遷移図](docs/system_state.svg) -->
<img src="docs/system_state.svg" alt="状態遷移図" width="480">

# ROS2の立ち上げ
`.env` ファイル作成
```
cat > /workspace/ros/config/.env << 'EOF'
# Azure OpenAI設定
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
AZURE_OPENAI_API_VERSION=2024-02-15-preview
USE_AZURE_OPENAI=true
EOF
```

ROS2のNode立ち上げ
```
cd ros
bash run_all_nodes.sh
```

各Nodeのログを確認
```
python ros/log_script/sub_rosout.py --node camera
python ros/log_script/sub_rosout.py --node vlm_detector
python ros/log_script/sub_rosout.py --node state_manager
python ros/log_script/sub_rosout.py --node vla_controller
```



