# ROS2 Jazzy + lerobot VLM Node Development README

このリポジトリは、ROS2 Jazzy＋lerobot 環境上でVLM（Visual Language Model）ノードを開発・実行するための手順をまとめたものです。

---

## 前提条件

* Docker コンテナ `ros2_lerobot_dev` が起動していること
* 仮想環境を構築済み（`/opt/venv` にインストール済み lerobot 等が存在）

## コンテナに入る

```bash
# 別のターミナルから実行
docker exec -it ros2_lerobot_dev bash
```

コンテナに入ると、自動的に `/workspace` ディレクトリに移動します。

## 仮想環境の有効化

lerobot 等の Python パッケージは仮想環境 (`/opt/venv`) にインストールされています。
毎回以下を実行して有効化してください：

```bash
source /opt/venv/bin/activate
```
pip install catkin-pkgを入れ忘れ
python-dotenv　も
openai 使う場合はそれも
→ dockerfileを修正
## VLM ノード開発

`/workspace/src` 以下に `vlm_node` パッケージを配置しています。

### ビルド

```bash
# ワークスペースルート (/workspace) で
colcon build --packages-select vlm_node
```

ビルドミスった時は
rm -rf build install log

### 確認する時
パッケージ認識
ros2 pkg list | grep vlm_node

### 環境設定読み込み

```bash
source install/setup.bash
```

### VLM ノード実行

```bash
ros2 run vlm_node vlm_node
```

* 標準出力で異常検知結果が 5 秒ごとに表示されます。
* トピック `anomaly_type` に String メッセージとして結果が Publish されます。

---

### ※忘れないポイント：

* **必ず** `source /opt/venv/bin/activate` を最初に実行
* ビルド後は毎回 `source install/setup.bash` を実行

以上で VLM ノードの開発・実行が可能です。
