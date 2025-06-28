# VLA Auto-Recovery ROSプロジェクト

## プロジェクト概要

このROSプロジェクトは、Vision-Language-Action (VLA) モデルを活用した自動復旧システムです。SO100ロボットアームと組み合わせて、カメラ映像を基にした異常検知と自動復旧機能を提供します。システムは4つの主要なROS2ノードで構成されており、相互に連携して動作します。

## システム構成

### 主要ノード

#### 1. Camera Node (`/image`)
- **役割**: カメラデバイスからの映像データを取得・配信
- **機能**:
  - 実時間でのカメラ映像取得
  - VLA処理用とVLM処理用の2つの映像ストリームを生成
- **Publisher**:
  - `/image_vla`: VLA(Vision-Language-Action)処理用の映像データ
  - `/image_vlm`: VLM(Vision-Language Model)処理用の映像データ
- **メッセージ型**: `sensor_msgs/msg/Image`

#### 2. VLM Watcher Node (`/vlm_watcher`)
- **役割**: 映像解析による異常検知とシステム状態の判定
- **機能**:
  - Azure OpenAI GPT-4 Visionを使用した映像分析
  - 異常状態の検出と分類
  - システム状態の変化を監視
- **Subscriber**:
  - `/image_vlm`: カメラ映像データを受信
- **Publisher**:
  - `/state`: システムの現在状態を発信
- **メッセージ型**: `vla_interfaces/msg/State`

#### 3. State Manager Node (`/state_manager`) 
- **役割**: システム状態の管理と適切なアクション指示
- **機能**:
  - VLM Watcherからの状態情報を基にした意思決定
  - 復旧アクションの計画と指示
  - システム全体のワークフロー制御
- **Subscriber**:
  - `/state`: システム状態情報を受信
- **Publisher**:
  - `/action`: 実行すべきアクションを指示
- **メッセージ型**: `vla_interfaces/msg/Action`

#### 4. VLA Controller Node (`/vla_controller`)
- **役割**: 物理的なロボット制御とアクション実行
- **機能**:
  - State Managerからのアクション指示を実行
  - NVIDIA Isaac VLAとの連携
  - VLA映像データを基にした精密な動作制御
- **Subscriber**:
  - `/action`: 実行すべきアクションを受信
  - `/image_vla`: アクション実行時の映像フィードバック

## メッセージ定義

### State メッセージ (`vla_interfaces/msg/State`)
```
uint8 NORMAL           = 0  # 正常状態
uint8 ANOMALY_DETECTED = 1  # 異常検知
uint8 RECOVERING       = 2  # 復旧中

uint8 state                          # 現在の状態
vla_interfaces/Action action         # 関連するアクション
uint64 stamp                         # タイムスタンプ
```

### Action メッセージ (`vla_interfaces/msg/Action`)
```
int8 action_id                # アクションID
string language_instruction   # 自然言語での指示内容
```

## システムワークフロー

1. **映像取得**: Camera Nodeがカメラデバイスから映像を取得
2. **異常検知**: VLM WatcherがGPT-4 Visionで映像を分析し、異常を検知
3. **状態管理**: State Managerが状態変化を監視し、適切なアクションを決定
4. **アクション実行**: GR00T Controllerが物理的なロボット動作を実行
5. **フィードバック**: 実行結果を基にシステム状態を更新

## 技術スタック

- **ROS2**: ロボット制御フレームワーク
- **Python 3**: 主要開発言語
- **Azure OpenAI GPT-4 Vision**: 映像解析とVLM処理
- **NVIDIA Isaac GR00T**: ロボット制御プラットフォーム  
- **LeRobot**: テレオペレーションとポリシー実行
- **OpenCV**: 映像処理

## セットアップと実行

### 環境構築
```bash
# ROSワークスペースでのビルド
cd /workspace/ros
colcon build
source install/setup.bash
```

### 設定ファイル
Azure OpenAI APIキーを設定:
```bash
cat > /workspace/ros/config/.env << 'EOF'
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
AZURE_OPENAI_API_VERSION=2024-02-15-preview
USE_AZURE_OPENAI=true
EOF
```

### ノードの実行

#### 個別実行
```bash
ros2 run vla_auto_recover camera_node
ros2 run vla_auto_recover vlm_watcher_node  
ros2 run vla_auto_recover state_manager_node
ros2 run vla_auto_recover vla_controller_node
```

#### 一括実行
```bash
ros2 launch vla_auto_recover all_nodes.launch.py
```

## ハードウェア要件

- **ロボットアーム**: SO100 Follower (シリアル接続: /dev/ttyACM1)
- **テレオペレータ**: SO100 Leader (シリアル接続: /dev/ttyACM0)  
- **カメラ**: OpenCVで認識可能なUSBカメラ
- **GPU**: NVIDIA GPU (Isaac GR00T実行用)

## 使用例

### テレオペレーション
```bash
python -m lerobot.teleoperate \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.cameras="{ center_cam: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM0
```

### 自動実行
```bash
python scripts/exe_policy_lerobot2.py \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM1 \
    --policy_host=localhost \
    --lang_instruction="move blocks from tray to matching dishes."
```

## 特徴

- **自動異常検知**: GPT-4 Visionによる高精度な映像解析
- **自動復旧**: 検知された異常に対する自動的な復旧アクション
- **モジュラー設計**: 各ノードが独立して動作し、柔軟な拡張が可能
- **リアルタイム処理**: 低遅延での状態監視と制御
- **自然言語指示**: 人間が理解しやすい形でのアクション記述

このシステムは、産業用ロボットの信頼性向上と自動化レベルの向上を目的とした次世代のロボット制御システムです。
