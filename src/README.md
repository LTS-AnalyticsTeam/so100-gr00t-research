# VLA & VLM Auto Recovery

Azure OpenAI GPT-4o(GPT-4.1)を使用してロボット(GR00T&SO-100)作業中の異常を検出し、自動的に修正アクションを実行する。

## システム構成

```
VLMNode/VLMWatcher → StateManager → GR00TController
    ↓                     ↓             ↓
  画像分析               状態管理      ロボット制御
```

## コンポーネント （本番環境）
### B. StateManager (`state_manager/`)
- **機能**: システム状態管理、VLA実行制御
- **状態**: Normal ↔ Recovering
- **出力**: `/vla_pause` (VLA制御), `/system_state` (状態通知)
- **特徴**: タイムアウト処理、詳細ログ

### C. GR00TController (`gr00t_controller/`)
- **機能**: VLA実行・リカバリーアクション実行
- **入力**: `/recovery_action`, `/vla_pause`
- **出力**: `/recovery_status` (実行状況)
- **特徴**: Pattern1実装（VLA統合型）

### D. VLMWatcher (`vlm_watcher/`)
- **機能**: リアルタイム画像ストリーム監視
- **入力**: `/rgb_image` (カメラストリーム)
- **用途**: 実際のロボット運用時

### E. VLAInterfaces (`vla_interfaces/`)
- **機能**: カスタムメッセージ定義
- **メッセージ**: Action, RecoveryStatus

## コンポーネント （テスト環境）
### A. VLMNode (`vlm_node/`)
- **機能**: 画像から異常検出、リカバリーアクション提案
- **入力**: 画像ファイル、プロンプト、アクションリスト
- **出力**: `/recovery_action` (異常検出時)
- **特徴**: Azure OpenAI対応、single-shot・連続処理モード

### B. StateManager (`state_manager/`)
- **機能**: システム状態管理、VLA実行制御
- **状態**: Normal ↔ Recovering
- **出力**: `/vla_pause` (VLA制御), `/system_state` (状態通知)
- **特徴**: タイムアウト処理、詳細ログ

### C. GR00TController (`gr00t_controller/`)
- **機能**: VLA実行・リカバリーアクション実行
- **入力**: `/recovery_action`, `/vla_pause`
- **出力**: `/recovery_status` (実行状況)
- **特徴**: Pattern1実装（VLA統合型）

### E. VLAInterfaces (`vla_interfaces/`)
- **機能**: カスタムメッセージ定義
- **メッセージ**: Action, RecoveryStatus


## セットアップ

### 必要なパッケージ
```bash
# 追加パッケージ
pip install empy lark pillow

# ROS2ワークスペース
cd /workspace
colcon build
source install/setup.bash # (こっちは各ターミナルで実行する)
```

### 環境設定
```bash
# .env ファイル作成
cat > /workspace/.env << 'EOF'
# Azure OpenAI設定
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
AZURE_OPENAI_API_VERSION=2024-02-15-preview
USE_AZURE_OPENAI=true
EOF
```

### 設定ファイル
```bash
# 以下に各ファイルを置く, 既にデフォルトのものは格納済み

# プロンプトファイル (/workspace/config/prompt.txt)
# アクションリスト (/workspace/config/action_list.jsonl)
# テスト画像 (/workspace/src/data/)
```

## 実行方法

### 基本実行（3つのターミナル）

**Terminal 1: StateManager**
```bash
cd /workspace
source install/setup.bash
ros2 run state_manager state_manager_node
```

**Terminal 2: GR00TController**
```bash
cd /workspace
source install/setup.bash
ros2 run gr00t_controller gr00t_controller_node
```

**Terminal 3: VLMNode（ファイル処理）**
```bash
cd /workspace
source install/setup.bash

# 1枚テスト
ros2 run vlm_node vlm_node \
  --input-dir /workspace/src/data \
  --prompt-file /workspace/config/prompt.txt \
  --action-list-file /workspace/config/action_list.jsonl \
  --output-file /tmp/vlm_results.txt \
  --single-shot

# 特定画像指定
ros2 run vlm_node vlm_node \
  --input-dir /workspace/src/data \
  --prompt-file /workspace/config/prompt.txt \
  --action-list-file /workspace/config/action_list.jsonl \
  --output-file /tmp/vlm_results.txt \
  --single-shot \
  --target-image "test_image.jpg"

# 連続処理
ros2 run vlm_node vlm_node \
  --input-dir /workspace/src/data \
  --prompt-file /workspace/config/prompt.txt \
  --action-list-file /workspace/config/action_list.jsonl \
  --output-file /tmp/vlm_results.txt
```

### リアルタイム監視（実機運用）

**Terminal 3: VLMWatcher（カメラ入力）**
```bash
ros2 run vlm_watcher vlm_watcher_node \
  --fps 5.0 \
  --action-list /workspace/config/action_list.jsonl \
  --prompt "Detect anomalies in block sorting task"
```

## 監視・デバッグ

### トピック監視
```bash
# 基本トピック
ros2 topic echo /recovery_action
ros2 topic echo /recovery_status  
ros2 topic echo /vla_pause
ros2 topic echo /system_state

# 全トピック同時監視
ros2 topic echo /recovery_action &
ros2 topic echo /recovery_status &
ros2 topic echo /vla_pause &
ros2 topic echo /system_state &
```

### ログ確認
```bash
# VLMNode詳細ログ
tail -f /tmp/vlm_node_*.log

# StateManager詳細ログ  
tail -f /tmp/state_manager_logs/state_manager_*.log

# GR00TController詳細ログ
tail -f /tmp/gr00t_logs/gr00t_controller_*.log

# 結果ファイル
cat /tmp/vlm_results.txt
```

### 手動テスト
```bash
# 手動でリカバリーアクション送信
ros2 topic pub /recovery_action vla_interfaces/Action "
name: 'Move the red block from the tray to the right dish.'
target_id: 'block_red'
" --once

# 手動でVLA制御
ros2 topic pub /vla_pause std_msgs/Bool "data: true" --once   # 停止
ros2 topic pub /vla_pause std_msgs/Bool "data: false" --once  # 再開

# 手動でリカバリー状況送信
ros2 topic pub /recovery_status vla_interfaces/RecoveryStatus "
completed: true
failed: false
action_name: 'Move the red block from the tray to the right dish.'
target_id: 'block_red'
progress: 1.0
status: 'completed'
" --once
```

## 動作フロー

1. **VLMNode**: 画像分析 → 異常検出 → `/recovery_action` 発行
2. **StateManager**: Normal → Recovering → `/vla_pause: true` 発行
3. **GR00TController**: VLA停止 → リカバリー実行 → `/recovery_status` 発行
4. **StateManager**: `/recovery_status` 受信 → Recovering → Normal → `/vla_pause: false`
5. **GR00TController**: VLA再開

## 設定ファイル例

### action_list.jsonl
```json
{"action": "Move the red block from the tray to the right dish.", "target_id": "block_red"}
{"action": "Move the green block from the tray to the left dish.", "target_id": "block_green"}
{"action": "Reposition the red block to the center of the right dish.", "target_id": "block_red"}
```

### prompt.txt
```
Analyze this image for anomalies in a block sorting task.

IMPORTANT: Respond ONLY with valid JSON:
{
  "anomaly": "normal|pose_error|missing_block|extra_block",
  "action": "action_name_from_list",
  "target_id": "target_id_from_list",
  "confidence": 0.85,
  "reasoning": "Brief explanation"
}
```
## システム要件

- **ROS2**: Jazzy以降
- **Python**: 3.8以降  
- **OpenAI**: GPT-4o, 4.1対応
- **ハードウェア**: カメラ、ロボットアーム（オプション）