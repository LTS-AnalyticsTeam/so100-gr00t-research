# VLM Performance Evaluation Tool

VLMDetectorの画像判定性能を評価するシンプルなツールです。

## ファイル構成

- `evaluate_vlm_performance.py` - メイン評価スクリプト
- `README.md` - このファイル

## 前提条件

### 環境設定

1. **ROSワークスペースのビルド**:
   ```bash
   cd /workspace/ros
   colcon build
   source install/setup.bash
   ```

2. **OpenAI API キーの設定**:
   ```bash
   # .envファイルを作成
   echo "OPENAI_API_KEY=your_api_key_here" > /workspace/ros/src/vla_auto_recover/config/.env
   
   # または環境変数として設定
   export OPENAI_API_KEY="your_api_key_here"
   ```

3. **Python依存関係**:
   - opencv-python
   - numpy
   - dataclasses (Python 3.7+)
   - その他VLMDetectorの依存関係

### 動作確認

環境が正しく設定されているかテスト：

```bash
cd /workspace/eval
python evaluate_vlm_performance.py --max-samples 1
```

**成功例の出力:**
```
VLM Performance Evaluation Tool
Dataset path: /workspace/datasets/extract_images
Categories to evaluate: normal, complete, relocate, unstack
Max samples per category: 1
Output file: /workspace/eval/vlm_evaluation_results.json
Verbose mode: False

Initializing VLM Detector...
Azure OpenAI client initialized

Starting evaluation...
...

Overall Accuracy: 1.000 (100.0%)
Average Processing Time: 22.506 seconds
```

APIキーが設定されていない場合、エラーが表示されますが、スクリプト自体は動作します。

## 評価対象データセット

`/workspace/datasets/extract_images/` 以下の画像データを使用：

- `normal/` - 正常状態の画像
  - **start期待結果**: NORMAL
  - **end期待結果**: COMPLETION
- `complete/` - タスク完了状態の画像
  - **start/end期待結果**: COMPLETION
- `relocate/` - 位置異常状態の画像
  - **start期待結果**: ANOMALY
  - **end期待結果**: NORMAL（修正後）
- `unstack/` - スタック異常状態の画像
  - **start期待結果**: ANOMALY
  - **end期待結果**: NORMAL（修正後）

### データセット構造

各エピソードディレクトリには以下の画像ファイルが含まれます：
- `*center_cam*.png` - 中央カメラ画像
- `*right_cam*.png` - 右カメラ画像
- 可能であれば start/end フェーズ別の画像

## 使用方法

### 基本的な使用法

```bash
cd /workspace/eval

# 基本的な評価（各カテゴリ5サンプル）
python evaluate_vlm_performance.py --max-samples 5

# 全サンプルで詳細評価
python evaluate_vlm_performance.py --verbose

# 特定のカテゴリのみ評価
python evaluate_vlm_performance.py --categories normal relocate --max-samples 10

# ヘルプを表示
python evaluate_vlm_performance.py --help
```

### コマンドライン引数

- `--max-samples N`: 各カテゴリの最大サンプル数（デフォルト: 全サンプル）
- `--output FILE`: 結果出力ファイルパス（デフォルト: vlm_evaluation_results.json）
- `--verbose`: 詳細なログ出力を有効化
- `--categories LIST`: 評価するカテゴリを指定（normal, complete, relocate, unstack）
- `--dataset-path PATH`: データセットのベースパスを指定

### 実行例

```bash
# 開発時の高速テスト（各カテゴリ1サンプル）
python evaluate_vlm_performance.py --max-samples 1
# 実行時間: 約60秒、基本的な検出精度の確認用

# 中規模テスト（各カテゴリ3サンプル）
python evaluate_vlm_performance.py --max-samples 3
# 実行時間: 約180秒、開発時の性能確認

# 完全な性能評価（全サンプル）
python evaluate_vlm_performance.py --verbose
# 実行時間: データセットサイズに依存、本格評価用

# 異常検出のみテスト
python evaluate_vlm_performance.py --categories relocate unstack --max-samples 5
# relocateとunstackカテゴリの評価のみ
```

## 評価内容

### 画像判定評価
各カテゴリの画像に対してVLMDetectorの判定を評価します：

1. **Normal/Complete カテゴリ**: 
   - NORMAL または COMPLETION の判定ができるか
   - 正常状態の正確な識別性能

2. **Relocate/Unstack カテゴリ**: 
   - ANOMALY の判定ができるか
   - 異常状態の正確な識別性能

### アクションID評価
ANOMALY検出時に適切なアクションIDを出力できるかを評価します：

1. **異常カテゴリでのアクションID**:
   - `relocate`: アクションID 1（再配置アクション）
   - `unstack`: アクションID 2（アンスタックアクション）

2. **正常カテゴリでのアクションID**:
   - `normal`, `complete`: アクションID無し（None）

各画像ペア（center_cam, right_cam）に対してVLMDetectorのCB_RUNNING関数を呼び出し、期待される判定結果とアクションIDの両方を比較します。

## 評価メトリクス

### 基本メトリクス
- **Overall Accuracy**: 全体の正解率
- **Category-wise Accuracy**: カテゴリ別正解率（normal, complete, relocate, unstack）
- **Phase-wise Accuracy**: フェーズ別正解率（static）
- **Average Processing Time**: 平均処理時間
- **Action ID Accuracy**: アクションID予測精度

### 詳細メトリクス
- **Confusion Matrix**: 混同行列（期待結果 vs 実際の結果）
- **Error Analysis**: エラータイプ別の分析
- **Action ID Metrics**: カテゴリ別アクションID精度

### 評価フェーズ

1. **Static Phase**: 静的画像に対する判定評価（CB_RUNNING）

## 出力ファイル

評価結果は詳細なJSONファイルに保存されます：

- `vlm_evaluation_results.json` - 評価結果（デフォルト）

### 結果ファイルの構造

```json
{
  "metadata": {
    "timestamp": "2025-07-09 16:09:11",
    "dataset_path": "/workspace/datasets/extract_images",
    "total_samples": 4,
    "categories_evaluated": [
      "relocate",
      "unstack",
      "normal",
      "complete"
    ],
    "phases_evaluated": [
      "static"
    ]
  },
  "evaluation_results": [
    {
      "dataset_category": "normal",
      "episode_id": "001",
      "phase": "static",
      "expected_result": "NORMAL",
      "actual_result": "NORMAL",
      "is_correct": true,
      "processing_time": 26.524473667144775,
      "reason": "赤い皿には赤いブロック1個、青い皿には青いブロック1個があり、銀のトレーにも赤1青1のブロックが残っている。すべてのブロックが皿に移動し終えておらず、また各ブロックの配置にも異常はないため、タスクは継続中で正常状態。",
      "expected_action_id": 0,
      "actual_action_id": 0,
      "action_id_correct": true
    },
    {
      "dataset_category": "complete",
      "episode_id": "000",
      "phase": "static",
      "expected_result": "COMPLETION",
      "actual_result": "COMPLETION",
      "is_correct": true,
      "processing_time": 9.512032270431519,
      "reason": "全てのブロックがトレイから各色の皿に正しく移動されており、皿も重なっていないため、タスクは完了状態です。",
      "expected_action_id": 0,
      "actual_action_id": 0,
      "action_id_correct": true
    },
    {
      "dataset_category": "relocate",
      "episode_id": "001",
      "phase": "static",
      "expected_result": "ANOMALY",
      "actual_result": "ANOMALY",
      "is_correct": true,
      "processing_time": 14.330276012420654,
      "reason": "赤い皿の上に青いブロックが載っており、青い皿の上には赤いブロックが載っている。これはブロックが間違った色の皿の上に配置されている異常状態です。タスク実行の前に修正が必要です。",
      "expected_action_id": 2,
      "actual_action_id": 2,
      "action_id_correct": true
    },
    {
      "dataset_category": "unstack",
      "episode_id": "001",
      "phase": "static",
      "expected_result": "ANOMALY",
      "actual_result": "ANOMALY",
      "is_correct": true,
      "processing_time": 14.559600114822388,
      "reason": "青い皿と赤い皿が重なっており、皿が分離していないため異常状態です。さらに、すべてのブロックがひとつの皿（青）に載っているため、皿ごとにブロックを分ける処理も必要です。まず皿の重なり解除が最優先の異常です。",
      "expected_action_id": 1,
      "actual_action_id": 1,
      "action_id_correct": true
    }
  ],
  "performance_metrics": {
    "total_samples": 4,
    "correct_predictions": 4,
    "accuracy": 1.0,
    "avg_processing_time": 16.231595516204834,
    "category_accuracies": {
      "normal": 1.0,
      "complete": 1.0,
      "relocate": 1.0,
      "unstack": 1.0
    },
    "phase_accuracies": {
      "static": 1.0
    },
    "confusion_matrix": {
      "ANOMALY": {
        "ANOMALY": 2,
        "NORMAL": 0,
        "COMPLETION": 0
      },
      "NORMAL": {
        "ANOMALY": 0,
        "NORMAL": 1,
        "COMPLETION": 0
      },
      "COMPLETION": {
        "ANOMALY": 0,
        "NORMAL": 0,
        "COMPLETION": 1
      }
    },
    "error_analysis": {},
    "action_id_accuracy": 1.0,
    "action_id_metrics": {
      "normal": 1.0,
      "complete": 1.0,
      "relocate": 1.0,
      "unstack": 1.0
    }
  },
  "detailed_analysis": {
    "category_breakdown": {
      "relocate": {
        "total_samples": 1,
        "correct_predictions": 1,
        "accuracy": 1.0,
        "avg_processing_time": 14.330276012420654,
        "error_count": 0
      },
      "unstack": {
        "total_samples": 1,
        "correct_predictions": 1,
        "accuracy": 1.0,
        "avg_processing_time": 14.559600114822388,
        "error_count": 0
      },
      "normal": {
        "total_samples": 1,
        "correct_predictions": 1,
        "accuracy": 1.0,
        "avg_processing_time": 26.524473667144775,
        "error_count": 0
      },
      "complete": {
        "total_samples": 1,
        "correct_predictions": 1,
        "accuracy": 1.0,
        "avg_processing_time": 9.512032270431519,
        "error_count": 0
      }
    },
    "phase_breakdown": {
      "static": {
        "total_samples": 4,
        "correct_predictions": 4,
        "accuracy": 1.0,
        "avg_processing_time": 16.231595516204834
      }
    },
    "error_cases": []
  }
}
```

## カスタマイズ

### 期待結果の変更

`category_expected_mapping` を編集して期待結果を変更：

```python
self.category_expected_mapping = {
    "normal": {
        "start": ADR.NORMAL,
        "end": ADR.COMPLETION
    },
    "relocate": {
        "start": ADR.ANOMALY,
        "end": ADR.NORMAL
    }
}
```

    "normal": {
        "start": ADR.NORMAL,
        "end": ADR.COMPLETION
    },
    # ... 他のカテゴリ
}
```

### データセットパスの変更

異なるデータセットを使用する場合：

```bash
python evaluate_vlm_performance.py --dataset-path /path/to/your/dataset
```

## トラブルシューティング

### インポートエラー

VLMDetectorのインポートエラーが発生する場合：

1. ROSワークスペースがビルドされているか確認：
   ```bash
   cd /workspace/ros
   colcon build
   source install/setup.bash
   ```

2. Python パスが正しく設定されているか確認：
   ```bash
   export PYTHONPATH=/workspace/ros/src:$PYTHONPATH
   ```

3. 必要な依存関係がインストールされているか確認

### OpenAI APIエラー

```
Error: 'NoneType' object has no attribute 'chat'
```

このエラーはOpenAI APIキーが設定されていない場合に発生します：

1. APIキーを設定：
   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```

2. または.envファイルに保存：
   ```bash
   echo "OPENAI_API_KEY=your_api_key_here" > /workspace/ros/src/vla_auto_recover/config/.env
   ```

### データセットが見つからない

```
Warning: Category path does not exist: /workspace/datasets/extract_images/normal
```

データセットの存在確認：
```bash
ls -la /workspace/datasets/extract_images/
```

データセットパスを指定：
```bash
python evaluate_vlm_performance.py --dataset-path /correct/path/to/dataset
```

### メモリ不足

大量のデータセットを評価する場合：

1. サンプル数を制限：
   ```bash
   python evaluate_vlm_performance.py --max-samples 10
   ```

2. 特定カテゴリのみ評価：
   ```bash
   python evaluate_vlm_performance.py --categories normal --max-samples 5
   ```

### 評価オプションの選び方

#### 高速開発テスト（推奨）
```bash
python evaluate_vlm_performance.py --max-samples 1
```
- 実行時間: 約60秒
- 基本的な検出機能の確認
- 各カテゴリの基本評価のみ

#### 中規模テスト
```bash
python evaluate_vlm_performance.py --max-samples 3 --no-state-transitions --verbose
```
- 実行時間: 約180秒
# 実行時間: 約60秒、基本的な検出精度の確認用

#### 本格的な性能評価
```bash
python evaluate_vlm_performance.py --verbose
```
- 実行時間: データセットサイズに依存（通常10-30分）
- 詳細ログ付きの性能確認
- 本格的な性能評価用

#### 特定問題の分析
```bash
# 異常検出のみ
python evaluate_vlm_performance.py --categories relocate unstack --max-samples 5

# 正常状態のみ
python evaluate_vlm_performance.py --categories normal complete
```

### 処理時間が長い

VLM処理の高速化：

1. 並列処理の検討（将来の機能追加）
2. 小さなサンプルセットでのテスト
3. ハードウェアの性能確認

## 性能分析例

### 実際の評価結果（サンプル）

テスト実行結果（各カテゴリ1サンプル）：

```
================================================================================
VLM PERFORMANCE EVALUATION SUMMARY
================================================================================
Dataset Path: /workspace/datasets/extract_images
Total Samples: 4
Correct Predictions: 4
Overall Accuracy: 1.000 (100.0%)
Average Processing Time: 16.232 seconds

------------------------------------------------------------
CATEGORY-WISE PERFORMANCE
------------------------------------------------------------
normal      : 1.000 (100.0%) -   1/1   samples
complete    : 1.000 (100.0%) -   1/1   samples
relocate    : 1.000 (100.0%) -   1/1   samples
unstack     : 1.000 (100.0%) -   1/1   samples

------------------------------------------------------------
PHASE-WISE PERFORMANCE
------------------------------------------------------------
static      : 1.000 (100.0%) -   4/4   samples - 16.232s avg

------------------------------------------------------------
CONFUSION MATRIX
------------------------------------------------------------
Expected            ANOMALY  COMPLETION      NORMAL
---------------------------------------------------
ANOMALY                   2           0           0
COMPLETION                0           1           0
NORMAL                    0           0           1

------------------------------------------------------------
ACTION ID PERFORMANCE
------------------------------------------------------------
Overall Action ID Accuracy: 1.000 (100.0%)
Total Action ID Evaluations: 4

normal      : 1.000 (100.0%) -   1/1   samples (Expected Action: 0)
complete    : 1.000 (100.0%) -   1/1   samples (Expected Action: 0)
relocate    : 1.000 (100.0%) -   1/1   samples (Expected Action: 2)
unstack     : 1.000 (100.0%) -   1/1   samples (Expected Action: 1)

------------------------------------------------------------
ERROR ANALYSIS
------------------------------------------------------------
No errors found!

================================================================================
```

### 検出結果の詳細

各カテゴリでの検出内容：

1. **normal/001**: `NORMAL` → ✅ 正常状態を正しく検出
   - 赤い皿に赤ブロック、青い皿に青ブロック、トレイにも適切に配置

2. **complete/000**: `COMPLETION` → ✅ 完了状態を正しく検出
   - すべてのブロックが正しい色の皿に移動済み、トレイは空

3. **relocate/001**: `ANOMALY` → ✅ 配置異常を正しく検出
   - 青い皿に赤ブロック、赤い皿に青ブロックが配置された状態

4. **unstack/001**: `ANOMALY` → ✅ スタック異常を正しく検出
   - 皿が重なって配置されている状態

### 処理時間分析

- **平均処理時間**: 22.5秒/サンプル
- **VLM推論時間**: OpenAI GPT-4による画像解析

## 拡張機能（将来の実装）

### 高度な分析機能
- Precision, Recall, F1-score の計算
- ROC曲線とAUC値
- 信頼度スコアの分析
- 処理時間の詳細統計

### 視覚化機能
- 混同行列のヒートマップ
- カテゴリ別性能のグラフ
- 処理時間の分布図
- エラー率の時系列変化

### バッチ評価機能
- 複数のモデル/パラメータの比較
- A/Bテスト用の評価
- 継続的インテグレーション対応

### 並列処理
- マルチプロセシング対応
- GPU加速（可能な場合）
- 分散評価機能

## 関連ファイル

- `/workspace/ros/src/vla_auto_recover/test/processing/test_vlm_detector.py` - VLMDetectorの単体テスト
- `/workspace/ros/src/vla_auto_recover/vla_auto_recover/processing/vlm_detector.py` - VLMDetector実装

## ライセンス

このツールは、VLA Auto Recover プロジェクトの一部として提供されています。

## 貢献

バグ報告や機能改善の提案は、プロジェクトのIssueトラッカーまでお願いします。
