# VLM Performance Evaluation Tool

VLMDetectorの画像判定性能を評価するツールです。並列処理、逐次ログ書き込み、エピソードスキップ機能により、大規模データセットの効率的で堅牢な評価が可能です。

## 主要機能

- **並列処理**: マルチプロセシングによる高速評価
- **逐次ログ保存**: 各エピソード結果をリアルタイムで保存
- **再開機能**: 中断された評価を既存ログから継続
- **エピソードスキップ**: 既評価データの自動スキップ
- **エラー耐性**: 部分エラー時も評価継続、中間結果保存
- **詳細メトリクス**: 混同行列、アクションID精度、処理時間統計

## 必要な環境設定

```bash
# ROSワークスペースのビルド
cd /workspace/ros && colcon build && source install/setup.bash

# OpenAI API キーの設定
export OPENAI_API_KEY="your_api_key_here"
```

## 基本的な使用法

```bash
cd /workspace/eval

# 並列評価（推奨）
python evaluate_vlm_performance.py --workers 4 --max-samples 5

# 中断・再開可能な評価
python evaluate_vlm_performance.py --log-file progress.log --workers 4

# 前回から再開
python evaluate_vlm_performance.py --log-file progress.log --resume
```

## オプション一覧

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--workers N` | 並列ワーカー数 | 4 |
| `--max-samples N` | 各カテゴリの最大サンプル数 | 制限なし |
| `--log-file PATH` | ログファイルパス | なし |
| `--resume` | ログファイルからの再開 | false |
| `--categories LIST` | 評価カテゴリ（カンマ区切り） | 全カテゴリ |
| `--output PATH` | 結果JSONファイルパス | `vlm_evaluation_results.json` |
| `--metrics-file PATH` | メトリクスファイルパス | `metrics/metrics_YYYYMMDD_HHMMSS.json` |
| `--verbose` | 詳細ログ出力 | false |

## 評価対象とカテゴリ

| カテゴリ | 期待される判定結果 | アクションID |
|----------|------------------|-------------|
| **normal** | NORMAL | - |
| **complete** | COMPLETION | - |
| **relocate** | ANOMALY | 2 |
| **unstack** | ANOMALY | 1 |

## 出力ファイル

### 1. 結果JSONファイル
- **デフォルト**: `vlm_evaluation_results.json`
- **内容**: 全エピソードの詳細な評価結果

### 2. メトリクスファイル
- **デフォルト**: `metrics/metrics_YYYYMMDD_HHMMSS.json`
- **内容**: 精度、混同行列、処理時間統計

### 3. ログファイル（オプション）
- **内容**: 各エピソードの実行ログ（再開に使用）

## 評価結果の見方

### コンソール出力例
```
================================================================================
VLM PERFORMANCE EVALUATION SUMMARY
================================================================================
Total Samples: 20
Overall Accuracy: 0.950 (95.0%)
Average Processing Time: 4.612 seconds

------------------------------------------------------------
CATEGORY-WISE PERFORMANCE
------------------------------------------------------------
normal      : 1.000 (100.0%) -   5/5   samples
complete    : 0.800 ( 80.0%) -   4/5   samples
relocate    : 1.000 (100.0%) -   5/5   samples
unstack     : 1.000 (100.0%) -   5/5   samples

------------------------------------------------------------
ACTION ID PERFORMANCE
------------------------------------------------------------
Overall Action ID Accuracy: 0.900 (90.0%)
relocate    : 1.000 (100.0%) -   5/5   samples
unstack     : 0.800 ( 80.0%) -   4/5   samples
================================================================================
```

### ログファイルの解釈

各行は以下のJSON形式で記録されます：
```json
{
  "timestamp": "2025-01-09T15:30:45.123456",
  "category": "relocate",
  "episode_id": "001",
  "expected_result": "ANOMALY",
  "actual_result": "ANOMALY",
  "is_correct": true,
  "action_id_correct": true,
  "processing_time": 4.82
}
```

### エピソード別実行時間の抽出

```bash
# ログファイルから実行時間を抽出
grep '"processing_time"' progress.log | sed 's/.*"processing_time": *\([0-9.]*\).*/\1/'

# 平均実行時間を計算
grep '"processing_time"' progress.log | sed 's/.*"processing_time": *\([0-9.]*\).*/\1/' | awk '{sum+=$1; count++} END {print "Average:", sum/count, "seconds"}'
```

## 実用的な使用例

### 大規模評価（全カテゴリ）
```bash
# ログ付き並列評価
python evaluate_vlm_performance.py \
  --workers 8 \
  --log-file logs/full_evaluation.log \
  --metrics-file metrics/full_metrics.json

# 中断後の再開
python evaluate_vlm_performance.py \
  --log-file logs/full_evaluation.log \
  --resume
```

### 特定カテゴリのテスト
```bash
# 異常検知カテゴリのみ
python evaluate_vlm_performance.py \
  --categories relocate,unstack \
  --max-samples 10 \
  --workers 4
```

### 開発・デバッグ用
```bash
# 少数サンプルでの動作確認
python evaluate_vlm_performance.py \
  --max-samples 2 \
  --workers 1 \
  --verbose
```

## トラブルシューティング

### よくある問題と解決方法

**1. 環境エラー**
```bash
# ROSワークスペースの再ビルド
cd /workspace/ros && colcon build --symlink-install && source install/setup.bash

# OpenAI APIキーの確認
echo $OPENAI_API_KEY
```

**2. 並列処理でのエラー**
```bash
# ワーカー数を減らす
python evaluate_vlm_performance.py --workers 2

# シーケンシャル実行
python evaluate_vlm_performance.py --workers 1 --verbose
```

**3. メモリ不足**
```bash
# バッチサイズを制限
python evaluate_vlm_performance.py --max-samples 5 --workers 2
```

**4. ログファイルの問題**
```bash
# 新しいログファイルで開始
python evaluate_vlm_performance.py --log-file new_evaluation.log
```

### エラー時の中間結果保存

エラーや中断が発生した場合、以下のファイルが自動保存されます：
- `vlm_evaluation_results_interrupted_YYYYMMDD_HHMMSS.json`
- `vlm_evaluation_results_error_YYYYMMDD_HHMMSS.json`
- `metrics/metrics_interrupted_YYYYMMDD_HHMMSS.json`
- `metrics/metrics_error_YYYYMMDD_HHMMSS.json`
