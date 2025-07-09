#!/usr/bin/env python3
"""
VLMDetectorの画像判定性能を評価するスクリプト

使用方法:
    python evaluate_vlm_performance.py [--max-samples N] [--output OUTPUT_FILE] [--verbose]
    
このスクリプトは datasets/extract_images 下の画像データを使用して
VLMDetectorの異常検出性能を評価し、正しい判定ができる割合を計測します。
"""

import sys
import os
import argparse
from pathlib import Path
import cv2
import json
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np

# プロジェクトのルートパスを追加
workspace_path = Path('/workspace')
ros_src_path = workspace_path / 'ros' / 'src'
sys.path.insert(0, str(ros_src_path))

# パッケージの直接インポート
sys.path.insert(0, str(ros_src_path / 'vla_auto_recover'))

from vla_auto_recover.processing.vlm_detector import VLMDetector
from vla_auto_recover.processing.config.system_settings import (
    ADR, CB_InputIF, CB_OutputIF
)


@dataclass
class EvaluationResult:
    """評価結果を格納するデータクラス"""
    dataset_category: str
    episode_id: str
    phase: str  # "static" - 静的画像評価
    expected_result: str
    actual_result: str
    is_correct: bool
    processing_time: float = 0.0
    reason: str = ""
    expected_action_id: Optional[int] = None
    actual_action_id: Optional[int] = None
    action_id_correct: Optional[bool] = None  # アクションIDが正しいかどうか
    

@dataclass
class PerformanceMetrics:
    """性能メトリクスを格納するデータクラス"""
    total_samples: int
    correct_predictions: int
    accuracy: float
    avg_processing_time: float
    category_accuracies: Dict[str, float]
    phase_accuracies: Dict[str, float]
    confusion_matrix: Dict[str, Dict[str, int]]
    error_analysis: Dict[str, int]
    action_id_accuracy: float = 0.0  # アクションID予測精度
    action_id_metrics: Dict[str, float] = None  # カテゴリ別アクションID精度


class VLMPerformanceEvaluator:
    """VLMDetectorの性能評価クラス"""
    
    def __init__(self, dataset_base_path: str = "/workspace/datasets/extract_images", verbose: bool = False):
        self.dataset_base_path = Path(dataset_base_path)
        self.verbose = verbose
        
        print("Initializing VLM Detector...")
        self.vlm_detector = VLMDetector()
        
        # カテゴリと期待結果のマッピング（静的画像判定）
        self.category_expected_mapping = {
            "normal": ADR.NORMAL,      # 正常状態（作業進行中）
            "complete": ADR.COMPLETION, # タスク完了状態
            "relocate": ADR.ANOMALY,   # 配置異常状態
            "unstack": ADR.ANOMALY,    # スタック異常状態
        }
        
        # 異常カテゴリに対する期待アクションIDのマッピング
        self.category_action_mapping = {
            "normal": 0,
            "complete": 0,
            "relocate": 2,      # 再配置アクション
            "unstack": 1,       # アンスタックアクション
        }
        
        self.evaluation_results: List[EvaluationResult] = []
    
    def load_image_pair(self, category: str, episode_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """画像ペアを読み込む"""
        episode_dir = self.dataset_base_path / category / episode_id
        
        # 画像ファイルを検索
        center_files = list(episode_dir.glob("*center_cam*.png"))
        right_files = list(episode_dir.glob("*right_cam*.png"))
        
        if not center_files or not right_files:
            raise FileNotFoundError(f"Image files not found in {episode_dir}")
        
        # 最初の画像を使用
        center_cam = cv2.imread(str(center_files[0]))
        right_cam = cv2.imread(str(right_files[0]))
        
        if center_cam is None or right_cam is None:
            raise ValueError(f"Failed to load images from {episode_dir}")
        
        return center_cam, right_cam
    
    def evaluate_single_sample(self, category: str, episode_id: str) -> EvaluationResult:
        """単一サンプルの評価"""
        start_time = time.time()
        
        try:
            # 画像ペアを読み込み
            center_cam, right_cam = self.load_image_pair(category, episode_id)
            images = [center_cam, right_cam]
            
            # 期待結果を取得
            expected_result = self.category_expected_mapping[category]
            expected_action_id = self.category_action_mapping[category]
            
            # VLMDetectorで評価
            input_data = CB_InputIF(images=images)
            output_data = self.vlm_detector.CB_RUNNING(input_data)
            
            processing_time = time.time() - start_time
            
            # 結果を比較
            is_correct = output_data.detection_result == expected_result
            
            # アクションIDの評価
            action_id_correct = None
            if expected_action_id is not None:
                # 異常検出時はアクションIDが必要
                action_id_correct = output_data.action_id == expected_action_id
            elif output_data.detection_result in [ADR.NORMAL, ADR.COMPLETION]:
                # 正常/完了時はアクションIDは不要（NoneまたはNullであることを期待）
                action_id_correct = output_data.action_id is None
            
            if self.verbose:
                action_info = ""
                if expected_action_id is not None:
                    action_info = f", Action: Expected {expected_action_id}, Got {output_data.action_id}"
                print(f"    {category}/{episode_id}: "
                      f"Expected {expected_result.value}, Got {output_data.detection_result.value if output_data.detection_result else 'NONE'}, "
                      f"Correct: {is_correct}{action_info}")
            
            return EvaluationResult(
                dataset_category=category,
                episode_id=episode_id,
                phase="static",
                expected_result=expected_result.value,
                actual_result=output_data.detection_result.value if output_data.detection_result else "NONE",
                is_correct=is_correct,
                processing_time=processing_time,
                reason=output_data.reason or "",
                expected_action_id=expected_action_id,
                actual_action_id=output_data.action_id,
                action_id_correct=action_id_correct
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            if self.verbose:
                print(f"    ERROR {category}/{episode_id}: {str(e)}")
            
            return EvaluationResult(
                dataset_category=category,
                episode_id=episode_id,
                phase="static",
                expected_result=self.category_expected_mapping[category].value,
                actual_result="ERROR",
                is_correct=False,
                processing_time=processing_time,
                reason=f"Error: {str(e)}",
                expected_action_id=self.category_action_mapping[category],
                actual_action_id=None,
                action_id_correct=False
            )
    
    def evaluate_category(self, category: str, max_samples: int = None) -> List[EvaluationResult]:
        """カテゴリ全体の評価"""
        category_path = self.dataset_base_path / category
        if not category_path.exists():
            print(f"Category path does not exist: {category_path}")
            return []
        
        # エピソードディレクトリを取得
        episode_dirs = [d for d in category_path.iterdir() if d.is_dir()]
        episode_dirs.sort()
        
        if max_samples:
            episode_dirs = episode_dirs[:max_samples]
        
        results = []
        print(f"\nEvaluating category: {category} ({len(episode_dirs)} episodes)")
        
        for i, episode_dir in enumerate(episode_dirs):
            episode_id = episode_dir.name
            
            if self.verbose:
                print(f"  Episode {i+1}/{len(episode_dirs)}: {episode_id}")
            else:
                print(f"  Progress: {i+1}/{len(episode_dirs)} - {episode_id}", end="\r")
            
            result = self.evaluate_single_sample(category, episode_id)
            results.append(result)
            self.evaluation_results.append(result)
        
        if not self.verbose:
            print(f"  Completed: {len(episode_dirs)} episodes")
        return results
    
    def evaluate_all_categories(self, max_samples_per_category: int = None, 
                               categories: List[str] = None) -> PerformanceMetrics:
        """指定されたカテゴリの評価"""
        print("Starting VLM Performance Evaluation...")
        print(f"Dataset path: {self.dataset_base_path}")
        
        # 評価するカテゴリを決定
        if categories is None:
            categories = list(self.category_expected_mapping.keys())
        
        # 各カテゴリを評価
        for category in categories:
            if category in self.category_expected_mapping:
                self.evaluate_category(category, max_samples_per_category)
            else:
                print(f"Warning: Unknown category '{category}' skipped")
        
        # 性能メトリクスを計算
        return self.calculate_metrics()
    
    def calculate_metrics(self) -> PerformanceMetrics:
        """性能メトリクスを計算"""
        if not self.evaluation_results:
            return PerformanceMetrics(0, 0, 0.0, 0.0, {}, {}, {}, {}, 0.0, {})
        
        total_samples = len(self.evaluation_results)
        correct_predictions = sum(1 for r in self.evaluation_results if r.is_correct)
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        avg_processing_time = sum(r.processing_time for r in self.evaluation_results) / total_samples
        
        # カテゴリ別精度
        category_accuracies = {}
        for category in self.category_expected_mapping.keys():
            category_results = [r for r in self.evaluation_results if r.dataset_category == category]
            if category_results:
                category_correct = sum(1 for r in category_results if r.is_correct)
                category_accuracies[category] = category_correct / len(category_results)
            else:
                category_accuracies[category] = 0.0
        
        # フェーズ別精度（静的評価では "static" のみ）
        phase_accuracies = {}
        for phase in ["static"]:
            phase_results = [r for r in self.evaluation_results if r.phase == phase]
            if phase_results:
                phase_correct = sum(1 for r in phase_results if r.is_correct)
                phase_accuracies[phase] = phase_correct / len(phase_results)
            else:
                phase_accuracies[phase] = 0.0
        
        # アクションID関連のメトリクス
        action_id_results = [r for r in self.evaluation_results if r.action_id_correct is not None]
        action_id_accuracy = 0.0
        if action_id_results:
            action_id_correct_count = sum(1 for r in action_id_results if r.action_id_correct)
            action_id_accuracy = action_id_correct_count / len(action_id_results)
        
        # カテゴリ別アクションID精度
        action_id_metrics = {}
        for category in self.category_expected_mapping.keys():
            category_action_results = [r for r in self.evaluation_results 
                                     if r.dataset_category == category and r.action_id_correct is not None]
            if category_action_results:
                category_action_correct = sum(1 for r in category_action_results if r.action_id_correct)
                action_id_metrics[category] = category_action_correct / len(category_action_results)
            else:
                action_id_metrics[category] = 0.0
        
        # 混同行列
        confusion_matrix = {}
        all_expected = set(r.expected_result for r in self.evaluation_results)
        all_actual = set(r.actual_result for r in self.evaluation_results)
        
        for expected in all_expected:
            confusion_matrix[expected] = {}
            for actual in all_actual:
                count = sum(1 for r in self.evaluation_results 
                          if r.expected_result == expected and r.actual_result == actual)
                confusion_matrix[expected][actual] = count
        
        # エラー分析
        error_analysis = {}
        error_results = [r for r in self.evaluation_results if not r.is_correct]
        for result in error_results:
            error_type = f"{result.expected_result} -> {result.actual_result}"
            error_analysis[error_type] = error_analysis.get(error_type, 0) + 1
        
        return PerformanceMetrics(
            total_samples=total_samples,
            correct_predictions=correct_predictions,
            accuracy=accuracy,
            avg_processing_time=avg_processing_time,
            category_accuracies=category_accuracies,
            phase_accuracies=phase_accuracies,
            confusion_matrix=confusion_matrix,
            error_analysis=error_analysis,
            action_id_accuracy=action_id_accuracy,
            action_id_metrics=action_id_metrics
        )
    
    def save_results(self, output_path: str = "/workspace/eval/vlm_evaluation_results.json"):
        """評価結果をJSONファイルに保存"""
        metrics = self.calculate_metrics()
        
        results_data = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "dataset_path": str(self.dataset_base_path),
                "total_samples": len(self.evaluation_results),
                "categories_evaluated": list(set(r.dataset_category for r in self.evaluation_results)),
                "phases_evaluated": list(set(r.phase for r in self.evaluation_results))
            },
            "evaluation_results": [asdict(r) for r in self.evaluation_results],
            "performance_metrics": asdict(metrics),
            "detailed_analysis": {
                "category_breakdown": self._get_category_breakdown(),
                "phase_breakdown": self._get_phase_breakdown(),
                "error_cases": [asdict(r) for r in self.evaluation_results if not r.is_correct]
            }
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_path}")
        return output_path
    
    def _get_category_breakdown(self) -> Dict:
        """カテゴリ別詳細分析"""
        breakdown = {}
        for category in set(r.dataset_category for r in self.evaluation_results):
            category_results = [r for r in self.evaluation_results if r.dataset_category == category]
            breakdown[category] = {
                "total_samples": len(category_results),
                "correct_predictions": sum(1 for r in category_results if r.is_correct),
                "accuracy": sum(1 for r in category_results if r.is_correct) / len(category_results),
                "avg_processing_time": sum(r.processing_time for r in category_results) / len(category_results),
                "error_count": sum(1 for r in category_results if not r.is_correct)
            }
        return breakdown
    
    def _get_phase_breakdown(self) -> Dict:
        """フェーズ別詳細分析"""
        breakdown = {}
        for phase in set(r.phase for r in self.evaluation_results):
            phase_results = [r for r in self.evaluation_results if r.phase == phase]
            breakdown[phase] = {
                "total_samples": len(phase_results),
                "correct_predictions": sum(1 for r in phase_results if r.is_correct),
                "accuracy": sum(1 for r in phase_results if r.is_correct) / len(phase_results),
                "avg_processing_time": sum(r.processing_time for r in phase_results) / len(phase_results)
            }
        return breakdown
    
    def print_summary(self):
        """評価結果のサマリーを表示"""
        metrics = self.calculate_metrics()
        
        print("\n" + "="*80)
        print("VLM PERFORMANCE EVALUATION SUMMARY")
        print("="*80)
        print(f"Dataset Path: {self.dataset_base_path}")
        print(f"Total Samples: {metrics.total_samples}")
        print(f"Correct Predictions: {metrics.correct_predictions}")
        print(f"Overall Accuracy: {metrics.accuracy:.3f} ({metrics.accuracy*100:.1f}%)")
        print(f"Average Processing Time: {metrics.avg_processing_time:.3f} seconds")
        
        print("\n" + "-"*60)
        print("CATEGORY-WISE PERFORMANCE")
        print("-"*60)
        for category, accuracy in metrics.category_accuracies.items():
            category_results = [r for r in self.evaluation_results if r.dataset_category == category]
            total = len(category_results)
            correct = sum(1 for r in category_results if r.is_correct)
            print(f"{category:12}: {accuracy:.3f} ({accuracy*100:>5.1f}%) - {correct:>3}/{total:<3} samples")
        
        print("\n" + "-"*60)
        print("PHASE-WISE PERFORMANCE")
        print("-"*60)
        for phase, accuracy in metrics.phase_accuracies.items():
            phase_results = [r for r in self.evaluation_results if r.phase == phase]
            if phase_results:
                total = len(phase_results)
                correct = sum(1 for r in phase_results if r.is_correct)
                avg_time = sum(r.processing_time for r in phase_results) / len(phase_results)
                print(f"{phase:12}: {accuracy:.3f} ({accuracy*100:>5.1f}%) - {correct:>3}/{total:<3} samples - {avg_time:.3f}s avg")
        
        print("\n" + "-"*60)
        print("CONFUSION MATRIX")
        print("-"*60)
        actual_labels = sorted(set().union(*metrics.confusion_matrix.values()))
        print(f"{'Expected':<15}", end="")
        for actual in actual_labels:
            print(f"{actual:>12}", end="")
        print()
        print("-" * (15 + 12 * len(actual_labels)))
        
        for expected, actual_counts in sorted(metrics.confusion_matrix.items()):
            print(f"{expected:<15}", end="")
            for actual in actual_labels:
                count = actual_counts.get(actual, 0)
                print(f"{count:>12}", end="")
            print()
        
        print("\n" + "-"*60)
        print("ACTION ID PERFORMANCE")
        print("-"*60)
        action_id_results = [r for r in self.evaluation_results if r.action_id_correct is not None]
        if action_id_results:
            print(f"Overall Action ID Accuracy: {metrics.action_id_accuracy:.3f} ({metrics.action_id_accuracy*100:.1f}%)")
            print(f"Total Action ID Evaluations: {len(action_id_results)}")
            print()
            for category, accuracy in metrics.action_id_metrics.items():
                category_action_results = [r for r in self.evaluation_results 
                                         if r.dataset_category == category and r.action_id_correct is not None]
                if category_action_results:
                    total = len(category_action_results)
                    correct = sum(1 for r in category_action_results if r.action_id_correct)
                    expected_action = self.category_action_mapping[category]
                    print(f"{category:12}: {accuracy:.3f} ({accuracy*100:>5.1f}%) - {correct:>3}/{total:<3} samples (Expected Action: {expected_action})")
        else:
            print("No action ID evaluations performed (all samples were NORMAL/COMPLETION)")
        
        print("\n" + "-"*60)
        print("ERROR ANALYSIS")
        print("-"*60)
        if metrics.error_analysis:
            for error_type, count in sorted(metrics.error_analysis.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / metrics.total_samples) * 100
                print(f"{error_type:<25}: {count:>3} cases ({percentage:>5.1f}%)")
        else:
            print("No errors found!")
        
        # アクションIDエラーの詳細表示
        action_id_errors = [r for r in self.evaluation_results if r.action_id_correct is False]
        if action_id_errors:
            print(f"\n" + "-"*60)
            print(f"ACTION ID ERROR CASES (showing first 10 of {len(action_id_errors)})")
            print("-"*60)
            for i, case in enumerate(action_id_errors[:10]):
                print(f"{i+1:>2}. {case.dataset_category}/{case.episode_id}: "
                      f"Expected Action {case.expected_action_id}, Got {case.actual_action_id}")
        
        # 最も多いエラーケースの詳細表示
        error_cases = [r for r in self.evaluation_results if not r.is_correct]
        if error_cases:
            print(f"\n" + "-"*60)
            print(f"SAMPLE ERROR CASES (showing first 10 of {len(error_cases)})")
            print("-"*60)
            for i, case in enumerate(error_cases[:10]):
                print(f"{i+1:>2}. {case.dataset_category}/{case.episode_id}({case.phase}): "
                      f"Expected '{case.expected_result}', Got '{case.actual_result}'")
                if case.reason:
                    reason_short = case.reason[:80] + "..." if len(case.reason) > 80 else case.reason
                    print(f"     Reason: {reason_short}")
        
        print("\n" + "="*80)


def parse_arguments():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(
        description="VLMDetectorの画像判定性能を評価します",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 全カテゴリを評価（各カテゴリ最大5サンプル）
  python evaluate_vlm_performance.py --max-samples 5

  # 詳細なログ出力で全カテゴリを評価
  python evaluate_vlm_performance.py --verbose

  # 特定のカテゴリのみ評価
  python evaluate_vlm_performance.py --categories normal complete
        """
    )
    
    parser.add_argument(
        "--max-samples", 
        type=int, 
        default=None,
        help="各カテゴリの最大サンプル数（デフォルト: 全サンプル）"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="/workspace/eval/vlm_evaluation_results.json",
        help="結果出力ファイルパス"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="詳細なログ出力を有効にする"
    )
    
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["normal", "complete", "relocate", "unstack"],
        choices=["normal", "complete", "relocate", "unstack"],
        help="評価するカテゴリを指定"
    )
    
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/workspace/datasets/extract_images",
        help="データセットのベースパス"
    )
    
    return parser.parse_args()


def main():
    """メイン実行関数"""
    args = parse_arguments()
    
    print("VLM Performance Evaluation Tool")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Categories to evaluate: {', '.join(args.categories)}")
    print(f"Max samples per category: {args.max_samples or 'All'}")
    print(f"Output file: {args.output}")
    print(f"Verbose mode: {args.verbose}")
    
    # 評価器の初期化
    evaluator = VLMPerformanceEvaluator(
        dataset_base_path=args.dataset_path,
        verbose=args.verbose
    )
    
    try:
        # 指定されたカテゴリを評価
        print(f"\nStarting evaluation...")
        start_time = time.time()
        
        metrics = evaluator.evaluate_all_categories(
            max_samples_per_category=args.max_samples,
            categories=args.categories
        )
        
        total_time = time.time() - start_time
        print(f"\nEvaluation completed in {total_time:.2f} seconds")
        
        # 結果の表示と保存
        evaluator.print_summary()
        output_file = evaluator.save_results(args.output)
        
        print(f"\nEvaluation completed successfully!")
        print(f"Results saved to: {output_file}")
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
