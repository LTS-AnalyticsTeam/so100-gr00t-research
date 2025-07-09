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
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing
import logging
from datetime import datetime

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
    
    def __init__(self, dataset_base_path: str = "/workspace/datasets/extract_images", 
                 verbose: bool = False, num_workers: int = None, 
                 log_file: str = None, resume_from_log: bool = False):
        self.dataset_base_path = Path(dataset_base_path)
        self.verbose = verbose
        self.num_workers = num_workers or min(4, multiprocessing.cpu_count())
        self.log_file = log_file
        self.resume_from_log = resume_from_log
        
        # ログ設定
        self.logger = self._setup_logger()
        
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
        self.completed_episodes = set()  # 完了済みエピソードを追跡
        
        # 既存のログファイルから完了済みエピソードを読み込み
        if self.resume_from_log and self.log_file and Path(self.log_file).exists():
            self._load_completed_episodes()
        # 既存の結果JSONファイルから完了済みエピソードを読み込み
        self.load_completed_episodes_from_results(log_file)
    
    def _setup_logger(self) -> logging.Logger:
        """ログ設定を初期化"""
        logger = logging.getLogger('VLMEvaluator')
        logger.setLevel(logging.INFO)
        
        # 既存のハンドラーをクリア
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # フォーマッターを設定
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # コンソールハンドラー
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO if self.verbose else logging.WARNING)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # ファイルハンドラー（指定されている場合）
        if self.log_file:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(self.log_file, mode='a', encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _load_completed_episodes(self):
        """既存のログファイルから完了済みエピソードを読み込み"""
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('EPISODE_RESULT:'):
                        # ログ形式: EPISODE_RESULT:category/episode_id:is_correct:actual_result:processing_time
                        parts = line.split(':', 4)
                        if len(parts) >= 5:
                            episode_key = parts[1]
                            is_correct = parts[2] == 'True'
                            actual_result = parts[3]
                            processing_time = float(parts[4])
                            
                            self.completed_episodes.add(episode_key)
                            
                            # ログからEvaluationResultを復元
                            category, episode_id = episode_key.split('/')
                            expected_result = self.category_expected_mapping[category]
                            expected_action_id = self.category_action_mapping[category]
                            
                            # アクションIDの推定（実際の値は復元不可なので期待値を使用）
                            actual_action_id = expected_action_id if is_correct else None
                            action_id_correct = is_correct if expected_action_id != 0 else True
                            
                            result = EvaluationResult(
                                dataset_category=category,
                                episode_id=episode_id,
                                phase="static",
                                expected_result=expected_result.value,
                                actual_result=actual_result,
                                is_correct=is_correct,
                                processing_time=processing_time,
                                reason="",  # ログからは復元不可
                                expected_action_id=expected_action_id,
                                actual_action_id=actual_action_id,
                                action_id_correct=action_id_correct
                            )
                            self.evaluation_results.append(result)
            
            self.logger.info(f"Loaded {len(self.completed_episodes)} completed episodes from log file")
            
        except Exception as e:
            self.logger.warning(f"Failed to load completed episodes from log: {e}")
    
    def load_completed_episodes_from_results(self, results_file: str):
        """既存の結果JSONファイルから完了済みエピソードを読み込み"""
        try:
            results_path = Path(results_file)
            if not results_path.exists():
                return
            
            with open(results_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'evaluation_results' in data:
                for result_data in data['evaluation_results']:
                    episode_key = f"{result_data['dataset_category']}/{result_data['episode_id']}"
                    self.completed_episodes.add(episode_key)
                    
                    # EvaluationResultオブジェクトを復元
                    result = EvaluationResult(
                        dataset_category=result_data['dataset_category'],
                        episode_id=result_data['episode_id'],
                        phase=result_data['phase'],
                        expected_result=result_data['expected_result'],
                        actual_result=result_data['actual_result'],
                        is_correct=result_data['is_correct'],
                        processing_time=result_data['processing_time'],
                        reason=result_data.get('reason', ''),
                        expected_action_id=result_data.get('expected_action_id'),
                        actual_action_id=result_data.get('actual_action_id'),
                        action_id_correct=result_data.get('action_id_correct')
                    )
                    self.evaluation_results.append(result)
            
            self.logger.info(f"Loaded {len(self.completed_episodes)} completed episodes from results file: {results_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to load completed episodes from results file: {e}")
    
    def _log_episode_result(self, result: EvaluationResult):
        """エピソード結果をログファイルに記録"""
        if self.log_file:
            try:
                episode_key = f"{result.dataset_category}/{result.episode_id}"
                log_entry = f"EPISODE_RESULT:{episode_key}:{result.is_correct}:{result.actual_result}:{result.processing_time:.3f}"
                
                # ログファイルに追記
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(f"{log_entry}\n")
                    f.flush()  # 即座にディスクに書き込み
                
                # 完了済みエピソードに追加
                self.completed_episodes.add(episode_key)
                
            except Exception as e:
                self.logger.error(f"Failed to log episode result: {e}")
    
    def _is_episode_completed(self, category: str, episode_id: str) -> bool:
        """エピソードが完了済みかチェック"""
        episode_key = f"{category}/{episode_id}"
        return episode_key in self.completed_episodes
    
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
        """カテゴリ全体の評価（並列処理対応）"""
        category_path = self.dataset_base_path / category
        if not category_path.exists():
            print(f"Category path does not exist: {category_path}")
            return []
        
        # エピソードディレクトリを取得
        episode_dirs = [d for d in category_path.iterdir() if d.is_dir()]
        episode_dirs.sort()
        
        if max_samples:
            episode_dirs = episode_dirs[:max_samples]
        
        # 完了済みエピソードを除外
        pending_episodes = []
        skipped_count = 0
        
        for episode_dir in episode_dirs:
            episode_id = episode_dir.name
            if self._is_episode_completed(category, episode_id):
                skipped_count += 1
                continue
            pending_episodes.append(episode_id)
        
        if skipped_count > 0:
            self.logger.info(f"Skipping {skipped_count} already completed episodes in category '{category}'")
        
        if not pending_episodes:
            self.logger.info(f"All episodes in category '{category}' are already completed")
            return []
        
        results = []
        print(f"\nEvaluating category: {category} ({len(pending_episodes)} pending episodes, {skipped_count} skipped)")
        
        if self.num_workers == 1:
            # シーケンシャル実行
            for i, episode_id in enumerate(pending_episodes):
                if self.verbose:
                    print(f"  Episode {i+1}/{len(pending_episodes)}: {episode_id}")
                else:
                    print(f"  Progress: {i+1}/{len(pending_episodes)} - {episode_id}", end="\r")
                
                result = self.evaluate_single_sample(category, episode_id)
                results.append(result)
                self.evaluation_results.append(result)
                
                # ログファイルに逐次書き込み
                self._log_episode_result(result)
        else:
            # 並列実行
            self.logger.info(f"Starting parallel evaluation with {self.num_workers} workers")
            
            # 並列処理用の引数を準備
            worker_args = [
                (category, episode_id, str(self.dataset_base_path), 
                 self.category_expected_mapping, self.category_action_mapping)
                for episode_id in pending_episodes
            ]
            
            completed_count = 0
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # 全てのタスクを送信
                future_to_episode = {
                    executor.submit(evaluate_single_episode_worker, args): args[1] 
                    for args in worker_args
                }
                
                # 完了したタスクから順次処理
                for future in as_completed(future_to_episode):
                    episode_id = future_to_episode[future]
                    try:
                        result = future.result()
                        results.append(result)
                        self.evaluation_results.append(result)
                        
                        # ログファイルに逐次書き込み
                        self._log_episode_result(result)
                        
                        completed_count += 1
                        
                        if self.verbose:
                            action_info = ""
                            if result.expected_action_id is not None:
                                action_info = f", Action: Expected {result.expected_action_id}, Got {result.actual_action_id}"
                            self.logger.info(f"    {category}/{episode_id}: "
                                          f"Expected {result.expected_result}, Got {result.actual_result}, "
                                          f"Correct: {result.is_correct}{action_info}")
                        else:
                            print(f"  Progress: {completed_count}/{len(pending_episodes)} - {episode_id}", end="\r")
                    
                    except Exception as e:
                        self.logger.error(f"Error processing episode {episode_id}: {e}")
                        # エラーの場合もダミーの結果を作成
                        error_result = EvaluationResult(
                            dataset_category=category,
                            episode_id=episode_id,
                            phase="static",
                            expected_result=self.category_expected_mapping[category].value,
                            actual_result="ERROR",
                            is_correct=False,
                            processing_time=0.0,
                            reason=f"Worker error: {str(e)}",
                            expected_action_id=self.category_action_mapping[category],
                            actual_action_id=None,
                            action_id_correct=False
                        )
                        results.append(error_result)
                        self.evaluation_results.append(error_result)
                        self._log_episode_result(error_result)
        
        if not self.verbose:
            print(f"  Completed: {len(pending_episodes)} episodes")
        
        return results
    
    def evaluate_all_categories(self, max_samples_per_category: int = None, 
                               categories: List[str] = None) -> PerformanceMetrics:
        """指定されたカテゴリの評価"""
        print("Starting VLM Performance Evaluation...")
        print(f"Dataset path: {self.dataset_base_path}")
        print(f"Parallel workers: {self.num_workers}")
        if self.log_file:
            print(f"Log file: {self.log_file}")
        if self.resume_from_log:
            print(f"Resume mode: {len(self.completed_episodes)} episodes already completed")
        
        # 評価するカテゴリを決定
        if categories is None:
            categories = list(self.category_expected_mapping.keys())
        
        self.logger.info(f"Starting evaluation for categories: {categories}")
        
        # 各カテゴリを評価
        for category in categories:
            if category in self.category_expected_mapping:
                try:
                    self.evaluate_category(category, max_samples_per_category)
                    self.logger.info(f"Completed evaluation for category: {category}")
                except Exception as e:
                    self.logger.error(f"Error evaluating category {category}: {e}")
                    if self.verbose:
                        import traceback
                        self.logger.error(traceback.format_exc())
            else:
                print(f"Warning: Unknown category '{category}' skipped")
                self.logger.warning(f"Unknown category '{category}' skipped")
        
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
    
    def save_metrics_to_file(self, metrics_file: str = None):
        """メトリクスをファイルに保存"""
        if not metrics_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_file = f"/workspace/eval/metrics/metrics_{timestamp}.json"
        
        metrics = self.calculate_metrics()
        
        metrics_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_path": str(self.dataset_base_path),
            "total_samples": len(self.evaluation_results),
            "performance_metrics": asdict(metrics)
        }
        
        metrics_path = Path(metrics_file)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Metrics saved to: {metrics_path}")
        return metrics_path
    
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


def evaluate_single_episode_worker(args):
    """並列処理用のワーカー関数（プロセス間で実行される）"""
    category, episode_id, dataset_base_path, category_expected_mapping, category_action_mapping = args
    
    try:
        # 各プロセスでVLMDetectorを初期化
        workspace_path = Path('/workspace')
        ros_src_path = workspace_path / 'ros' / 'src'
        sys.path.insert(0, str(ros_src_path))
        sys.path.insert(0, str(ros_src_path / 'vla_auto_recover'))
        
        from vla_auto_recover.processing.vlm_detector import VLMDetector
        from vla_auto_recover.processing.config.system_settings import ADR, CB_InputIF
        
        vlm_detector = VLMDetector()
        
        start_time = time.time()
        
        # 画像ペアを読み込み
        episode_dir = Path(dataset_base_path) / category / episode_id
        
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
        
        images = [center_cam, right_cam]
        
        # 期待結果を取得
        expected_result = category_expected_mapping[category]
        expected_action_id = category_action_mapping[category]
        
        # VLMDetectorで評価
        input_data = CB_InputIF(images=images)
        output_data = vlm_detector.CB_RUNNING(input_data)
        
        processing_time = time.time() - start_time
        
        # 結果を比較
        is_correct = output_data.detection_result == expected_result
        
        # アクションIDの評価
        action_id_correct = None
        if expected_action_id is not None:
            action_id_correct = output_data.action_id == expected_action_id
        elif output_data.detection_result in [ADR.NORMAL, ADR.COMPLETION]:
            action_id_correct = output_data.action_id is None
        
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
        
        return EvaluationResult(
            dataset_category=category,
            episode_id=episode_id,
            phase="static",
            expected_result=category_expected_mapping[category].value,
            actual_result="ERROR",
            is_correct=False,
            processing_time=processing_time,
            reason=f"Error: {str(e)}",
            expected_action_id=category_action_mapping[category],
            actual_action_id=None,
            action_id_correct=False
        )


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
  
  # 並列処理で実行（8ワーカー）
  python evaluate_vlm_performance.py --workers 8
  
  # ログファイルを指定して逐次保存
  python evaluate_vlm_performance.py --log-file eval_progress.log
  
  # 前回のログから再開
  python evaluate_vlm_performance.py --log-file eval_progress.log --resume
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
    
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"並列処理のワーカー数（デフォルト: min(4, CPU数={multiprocessing.cpu_count()})）"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="進行状況を記録するログファイルパス"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="既存のログファイルから評価を再開する"
    )
    
    parser.add_argument(
        "--resume-from-results",
        type=str,
        default=None,
        help="既存の結果JSONファイルから完了済みエピソードを読み込んで再開する"
    )
    
    parser.add_argument(
        "--metrics-file",
        type=str,
        default=None,
        help="メトリクスを保存するファイルパス"
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
    print(f"Workers: {args.workers or f'Auto ({min(4, multiprocessing.cpu_count())})'}")
    if args.log_file:
        print(f"Log file: {args.log_file}")
        print(f"Resume mode: {args.resume}")
    if args.metrics_file:
        print(f"Metrics file: {args.metrics_file}")
    
    # デフォルトのログファイルを生成（指定されていない場合）
    if not args.log_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_file = f"/workspace/eval/log/eval_progress_{timestamp}.log"
    
    # 評価器の初期化
    evaluator = VLMPerformanceEvaluator(
        dataset_base_path=args.dataset_path,
        verbose=args.verbose,
        num_workers=args.workers,
        log_file=args.log_file,
        resume_from_log=args.resume
    )
    
    # 既存の結果ファイルからの復元
    if args.resume_from_results:
        evaluator.load_completed_episodes_from_results(args.resume_from_results)
        print(f"Resumed from results file: {args.resume_from_results}")
    
    try:
        # 指定されたカテゴリを評価
        print(f"\nStarting evaluation...")
        evaluator.logger.info("="*60)
        evaluator.logger.info("VLM PERFORMANCE EVALUATION STARTED")
        evaluator.logger.info("="*60)
        evaluator.logger.info(f"Dataset path: {args.dataset_path}")
        evaluator.logger.info(f"Categories: {args.categories}")
        evaluator.logger.info(f"Max samples per category: {args.max_samples}")
        evaluator.logger.info(f"Workers: {evaluator.num_workers}")
        evaluator.logger.info(f"Resume mode: {args.resume}")
        
        start_time = time.time()
        
        metrics = evaluator.evaluate_all_categories(
            max_samples_per_category=args.max_samples,
            categories=args.categories
        )
        
        total_time = time.time() - start_time
        print(f"\nEvaluation completed in {total_time:.2f} seconds")
        
        evaluator.logger.info("="*60)
        evaluator.logger.info("VLM PERFORMANCE EVALUATION COMPLETED")
        evaluator.logger.info(f"Total time: {total_time:.2f} seconds")
        evaluator.logger.info(f"Total samples: {metrics.total_samples}")
        evaluator.logger.info(f"Overall accuracy: {metrics.accuracy:.3f}")
        evaluator.logger.info("="*60)
        
        # 結果の表示と保存
        evaluator.print_summary()
        output_file = evaluator.save_results(args.output)
        
        # メトリクスファイルの保存
        if args.metrics_file or evaluator.evaluation_results:
            metrics_file = evaluator.save_metrics_to_file(args.metrics_file)
            print(f"Metrics saved to: {metrics_file}")
        
        print(f"\nEvaluation completed successfully!")
        print(f"Results saved to: {output_file}")
        print(f"Log file: {args.log_file}")
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        evaluator.logger.info("Evaluation interrupted by user")
        # 中間結果も保存
        if evaluator.evaluation_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            interrupted_output = f"/workspace/eval/vlm_evaluation_results_interrupted_{timestamp}.json"
            evaluator.save_results(interrupted_output)
            # メトリクスも保存
            metrics_file = evaluator.save_metrics_to_file(f"/workspace/eval/metrics/metrics_interrupted_{timestamp}.json")
            print(f"Partial results saved to: {interrupted_output}")
            print(f"Partial metrics saved to: {metrics_file}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        evaluator.logger.error(f"Evaluation failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
            evaluator.logger.error(traceback.format_exc())
        # エラー時も中間結果を保存
        if evaluator.evaluation_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            error_output = f"/workspace/eval/vlm_evaluation_results_error_{timestamp}.json"
            evaluator.save_results(error_output)
            # メトリクスも保存
            metrics_file = evaluator.save_metrics_to_file(f"/workspace/eval/metrics/metrics_error_{timestamp}.json")
            print(f"Partial results saved to: {error_output}")
            print(f"Partial metrics saved to: {metrics_file}")
        sys.exit(1)


if __name__ == "__main__":
    main()
