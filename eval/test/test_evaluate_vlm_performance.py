#!/usr/bin/env python3
"""
evaluate_vlm_performance.py のユニットテスト

使用方法:
    python -m pytest test/test_evaluate_vlm_performance.py -v
    
このテストは VLMPerformanceEvaluator クラスの各メソッドと
評価ロジックを検証します。
"""

import sys
import os
import unittest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import cv2

# テスト対象のモジュールをインポート
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluate_vlm_performance import VLMPerformanceEvaluator, EvaluationResult, PerformanceMetrics


class MockADR:
    """ADR列挙型のモック"""
    NORMAL = Mock()
    NORMAL.value = "NORMAL"
    
    COMPLETION = Mock()
    COMPLETION.value = "COMPLETION"
    
    ANOMALY = Mock()
    ANOMALY.value = "ANOMALY"


class MockCB_InputIF:
    """CB_InputIF データクラスのモック"""
    def __init__(self, images=None, action_id=None):
        self.images = images
        self.action_id = action_id


class MockCB_OutputIF:
    """CB_OutputIF データクラスのモック"""
    def __init__(self, detection_result=None, action_id=None, reason=None):
        self.detection_result = detection_result
        self.action_id = action_id
        self.reason = reason


class MockVLMDetector:
    """VLMDetector のモック"""
    def __init__(self):
        self.responses = {}
    
    def set_response(self, category, episode_id, detection_result, action_id=None, reason=""):
        """テスト用にレスポンスを設定"""
        key = f"{category}/{episode_id}"
        self.responses[key] = MockCB_OutputIF(detection_result, action_id, reason)
    
    def CB_RUNNING(self, input_data):
        """モックされたCB_RUNNING メソッド"""
        # デフォルトレスポンス
        return self.responses.get("default", MockCB_OutputIF(MockADR.NORMAL, None, "Mock response"))


class TestVLMPerformanceEvaluator(unittest.TestCase):
    """VLMPerformanceEvaluator のテストクラス"""
    
    def setUp(self):
        """テストケースの前処理"""
        # 一時ディレクトリを作成
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_path = Path(self.temp_dir) / "test_dataset"
        
        # テスト用データセット構造を作成
        self.create_test_dataset()
        
        # ADR, CB_InputIF, CB_OutputIF のモック
        self.adr_mock = MockADR()
        self.cb_input_mock = MockCB_InputIF
        self.cb_output_mock = MockCB_OutputIF
        
        # VLMDetector のモック
        self.vlm_detector_mock = MockVLMDetector()
    
    def tearDown(self):
        """テストケースの後処理"""
        # 一時ディレクトリを削除
        shutil.rmtree(self.temp_dir)
    
    def create_test_dataset(self):
        """テスト用のデータセット構造を作成"""
        categories = ["normal", "complete", "relocate", "unstack"]
        
        for category in categories:
            category_dir = self.dataset_path / category
            category_dir.mkdir(parents=True, exist_ok=True)
            
            # 各カテゴリに2つのエピソードを作成
            for episode_num in range(1, 3):
                episode_dir = category_dir / f"{episode_num:03d}"
                episode_dir.mkdir(exist_ok=True)
                
                # ダミー画像ファイルを作成
                center_cam_path = episode_dir / f"episode_000000_observation.images.center_cam_frame_00000{episode_num}.png"
                right_cam_path = episode_dir / f"episode_000000_observation.images.right_cam_frame_00000{episode_num}.png"
                
                # 100x100のダミー画像を作成
                dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
                cv2.imwrite(str(center_cam_path), dummy_image)
                cv2.imwrite(str(right_cam_path), dummy_image)
    
    @patch('evaluate_vlm_performance.VLMDetector')
    @patch('evaluate_vlm_performance.ADR', new_callable=lambda: MockADR())
    @patch('evaluate_vlm_performance.CB_InputIF', new_callable=lambda: MockCB_InputIF)
    def test_evaluator_initialization(self, mock_cb_input, mock_adr, mock_vlm_detector):
        """VLMPerformanceEvaluator の初期化テスト"""
        mock_vlm_detector.return_value = self.vlm_detector_mock
        
        evaluator = VLMPerformanceEvaluator(
            dataset_base_path=str(self.dataset_path),
            verbose=False
        )
        
        self.assertEqual(evaluator.dataset_base_path, self.dataset_path)
        self.assertFalse(evaluator.verbose)
        self.assertEqual(len(evaluator.category_expected_mapping), 4)
        self.assertEqual(len(evaluator.category_action_mapping), 4)
        self.assertEqual(len(evaluator.evaluation_results), 0)
    
    @patch('evaluate_vlm_performance.VLMDetector')
    @patch('evaluate_vlm_performance.ADR', new_callable=lambda: MockADR())
    @patch('evaluate_vlm_performance.CB_InputIF', new_callable=lambda: MockCB_InputIF)
    def test_load_image_pair(self, mock_cb_input, mock_adr, mock_vlm_detector):
        """画像ペア読み込みのテスト"""
        mock_vlm_detector.return_value = self.vlm_detector_mock
        
        evaluator = VLMPerformanceEvaluator(
            dataset_base_path=str(self.dataset_path),
            verbose=False
        )
        
        # 正常なケース
        center_cam, right_cam = evaluator.load_image_pair("normal", "001")
        self.assertIsInstance(center_cam, np.ndarray)
        self.assertIsInstance(right_cam, np.ndarray)
        self.assertEqual(center_cam.shape, (100, 100, 3))
        self.assertEqual(right_cam.shape, (100, 100, 3))
        
        # 存在しないエピソードのテスト
        with self.assertRaises(FileNotFoundError):
            evaluator.load_image_pair("normal", "999")
    
    @patch('evaluate_vlm_performance.VLMDetector')
    @patch('evaluate_vlm_performance.ADR', new_callable=lambda: MockADR())
    @patch('evaluate_vlm_performance.CB_InputIF', new_callable=lambda: MockCB_InputIF)
    def test_evaluate_single_sample_normal(self, mock_cb_input, mock_adr, mock_vlm_detector):
        """単一サンプル評価のテスト（正常ケース）"""
        mock_vlm_detector.return_value = self.vlm_detector_mock
        
        # VLMDetectorのレスポンスを設定
        self.vlm_detector_mock.set_response(
            "normal", "001", 
            MockADR.NORMAL, 0, "Normal state detected"
        )
        self.vlm_detector_mock.responses["default"] = MockCB_OutputIF(
            MockADR.NORMAL, 0, "Normal state detected"
        )
        
        evaluator = VLMPerformanceEvaluator(
            dataset_base_path=str(self.dataset_path),
            verbose=False
        )
        
        result = evaluator.evaluate_single_sample("normal", "001")
        
        self.assertIsInstance(result, EvaluationResult)
        self.assertEqual(result.dataset_category, "normal")
        self.assertEqual(result.episode_id, "001")
        self.assertEqual(result.phase, "static")
        self.assertEqual(result.expected_result, "NORMAL")
        self.assertEqual(result.actual_result, "NORMAL")
        self.assertTrue(result.is_correct)
        self.assertEqual(result.expected_action_id, 0)
        self.assertEqual(result.actual_action_id, 0)
        self.assertTrue(result.action_id_correct)
    
    @patch('evaluate_vlm_performance.VLMDetector')
    @patch('evaluate_vlm_performance.ADR', new_callable=lambda: MockADR())
    @patch('evaluate_vlm_performance.CB_InputIF', new_callable=lambda: MockCB_InputIF)
    def test_evaluate_single_sample_anomaly(self, mock_cb_input, mock_adr, mock_vlm_detector):
        """単一サンプル評価のテスト（異常ケース）"""
        mock_vlm_detector.return_value = self.vlm_detector_mock
        
        # VLMDetectorのレスポンスを設定
        self.vlm_detector_mock.responses["default"] = MockCB_OutputIF(
            MockADR.ANOMALY, 2, "Relocation anomaly detected"
        )
        
        evaluator = VLMPerformanceEvaluator(
            dataset_base_path=str(self.dataset_path),
            verbose=False
        )
        
        result = evaluator.evaluate_single_sample("relocate", "001")
        
        self.assertIsInstance(result, EvaluationResult)
        self.assertEqual(result.dataset_category, "relocate")
        self.assertEqual(result.episode_id, "001")
        self.assertEqual(result.expected_result, "ANOMALY")
        self.assertEqual(result.actual_result, "ANOMALY")
        self.assertTrue(result.is_correct)
        self.assertEqual(result.expected_action_id, 2)
        self.assertEqual(result.actual_action_id, 2)
        self.assertTrue(result.action_id_correct)
    
    @patch('evaluate_vlm_performance.VLMDetector')
    @patch('evaluate_vlm_performance.ADR', new_callable=lambda: MockADR())
    @patch('evaluate_vlm_performance.CB_InputIF', new_callable=lambda: MockCB_InputIF)
    def test_evaluate_single_sample_error(self, mock_cb_input, mock_adr, mock_vlm_detector):
        """単一サンプル評価のテスト（エラーケース）"""
        mock_vlm_detector.return_value = self.vlm_detector_mock
        
        evaluator = VLMPerformanceEvaluator(
            dataset_base_path=str(self.dataset_path),
            verbose=False
        )
        
        # 存在しないエピソードでエラーを発生させる
        result = evaluator.evaluate_single_sample("normal", "999")
        
        self.assertIsInstance(result, EvaluationResult)
        self.assertEqual(result.dataset_category, "normal")
        self.assertEqual(result.episode_id, "999")
        self.assertEqual(result.actual_result, "ERROR")
        self.assertFalse(result.is_correct)
        self.assertIn("Error:", result.reason)
    
    @patch('evaluate_vlm_performance.VLMDetector')
    @patch('evaluate_vlm_performance.ADR', new_callable=lambda: MockADR())
    @patch('evaluate_vlm_performance.CB_InputIF', new_callable=lambda: MockCB_InputIF)
    def test_evaluate_category(self, mock_cb_input, mock_adr, mock_vlm_detector):
        """カテゴリ評価のテスト"""
        mock_vlm_detector.return_value = self.vlm_detector_mock
        
        # VLMDetectorのレスポンスを設定
        self.vlm_detector_mock.responses["default"] = MockCB_OutputIF(
            MockADR.NORMAL, 0, "Normal state detected"
        )
        
        evaluator = VLMPerformanceEvaluator(
            dataset_base_path=str(self.dataset_path),
            verbose=False
        )
        
        results = evaluator.evaluate_category("normal", max_samples=1)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(len(evaluator.evaluation_results), 1)
        self.assertEqual(results[0].dataset_category, "normal")
    
    @patch('evaluate_vlm_performance.VLMDetector')
    @patch('evaluate_vlm_performance.ADR', new_callable=lambda: MockADR())
    @patch('evaluate_vlm_performance.CB_InputIF', new_callable=lambda: MockCB_InputIF)
    def test_calculate_metrics(self, mock_cb_input, mock_adr, mock_vlm_detector):
        """メトリクス計算のテスト"""
        mock_vlm_detector.return_value = self.vlm_detector_mock
        
        evaluator = VLMPerformanceEvaluator(
            dataset_base_path=str(self.dataset_path),
            verbose=False
        )
        
        # テスト用の評価結果を追加
        evaluator.evaluation_results = [
            EvaluationResult(
                dataset_category="normal",
                episode_id="001",
                phase="static",
                expected_result="NORMAL",
                actual_result="NORMAL",
                is_correct=True,
                processing_time=1.0,
                reason="",
                expected_action_id=0,
                actual_action_id=0,
                action_id_correct=True
            ),
            EvaluationResult(
                dataset_category="relocate",
                episode_id="001",
                phase="static",
                expected_result="ANOMALY",
                actual_result="ANOMALY",
                is_correct=True,
                processing_time=2.0,
                reason="",
                expected_action_id=2,
                actual_action_id=2,
                action_id_correct=True
            ),
            EvaluationResult(
                dataset_category="relocate",
                episode_id="002",
                phase="static",
                expected_result="ANOMALY",
                actual_result="NORMAL",
                is_correct=False,
                processing_time=1.5,
                reason="",
                expected_action_id=2,
                actual_action_id=None,
                action_id_correct=False
            )
        ]
        
        metrics = evaluator.calculate_metrics()
        
        self.assertIsInstance(metrics, PerformanceMetrics)
        self.assertEqual(metrics.total_samples, 3)
        self.assertEqual(metrics.correct_predictions, 2)
        self.assertAlmostEqual(metrics.accuracy, 2/3, places=3)
        self.assertAlmostEqual(metrics.avg_processing_time, 1.5, places=3)
        
        # カテゴリ別精度の確認
        self.assertAlmostEqual(metrics.category_accuracies["normal"], 1.0)
        self.assertAlmostEqual(metrics.category_accuracies["relocate"], 0.5)
        
        # アクションID精度の確認
        self.assertAlmostEqual(metrics.action_id_accuracy, 2/3, places=3)
        self.assertAlmostEqual(metrics.action_id_metrics["normal"], 1.0)
        self.assertAlmostEqual(metrics.action_id_metrics["relocate"], 0.5)
    
    @patch('evaluate_vlm_performance.VLMDetector')
    @patch('evaluate_vlm_performance.ADR', new_callable=lambda: MockADR())
    @patch('evaluate_vlm_performance.CB_InputIF', new_callable=lambda: MockCB_InputIF)
    def test_save_results(self, mock_cb_input, mock_adr, mock_vlm_detector):
        """結果保存のテスト"""
        mock_vlm_detector.return_value = self.vlm_detector_mock
        
        evaluator = VLMPerformanceEvaluator(
            dataset_base_path=str(self.dataset_path),
            verbose=False
        )
        
        # テスト用の評価結果を追加
        evaluator.evaluation_results = [
            EvaluationResult(
                dataset_category="normal",
                episode_id="001",
                phase="static",
                expected_result="NORMAL",
                actual_result="NORMAL",
                is_correct=True,
                processing_time=1.0,
                reason="",
                expected_action_id=0,
                actual_action_id=0,
                action_id_correct=True
            )
        ]
        
        # 一時ファイルに保存
        output_path = Path(self.temp_dir) / "test_results.json"
        saved_path = evaluator.save_results(str(output_path))
        
        self.assertTrue(output_path.exists())
        self.assertEqual(saved_path, output_path)
        
        # 保存された内容を確認
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.assertIn("metadata", data)
        self.assertIn("evaluation_results", data)
        self.assertIn("performance_metrics", data)
        self.assertIn("detailed_analysis", data)
        
        # メタデータの確認
        self.assertEqual(data["metadata"]["total_samples"], 1)
        self.assertEqual(data["metadata"]["categories_evaluated"], ["normal"])
        self.assertEqual(data["metadata"]["phases_evaluated"], ["static"])
    
    @patch('evaluate_vlm_performance.VLMDetector')
    @patch('evaluate_vlm_performance.ADR', new_callable=lambda: MockADR())
    @patch('evaluate_vlm_performance.CB_InputIF', new_callable=lambda: MockCB_InputIF)
    def test_action_id_mapping(self, mock_cb_input, mock_adr, mock_vlm_detector):
        """アクションIDマッピングのテスト"""
        mock_vlm_detector.return_value = self.vlm_detector_mock
        
        evaluator = VLMPerformanceEvaluator(
            dataset_base_path=str(self.dataset_path),
            verbose=False
        )
        
        # アクションIDマッピングの確認
        self.assertEqual(evaluator.category_action_mapping["normal"], 0)
        self.assertEqual(evaluator.category_action_mapping["complete"], 0)
        self.assertEqual(evaluator.category_action_mapping["relocate"], 2)
        self.assertEqual(evaluator.category_action_mapping["unstack"], 1)
    
    @patch('evaluate_vlm_performance.VLMDetector')
    @patch('evaluate_vlm_performance.ADR', new_callable=lambda: MockADR())
    @patch('evaluate_vlm_performance.CB_InputIF', new_callable=lambda: MockCB_InputIF)
    def test_evaluate_all_categories(self, mock_cb_input, mock_adr, mock_vlm_detector):
        """全カテゴリ評価のテスト"""
        mock_vlm_detector.return_value = self.vlm_detector_mock
        
        # VLMDetectorのレスポンスを設定
        self.vlm_detector_mock.responses["default"] = MockCB_OutputIF(
            MockADR.NORMAL, 0, "Normal state detected"
        )
        
        evaluator = VLMPerformanceEvaluator(
            dataset_base_path=str(self.dataset_path),
            verbose=False
        )
        
        metrics = evaluator.evaluate_all_categories(
            max_samples_per_category=1,
            categories=["normal", "complete"]
        )
        
        self.assertIsInstance(metrics, PerformanceMetrics)
        self.assertEqual(len(evaluator.evaluation_results), 2)  # 2カテゴリ × 1サンプル
        
        # 評価されたカテゴリが正しいか確認
        evaluated_categories = set(r.dataset_category for r in evaluator.evaluation_results)
        self.assertEqual(evaluated_categories, {"normal", "complete"})


class TestMockData(unittest.TestCase):
    """モックデータとヘルパー関数のテスト"""
    
    def test_mock_adr(self):
        """MockADR のテスト"""
        adr = MockADR()
        self.assertEqual(adr.NORMAL.value, "NORMAL")
        self.assertEqual(adr.COMPLETION.value, "COMPLETION")
        self.assertEqual(adr.ANOMALY.value, "ANOMALY")
    
    def test_mock_cb_input_if(self):
        """MockCB_InputIF のテスト"""
        images = [np.zeros((100, 100, 3))]
        input_data = MockCB_InputIF(images=images, action_id=1)
        self.assertEqual(input_data.images, images)
        self.assertEqual(input_data.action_id, 1)
    
    def test_mock_cb_output_if(self):
        """MockCB_OutputIF のテスト"""
        output_data = MockCB_OutputIF(
            detection_result=MockADR.ANOMALY,
            action_id=2,
            reason="Test reason"
        )
        self.assertEqual(output_data.detection_result, MockADR.ANOMALY)
        self.assertEqual(output_data.action_id, 2)
        self.assertEqual(output_data.reason, "Test reason")
    
    def test_mock_vlm_detector(self):
        """MockVLMDetector のテスト"""
        detector = MockVLMDetector()
        
        # レスポンスを設定
        detector.set_response("normal", "001", MockADR.NORMAL, 0, "Normal")
        
        # レスポンスを確認
        key = "normal/001"
        self.assertIn(key, detector.responses)
        response = detector.responses[key]
        self.assertEqual(response.detection_result, MockADR.NORMAL)
        self.assertEqual(response.action_id, 0)
        self.assertEqual(response.reason, "Normal")


if __name__ == "__main__":
    # テストの実行
    unittest.main(verbosity=2)
