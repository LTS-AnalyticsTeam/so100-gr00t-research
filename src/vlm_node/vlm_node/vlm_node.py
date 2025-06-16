#!/usr/bin/env python3
import os
import sys
import argparse
import glob
import base64
import threading
from datetime import datetime
from dotenv import load_dotenv
from queue import Queue, Empty

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def parse_args():
    parser = argparse.ArgumentParser(description='VLM Node with input and file output')
    parser.add_argument('--input-dir',    required=True,
                        help='画像データのディレクトリパス')
    parser.add_argument('--prompt-file',  required=True,
                        help='プロンプト定義ファイル（テキスト）へのパス')
    parser.add_argument('--output-file',  required=True,
                        help='推論結果を追記していくログファイルのパス')
    return parser.parse_args()


class AsyncVLMNode(Node):
    def __init__(self, args):
        super().__init__('vlm_node')

        self.get_logger().info(f'Starting AsyncVLMNode with input_dir={args.input_dir}')

        # トピック定義
        self.publisher = self.create_publisher(String, 'anomaly_type', 10)

        # 画像撮影タイマー：100ms周期（VLA同期用）
        self.capture_timer = self.create_timer(0.1, self.on_capture_timer)
        
        # 結果処理タイマー：100ms周期
        self.result_timer = self.create_timer(0.1, self.on_result_timer)

        # OpenAI クライアント初期化
        self.openai_client = self._init_openai()

        # 画像ファイルリスト
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
        self.files = []
        for ext in image_extensions:
            self.files.extend(glob.glob(os.path.join(args.input_dir, ext)))
        self.files = sorted(self.files)
        
        if not self.files:
            self.get_logger().warning(f'No image files found in input_dir: {args.input_dir}')
        else:
            self.get_logger().info(f'Found {len(self.files)} image files')
        
        self.index = 0

        # プロンプト読み込み
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            self.prompt = f.read().strip()

        self.out_fp = args.output_file

        # 非同期処理用キュー
        self.processing_queue = Queue()
        self.result_queue = Queue()
        self.last_analysis_time = 0
        self.analysis_interval = 0.5  # 500msに1回分析実行

        # 複数ワーカーで並列処理
        self.num_workers = 4
        for _ in range(self.num_workers):
            t = threading.Thread(target=self._async_processor, daemon=True)
            t.start()

    def _init_openai(self):
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                self.get_logger().info('OpenAI API client initialized')
                return client
            except ImportError:
                self.get_logger().error('openai package not found')
                return None
        else:
            self.get_logger().warning('OPENAI_API_KEY not set')
            return None

    def encode_image(self, image_path):
        """画像をbase64エンコードする（サイズ最適化付き）"""
        try:
            from PIL import Image
            import io
            
            # 画像を読み込み、サイズを最適化
            with Image.open(image_path) as img:
                # 最大幅を制限（API効率化）
                max_width = 512
                if img.width > max_width:
                    ratio = max_width / img.width
                    new_height = int(img.height * ratio)
                    img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
                
                # JPEG形式で保存（圧縮）
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=70)
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
                
        except ImportError:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            self.get_logger().error(f'Failed to encode image {image_path}: {e}')
            return None

    def on_capture_timer(self):
        """100ms周期：VLA同期用の高頻度監視"""
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        # 500msに1回だけ実際の分析をキューに追加
        if current_time - self.last_analysis_time > self.analysis_interval:
            if not self.files:
                return
            
            file_path = self.files[self.index % len(self.files)]
            self.index += 1
            
            # 分析タスクをキューに追加（非ブロッキング）
            if self.processing_queue.qsize() < self.num_workers * 2:
                self.processing_queue.put((file_path, current_time))
                self.last_analysis_time = current_time
                self.get_logger().info(f'Queued analysis for: {os.path.basename(file_path)}')

    def on_result_timer(self):
        """100ms周期：結果処理"""
        while not self.result_queue.empty():
            try:
                filename, anomaly, timestamp = self.result_queue.get_nowait()
                
                # トピックPublish
                msg = String()
                msg.data = anomaly
                self.publisher.publish(msg)
                self.get_logger().info(f'Published result for {filename}')

                # ファイル出力
                os.makedirs(os.path.dirname(self.out_fp), exist_ok=True)
                with open(self.out_fp, 'a', encoding='utf-8') as fw:
                    fw.write(f'{timestamp}\t{filename}\t{anomaly}\n')
            except Exception as e:
                self.get_logger().error(f'Error processing result: {e}')

    def _async_processor(self):
        """バックグラウンドでOpenAI API処理"""
        while True:
            try:
                file_path, request_time = self.processing_queue.get(timeout=1.0)
            except Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Unexpected error getting task: {e}')
                continue

            filename = os.path.basename(file_path)
            start_time = datetime.now()
            self.get_logger().info(f'Starting analysis: {filename}')

            if self.openai_client:
                anomaly = self._analyze_with_openai(file_path)
            else:
                anomaly = f"No API client - dummy result for {filename}"

            duration = (datetime.now() - start_time).total_seconds()
            self.get_logger().info(f'Analysis completed: {filename} ({duration:.2f}s)')
            timestamp = datetime.now().isoformat()
            self.result_queue.put((filename, anomaly, timestamp))

    def _analyze_with_openai(self, image_path):
        """OpenAI APIで画像分析"""
        base64_image = self.encode_image(image_path)
        if not base64_image:
            return "Failed to encode image"

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text",  "text": self.prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"API Error: {str(e)}"


def main(args=None):
    parsed = parse_args()
    rclpy.init(args=args)
    node = AsyncVLMNode(parsed)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    sys.argv = [sys.argv[0]] + rclpy.utilities.remove_ros_args(sys.argv[1:])
    main()
