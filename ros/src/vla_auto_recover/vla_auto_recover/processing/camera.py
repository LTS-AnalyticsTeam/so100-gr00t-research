import cv2
import os
import time
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple
import glob
import datetime

CAMRERA_DIR = Path("/workspace/ros/src/vla_auto_recover/vla_auto_recover/processing/__camera_images__")


class CameraSaver:
    """
    OpenCVを使用してcenter_camとright_camから画像を取得し、
    タイムスタンプ付きでファイル保存し、古いファイルを自動削除するクラス
    """
    
    def __init__(self, 
                 center_cam_id: int = 0, 
                 right_cam_id: int = 2,
                 output_dir: str = str(CAMRERA_DIR),
                 max_files: int = 100,
                 fps: int = 30):
        """
        Args:
            center_cam_id: center_camのデバイスID
            right_cam_id: right_camのデバイスID  
            output_dir: 画像保存先ディレクトリ
            max_files: 各カメラディレクトリの最大ファイル数
            fps: カメラのフレームレート（FPS）
        """
        self.center_cam_id = center_cam_id
        self.right_cam_id = right_cam_id
        self.max_files = max_files
        self.fps = fps
        
        # 出力ディレクトリの設定
        self.output_dir = Path(output_dir)
        self.center_cam_dir = self.output_dir / "center_cam"
        self.right_cam_dir = self.output_dir / "right_cam"
        
        # ディレクトリ作成
        self.center_cam_dir.mkdir(parents=True, exist_ok=True)
        self.right_cam_dir.mkdir(parents=True, exist_ok=True)
        
        # カメラオブジェクト
        self.cameras: Dict[str, cv2.VideoCapture] = {}
        self.running = False
        self.capture_thread = None
        
    def initialize_cameras(self) -> bool:
        """カメラを初期化"""
        try:
            # center_cam初期化
            center_cap = cv2.VideoCapture(self.center_cam_id)
            if not center_cap.isOpened():
                print(f"Error: Cannot open center_cam (ID: {self.center_cam_id})")
                return False
            
            # center_camのFPS設定
            center_cap.set(cv2.CAP_PROP_FPS, self.fps)
            actual_fps_center = center_cap.get(cv2.CAP_PROP_FPS)
            print(f"Center cam FPS set to: {actual_fps_center}")
            
            self.cameras['center_cam'] = center_cap
            
            # right_cam初期化
            right_cap = cv2.VideoCapture(self.right_cam_id)
            if not right_cap.isOpened():
                print(f"Error: Cannot open right_cam (ID: {self.right_cam_id})")
                center_cap.release()
                return False
            
            # right_camのFPS設定
            right_cap.set(cv2.CAP_PROP_FPS, self.fps)
            actual_fps_right = right_cap.get(cv2.CAP_PROP_FPS)
            print(f"Right cam FPS set to: {actual_fps_right}")
            
            self.cameras['right_cam'] = right_cap
            
            print("Cameras initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing cameras: {e}")
            return False
    
    def capture_frames(self) -> Tuple[Optional[cv2.Mat], Optional[cv2.Mat]]:
        """両カメラから同時にフレームを取得"""
        center_frame = None
        right_frame = None
        
        if 'center_cam' in self.cameras:
            ret_center, center_frame = self.cameras['center_cam'].read()
            if not ret_center:
                center_frame = None
                
        if 'right_cam' in self.cameras:
            ret_right, right_frame = self.cameras['right_cam'].read()
            if not ret_right:
                right_frame = None
                
        return center_frame, right_frame
    
    def generate_timestamp(self) -> str:
        """タイムスタンプを生成"""
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # ミリ秒まで
    
    def save_images(self, center_frame: cv2.Mat, right_frame: cv2.Mat, timestamp: str):
        """画像を保存"""
        saved_files = []
        
        if center_frame is not None:
            center_filename = self.center_cam_dir / f"{timestamp}.jpg"
            cv2.imwrite(str(center_filename), center_frame)
            saved_files.append(str(center_filename))
            
        if right_frame is not None:
            right_filename = self.right_cam_dir / f"{timestamp}.jpg"
            cv2.imwrite(str(right_filename), right_frame)
            saved_files.append(str(right_filename))
            
        return saved_files
    
    def cleanup_old_files(self):
        """古いファイルを削除して最大ファイル数を維持"""
        for cam_dir in [self.center_cam_dir, self.right_cam_dir]:
            # jpg ファイルのリストを取得
            image_files = list(cam_dir.glob("*.jpg"))
            
            if len(image_files) > self.max_files:
                # ファイル名（タイムスタンプ）でソート
                image_files.sort(key=lambda x: x.name)
                
                # 古いファイルを削除
                files_to_delete = image_files[:-self.max_files]
                for file_path in files_to_delete:
                    try:
                        file_path.unlink()
                        print(f"Deleted old file: {file_path}")
                    except OSError as e:
                        print(f"Error deleting file {file_path}: {e}")
    
    def capture_single_frame(self) -> bool:
        """単一フレームをキャプチャして保存"""
        center_frame, right_frame = self.capture_frames()
        
        if center_frame is None and right_frame is None:
            print("Error: No frames captured from either camera")
            return False
        
        # 同じタイムスタンプで保存
        timestamp = self.generate_timestamp()
        saved_files = self.save_images(center_frame, right_frame, timestamp)
        
        if saved_files:
            print(f"Saved images with timestamp {timestamp}:")
            for file_path in saved_files:
                print(f"  - {file_path}")
        
        # 古いファイルを削除
        self.cleanup_old_files()
        
        return True
    
    def start_continuous_capture(self, interval: float = 1.0):
        """連続キャプチャを開始"""
        if self.running:
            print("Continuous capture is already running")
            return
        
        self.running = True
        self.capture_thread = threading.Thread(
            target=self._continuous_capture_loop, 
            args=(interval,),
            daemon=True
        )
        self.capture_thread.start()
        print(f"Started continuous capture with {interval}s interval")
    
    def stop_continuous_capture(self):
        """連続キャプチャを停止"""
        if not self.running:
            print("Continuous capture is not running")
            return
        
        self.running = False
        if self.capture_thread:
            self.capture_thread.join()
        print("Stopped continuous capture")
    
    def _continuous_capture_loop(self, interval: float):
        """連続キャプチャのメインループ"""
        while self.running:
            self.capture_single_frame()
            time.sleep(interval)
    
    def release_cameras(self):
        """カメラリソースを解放"""
        for name, camera in self.cameras.items():
            if camera.isOpened():
                camera.release()
                print(f"Released {name}")
        self.cameras.clear()
    
    def __del__(self):
        """デストラクタ"""
        self.stop_continuous_capture()
        self.release_cameras()


def main(interval=0.1, period_sec=60, output_dir=str(CAMRERA_DIR)):
    """使用例"""
    # カメラセーバーを初期化
    camera_saver = CameraSaver(output_dir=output_dir)
    
    # カメラを初期化
    if not camera_saver.initialize_cameras():
        print("Failed to initialize cameras")
        return
    
    try:
        # 単一フレームキャプチャの例
        print("Capturing single frame...")
        camera_saver.capture_single_frame()
        
        # 連続キャプチャの例（5秒間、1秒間隔）
        print(f"Starting continuous capture for {period_sec} seconds...")
        camera_saver.start_continuous_capture(interval=interval)
        time.sleep(period_sec)
        camera_saver.stop_continuous_capture()
        
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        camera_saver.release_cameras()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Camera Saver Example")
    parser.add_argument("--interval", type=float, default=0.1, help="Capture interval in seconds")
    parser.add_argument("--period_sec", type=int, default=60, help="Total capture period in seconds")
    parser.add_argument("--output_dir", type=str, default=str(CAMRERA_DIR), help="Output directory for saved images")
    args = parser.parse_args()
    main(args.interval, args.period_sec, args.output_dir)