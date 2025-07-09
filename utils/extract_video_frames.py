#!/usr/bin/env python
"""
動画フォルダから指定フレームを画像として抽出するユーティリティ

Usage:
  # 指定フレーム（10フレーム目）を抽出
  python extract_video_frames.py --input-dir ./datasets/row_datasets/lt-s__*/videos/chunk-000 \
      --output-dir ./extracted_frames --frame-mode specific --frame-number 10

  # ランダムフレームを抽出
  python extract_video_frames.py --input-dir ./datasets/row_datasets/lt-s__*/videos/chunk-000 \
      --output-dir ./extracted_frames --frame-mode random

  # 最初のフレームを抽出
  python extract_video_frames.py --input-dir ./datasets/row_datasets/lt-s__*/videos/chunk-000 \
      --output-dir ./extracted_frames --frame-mode first

  # 最後のフレームを抽出
  python extract_video_frames.py --input-dir ./datasets/row_datasets/lt-s__*/videos/chunk-000 \
      --output-dir ./extracted_frames --frame-mode last
"""
import argparse
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2


def find_video_pairs(input_dir: Path) -> Dict[str, Dict[str, Path]]:
    """
    入力ディレクトリから動画ペアを検索する
    
    Returns:
        Dict[episode_name, Dict[camera_name, video_path]]
    """
    video_pairs = {}
    
    # サブディレクトリを探索
    for subdir in input_dir.iterdir():
        if not subdir.is_dir():
            continue
            
        camera_name = subdir.name
        
        # 動画ファイルを検索
        for video_file in subdir.glob("*.mp4"):
            episode_name = video_file.stem  # 拡張子を除いたファイル名
            
            if episode_name not in video_pairs:
                video_pairs[episode_name] = {}
            
            video_pairs[episode_name][camera_name] = video_file
    
    return video_pairs


def extract_frame_with_ffmpeg(video_path: Path, frame_mode: str, frame_number: int = None) -> Tuple[bool, str, str]:
    """
    FFmpegを使用して動画から指定されたフレームを抽出する
    
    Args:
        video_path: 動画ファイルのパス
        frame_mode: フレーム抽出モード（'first', 'last', 'specific', 'random'）
        frame_number: 特定フレーム番号（frame_mode='specific'の場合）
    
    Returns:
        (success, temp_image_path, error_message) のタプル
    """
    try:
        # 一時ファイル名を生成
        temp_image = f"/tmp/frame_{os.getpid()}_{random.randint(1000,9999)}.png"
        
        # フレームモードに応じてFFmpegコマンドを構築
        if frame_mode == "first":
            # 最初のフレーム: 0秒から1フレーム抽出
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-v', 'error',  # エラーメッセージのみ表示
                '-i', str(video_path),
                '-vf', 'select=eq(n\\,0)',  # 最初のフレームを選択
                '-vsync', 'vfr',  # 可変フレームレート対応
                '-frames:v', '1',
                '-q:v', '2',
                temp_image
            ]
        elif frame_mode == "last":
            # 最後のフレーム: 複数のアプローチを試行
            # アプローチ1: 動画を逆順で読み込んで最初のフレームを取得
            ffmpeg_cmd_reverse = [
                'ffmpeg', '-y', '-v', 'error',
                '-i', str(video_path),
                '-vf', 'reverse,select=eq(n\\,0)',  # 逆順にして最初のフレーム
                '-vsync', 'vfr',
                '-frames:v', '1',
                '-q:v', '2',
                temp_image
            ]
            
            # まず逆順アプローチを試行
            result = subprocess.run(ffmpeg_cmd_reverse, capture_output=True, text=True, timeout=30)
            
            # 逆順が失敗した場合、別のアプローチを試行
            if result.returncode != 0 or not Path(temp_image).exists() or Path(temp_image).stat().st_size == 0:
                # アプローチ2: 動画の長さを取得して終端近くから抽出
                duration_cmd = [
                    'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)
                ]
                dur_result = subprocess.run(duration_cmd, capture_output=True, text=True)
                
                if dur_result.returncode == 0:
                    try:
                        duration = float(dur_result.stdout.strip())
                        # 最後の0.1秒前からシークして複数フレームから最後を選択
                        seek_time = max(0, duration - 0.1)
                        ffmpeg_cmd = [
                            'ffmpeg', '-y', '-v', 'error',
                            '-ss', f"{seek_time:.3f}",
                            '-i', str(video_path),
                            '-frames:v', '1',
                            '-q:v', '2',
                            temp_image
                        ]
                    except (ValueError, TypeError):
                        # フォールバック: 終端から3秒前
                        ffmpeg_cmd = [
                            'ffmpeg', '-y', '-v', 'error',
                            '-sseof', '-1',  # 終了1秒前からシーク
                            '-i', str(video_path),
                            '-frames:v', '1',
                            '-q:v', '2',
                            temp_image
                        ]
                else:
                    # フォールバック: 終端から1秒前
                    ffmpeg_cmd = [
                        'ffmpeg', '-y', '-v', 'error',
                        '-sseof', '-1',
                        '-i', str(video_path),
                        '-frames:v', '1',
                        '-q:v', '2',
                        temp_image
                    ]
                
                # フォールバックコマンドを実行
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=30)
        elif frame_mode == "random":
            # ランダムフレーム: まず動画の長さを取得
            duration_cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)
            ]
            result = subprocess.run(duration_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return False, "", f"Failed to get video duration: {result.stderr}"
            
            try:
                duration = float(result.stdout.strip())
                # ランダムな時間を生成（最初と最後の1秒は除く）
                random_time = random.uniform(1.0, max(1.0, duration - 1.0))
            except (ValueError, TypeError):
                random_time = 1.0  # フォールバック
            
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-v', 'error',
                '-ss', f"{random_time:.3f}",
                '-i', str(video_path),
                '-frames:v', '1',
                '-q:v', '2',
                temp_image
            ]
        elif frame_mode == "specific":
            if frame_number is None:
                raise ValueError("frame_number must be specified for 'specific' mode")
            
            # 特定フレーム: フレーム番号を直接指定
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-v', 'error',
                '-i', str(video_path),
                '-vf', f'select=eq(n\\,{frame_number})',
                '-vsync', 'vfr',
                '-frames:v', '1',
                '-q:v', '2',
                temp_image
            ]
        else:
            raise ValueError(f"Invalid frame_mode: {frame_mode}")
        
        # FFmpegコマンドを実行
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and Path(temp_image).exists() and Path(temp_image).stat().st_size > 0:
            return True, temp_image, ""
        else:
            # エラーの詳細を取得
            error_msg = result.stderr if result.stderr else "Unknown FFmpeg error"
            return False, "", f"FFmpeg failed: {error_msg}"
            
    except subprocess.TimeoutExpired:
        return False, "", "FFmpeg command timed out"
    except Exception as e:
        return False, "", f"Exception in FFmpeg extraction: {str(e)}"


def extract_frame_with_ffmpeg_alternative(video_path: Path) -> Tuple[bool, str, str]:
    """
    最後のフレーム専用の代替抽出方法
    すべてのフレームを一度に抽出して最後のものを取得
    """
    try:
        temp_dir = f"/tmp/frames_{os.getpid()}_{random.randint(1000,9999)}"
        os.makedirs(temp_dir, exist_ok=True)
        
        # すべてのフレームを一時ディレクトリに抽出
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-v', 'error',
            '-i', str(video_path),
            '-q:v', '2',
            f"{temp_dir}/frame_%06d.png"
        ]
        
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            # 抽出されたフレームファイルを取得
            frame_files = sorted(Path(temp_dir).glob("frame_*.png"))
            if frame_files:
                # 最後のフレームファイルを取得
                last_frame = frame_files[-1]
                final_temp_image = f"/tmp/final_frame_{os.getpid()}_{random.randint(1000,9999)}.png"
                
                # 最後のフレームをコピー
                import shutil
                shutil.copy2(last_frame, final_temp_image)
                
                # 一時ディレクトリを削除
                shutil.rmtree(temp_dir)
                
                return True, final_temp_image, ""
            else:
                shutil.rmtree(temp_dir, ignore_errors=True)
                return False, "", "No frames extracted"
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return False, "", f"FFmpeg failed: {result.stderr}"
            
    except Exception as e:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        return False, "", f"Exception in alternative extraction: {str(e)}"


def extract_frame(video_path: Path, frame_mode: str, frame_number: int = None) -> Tuple[bool, any, str]:
    """
    動画から指定されたフレームを抽出する（FFmpeg優先、OpenCVフォールバック）
    
    Args:
        video_path: 動画ファイルのパス
        frame_mode: フレーム抽出モード（'first', 'last', 'specific', 'random'）
        frame_number: 特定フレーム番号（frame_mode='specific'の場合）
    
    Returns:
        (success, frame, error_message) のタプル
    """
    # まずFFmpegを試行
    success, temp_image, error_msg = extract_frame_with_ffmpeg(video_path, frame_mode, frame_number)
    
    # 最後のフレームで失敗した場合、代替方法を試行
    if not success and frame_mode == "last":
        print(f"    First attempt failed, trying alternative method for last frame...")
        success, temp_image, error_msg = extract_frame_with_ffmpeg_alternative(video_path)
    
    if success:
        try:
            # 一時画像ファイルを読み込み
            frame = cv2.imread(temp_image)
            # 一時ファイルを削除
            Path(temp_image).unlink(missing_ok=True)
            
            if frame is not None:
                return True, frame, ""
            else:
                return False, None, "Failed to load extracted image"
        except Exception as e:
            Path(temp_image).unlink(missing_ok=True)
            return False, None, f"Error loading extracted image: {str(e)}"
    
    # FFmpegが失敗した場合、OpenCVを試行
    print(f"    FFmpeg failed ({error_msg}), trying OpenCV...")
    
    # AV1コーデックの問題を回避するため、異なるバックエンドを試行
    backends = [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
    
    for backend in backends:
        cap = cv2.VideoCapture(str(video_path), backend)
        
        if not cap.isOpened():
            continue
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                cap.release()
                continue
            
            # フレーム番号を決定
            if frame_mode == "first":
                target_frame = 0
            elif frame_mode == "last":
                target_frame = total_frames - 1
            elif frame_mode == "random":
                target_frame = random.randint(0, total_frames - 1)
            elif frame_mode == "specific":
                if frame_number is None:
                    raise ValueError("frame_number must be specified for 'specific' mode")
                if frame_number >= total_frames:
                    print(f"    Warning: Frame {frame_number} is beyond video length ({total_frames}). Using last frame.")
                    target_frame = total_frames - 1
                else:
                    target_frame = frame_number
            else:
                raise ValueError(f"Invalid frame_mode: {frame_mode}")
            
            # 指定フレームに移動
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            
            if ret and frame is not None:
                return True, frame, ""
            else:
                cap.release()
                continue
                
        except Exception as e:
            cap.release()
            continue
        finally:
            if cap.isOpened():
                cap.release
    
    # すべての方法で失敗した場合
    return False, None, f"Failed with both FFmpeg ({error_msg}) and OpenCV"


def process_videos(input_dir: Path, output_dir: Path, frame_mode: str, frame_number: int = None) -> None:
    """
    動画ペアを処理してフレームを抽出する
    """
    # 出力ディレクトリを作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 動画ペアを検索
    video_pairs = find_video_pairs(input_dir)
    
    if not video_pairs:
        print("No video pairs found in the input directory.")
        return
    
    print(f"Found {len(video_pairs)} episode(s) to process.")
    
    # エピソード名でソートして連番フォルダを作成
    sorted_episodes = sorted(video_pairs.items())
    
    # 統計情報
    success_count = 0
    error_count = 0
    total_videos = sum(len(cameras) for _, cameras in video_pairs.items())
    
    # 各エピソードを処理
    for episode_idx, (episode_name, cameras) in enumerate(sorted_episodes):
        print(f"Processing {episode_name}...")
        
        # 連番サブフォルダを作成（例: 000, 001, 002, ...）
        episode_folder = output_dir / f"{episode_idx:03d}"
        episode_folder.mkdir(exist_ok=True)
        
        for camera_name, video_path in cameras.items():
            if not video_path.exists():
                print(f"  Warning: Video file not found: {video_path}")
                error_count += 1
                continue
            
            # フレームを抽出
            success, frame, error_msg = extract_frame(video_path, frame_mode, frame_number)
            
            if success:
                # 出力ファイル名を生成
                if frame_mode == "specific":
                    frame_suffix = f"frame_{frame_number:06d}"
                elif frame_mode == "random":
                    frame_suffix = "random"
                elif frame_mode == "first":
                    frame_suffix = "first"
                elif frame_mode == "last":
                    frame_suffix = "last"
                
                output_filename = f"{episode_name}_{camera_name}_{frame_suffix}.png"
                output_path = episode_folder / output_filename
                
                # 画像を保存
                cv2.imwrite(str(output_path), frame)
                print(f"  Saved: {episode_folder.name}/{output_filename}")
                success_count += 1
            else:
                print(f"  Error: Failed to extract frame from {video_path}")
                if error_msg:
                    print(f"    Reason: {error_msg}")
                print(f"    This might be due to AV1 codec compatibility issues.")
                error_count += 1
    
    # 結果サマリーを表示
    print(f"\n=== Processing Summary ===")
    print(f"Total videos: {total_videos}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {error_count}")
    if error_count > 0:
        print(f"Note: Some videos failed due to codec compatibility issues (likely AV1).")
        print(f"FFmpeg was used as primary method, OpenCV as fallback.")
        print(f"Consider updating FFmpeg if issues persist.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from video pairs")
    parser.add_argument("--input-dir", required=True, type=Path,
                       help="Input directory containing video subdirectories")
    parser.add_argument("--output-dir", required=True, type=Path,
                       help="Output directory for extracted frames")
    parser.add_argument("--frame-mode", required=True,
                       choices=["first", "last", "specific", "random"],
                       help="Frame extraction mode")
    parser.add_argument("--frame-number", type=int,
                       help="Specific frame number (required for 'specific' mode)")
    
    args = parser.parse_args()
    
    # 引数の検証
    if args.frame_mode == "specific" and args.frame_number is None:
        parser.error("--frame-number is required when --frame-mode is 'specific'")
    
    if not args.input_dir.exists():
        sys.exit(f"Error: Input directory does not exist: {args.input_dir}")
    
    # 動画処理を実行
    try:
        process_videos(args.input_dir, args.output_dir, args.frame_mode, args.frame_number)
        print("Frame extraction completed successfully.")
    except Exception as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
