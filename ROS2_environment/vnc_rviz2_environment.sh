#!/bin/bash
# VNC + RViz2 起動スクリプト

# 環境変数設定
export USER=root
export DISPLAY=:1
export XDG_RUNTIME_DIR=/tmp/runtime-root
mkdir -p $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR

# 既存プロセスを停止
pkill -f Xvfb 2>/dev/null || true
pkill -f x11vnc 2>/dev/null || true

# 仮想ディスプレイを起動
Xvfb :1 -screen 0 1920x1080x24 &
sleep 2

# VNC共有を開始
x11vnc -display :1 -nopw -listen localhost -xkb -forever &
sleep 2

echo "VNC接続先: localhost:5900"
echo "RViz2を起動するには: ros2 run rviz2 rviz2"