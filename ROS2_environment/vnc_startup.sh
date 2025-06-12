# docker/vnc_startup.sh
#!/bin/bash

echo "=== ROS2 Jazzy + lerobot Development Environment ==="

# VNCサーバー停止
vncserver -kill :1 2>/dev/null || true

# X11の設定
export DISPLAY=:1

# VNCパスワードがセットまだなら再生成
if [ ! -f "$HOME/.vnc/passwd" ] && [ -n "$VNC_PASSWORD" ]; then
    mkdir -p "$HOME/.vnc"
    echo "$VNC_PASSWORD" | vncpasswd -f > "$HOME/.vnc/passwd"
    chmod 600 "$HOME/.vnc/passwd"
fi

# VNCサーバー起動
vncserver :1 -geometry 1920x1080 -depth 24 -localhost no

echo "VNC Server started on :1"
echo "Connect with VNC Viewer to localhost:5901"
echo "Password: ${VNC_PASSWORD:-<ビルド時デフォルト>}"

# ROS2環境設定
source /opt/ros/jazzy/setup.bash

# ワークスペースに移動
cd /workspace

# ワークスペースが初期化されていない場合の初期セットアップ
if [ ! -f "install/setup.bash" ]; then
    echo "🔧 初回セットアップ中..."
    if [ -d "src" ] && [ "$(ls -A src)" ]; then
        colcon build
        echo "Workspace built successfully"
    else
        echo "No packages found in src/ - workspace ready for development"
    fi
fi

# 仮想環境を有効化してlerobot動作確認
source $VENV_DIR/bin/activate
python3 -c "import lerobot; print('✅ lerobot imported successfully')" 2>/dev/null || echo "⚠️ lerobot import test failed"

echo ""
echo "Environment ready! Connect via VNC to start development."

# 無限ループでコンテナ維持
tail -f /dev/null
