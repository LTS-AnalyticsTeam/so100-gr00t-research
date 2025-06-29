#!/bin/bash
set -e

echo "=== VLA Auto Recover GUI Environment ==="

# シンプルなxstartupスクリプトを作成
mkdir -p $HOME/.vnc
cat > "$HOME/.vnc/xstartup" << 'EOF'
#!/bin/bash
export DISPLAY=:1
xrdb $HOME/.Xresources 2>/dev/null || true
xsetroot -solid grey
# バックグラウンドでopenboxを起動
openbox &
# フォアグラウンドでxtermを起動（セッションを維持）
exec xterm -geometry 80x24+100+100
EOF
chmod +x "$HOME/.vnc/xstartup"

echo "Starting VNC server (no password)..."

# VNCサーバーを停止（既存があれば）
tigervncserver -kill :1 2>/dev/null || true

# パスワードなしでVNCサーバーを起動 - シンプルなターミナルで
tigervncserver :1 -geometry 1920x1080 -depth 24 -localhost no \
    -SecurityTypes None --I-KNOW-THIS-IS-INSECURE \
    -xstartup /usr/bin/lxterminal

echo "VNC Server started on :1 (no password required)"
echo "Connect with VNC Viewer to localhost:5901"

# noVNC起動
if command -v websockify >/dev/null 2>&1; then
    websockify --web=/usr/share/novnc/ 6080 localhost:5901 &
    echo "noVNC available at: http://localhost:6080/vnc.html"
fi

# ROS2環境設定
source /opt/ros/jazzy/setup.bash
source /opt/venv/bin/activate

# ワークスペースに移動
cd /workspace

# ワークスペースが初期化されていない場合の初期セットアップ
if [ ! -f "ros/install/setup.bash" ]; then
    echo "🔧 初回セットアップ中..."
    if [ -d "ros/src" ] && [ "$(ls -A ros/src)" ]; then
        cd ros
        colcon build
        echo "ROS Workspace built successfully"
        cd /workspace
    fi
fi

# ROSワークスペースをセットアップ（存在する場合）
if [ -f "ros/install/setup.bash" ]; then
    echo "🚀 ROSワークスペースをセットアップ中..."
    source ros/install/setup.bash
    echo "ROS workspace sourced"
fi

echo "🎉 VLA Auto Recover GUI environment ready!"
echo "   VNC: localhost:5901"
echo "   Web: http://localhost:6080/vnc.html"
echo ""

# 環境設定の表示
echo "Environment variables:"
echo "  ROS_DOMAIN_ID: $ROS_DOMAIN_ID"
echo "  DISPLAY: $DISPLAY"
echo "  PYTHONPATH: $PYTHONPATH"
echo ""
