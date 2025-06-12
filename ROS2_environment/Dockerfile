# docker/Dockerfile
FROM osrf/ros:jazzy-desktop-full

# ビルド時に渡せるVNCパスワード
ARG VNC_PASSWORD
# タイムゾーン＆Python仮想環境ディレクトリ
ENV TZ=Asia/Tokyo \
    VENV_DIR=/opt/venv

# タイムゾーン設定
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# システム更新＋パッケージ一括インストール
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      tightvncserver \
      xfce4 \
      xfce4-goodies \
      firefox \
      python3-pip \
      python3-venv \
      pipx \
      git \
      curl \
      vim \
      wget \
      gedit \
      terminator \
      catkin-pkg \
      python-dotenv \
      openai \
    && rm -rf /var/lib/apt/lists/*

# Python仮想環境作成＆パッケージインストール
RUN python3 -m venv $VENV_DIR && \
    . $VENV_DIR/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    pip install lerobot && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 追加のROS2ツール（Jazzy用）
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      ros-jazzy-rqt* \
      ros-jazzy-rviz2 \
      ros-jazzy-demo-nodes-cpp \
      ros-jazzy-demo-nodes-py \
    && rm -rf /var/lib/apt/lists/*

# VNC設定ディレクトリ＆パスワードファイル
RUN mkdir -p /root/.vnc && \
    echo "${VNC_PASSWORD:-changeme}" | vncpasswd -f > /root/.vnc/passwd && \
    chmod 600 /root/.vnc/passwd

# VNC xstartup（XFCE起動スクリプト）
RUN echo "#!/bin/bash\nxrdb \$HOME/.Xresources\nxsetroot -solid grey\n/usr/bin/startxfce4 &" \
     > /root/.vnc/xstartup && \
    chmod +x /root/.vnc/xstartup

# VNC起動用スクリプトをコピー
COPY vnc_startup.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/vnc_startup.sh

# ROS2ワークスペース準備
WORKDIR /workspace
RUN mkdir -p src

# コンテナ起動時に自動でROS2環境セット
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc && \
    echo "cd /workspace" >> /root/.bashrc && \
    echo "echo 'ROS2 Jazzy + lerobot development environment ready!'" >> /root/.bashrc

EXPOSE 5901
CMD ["/usr/local/bin/vnc_startup.sh"]
