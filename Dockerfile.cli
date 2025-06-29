FROM ros:jazzy-ros-base

# 基本的な依存関係をインストール
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-pip \
    git \
    build-essential \
    cmake \
    pkg-config \
    libffi-dev \
    python3-dev \
    python3-venv \
    curl \
    wget \
    ffmpeg \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# node可視化用の依存関係をインストール
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y universe && \
    apt-get update && \
    apt-get install -y nodejs npm && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


# 作業ディレクトリを設定
WORKDIR /workspace

# ROSの設定を追加
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc && \
    echo "source /opt/venv/bin/activate" >> /root/.bashrc && \
    echo "export ROS_DOMAIN_ID=0" >> /root/.bashrc

# システムパッケージの競合を避けるため、仮想環境を作成
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# pipを最新版にアップグレード
RUN pip install --upgrade pip setuptools wheel

# 基本的なPythonパッケージを先にインストール
RUN pip install numpy scipy

# LeRobotとその依存関係をインストール
RUN git clone https://github.com/huggingface/lerobot.git /tmp/lerobot && \
    cd /tmp/lerobot && \
    pip install -e ".[feetech]"

# 追加の要件をインストール
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 途中でインストールされるargparseを削除（build-inのargparseを利用すべき）
RUN pip uninstall -y argparse

# ROSワークスペースの設定
RUN mkdir -p /workspace/ros_ws/src
WORKDIR /workspace

# 環境変数を設定
ENV PYTHONPATH="/opt/venv/lib/python3.12/site-packages:$PYTHONPATH"
ENV ROS_DOMAIN_ID=0

# エントリーポイントスクリプトを作成
RUN echo '#!/bin/bash\nsource /opt/ros/jazzy/setup.bash\nsource /opt/venv/bin/activate\nexec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
