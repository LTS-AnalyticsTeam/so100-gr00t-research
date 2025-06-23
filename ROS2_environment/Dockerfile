# docker/Dockerfile
FROM osrf/ros:jazzy-desktop-full

##### ビルド引数・環境変数 ############################################
ARG VNC_PASSWORD=changeme
ENV TZ=Asia/Tokyo \
    VENV_DIR=/opt/venv \
    QT_X11_NO_MITSHM=1 \
    DISPLAY=:1 \
    USER=root \
    HOME=/root

##### タイムゾーン設定 ################################################
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone

##### システム更新 & 必要パッケージ ##################################
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        tigervnc-standalone-server xfce4 xfce4-goodies firefox \
        dbus-x11 x11-xserver-utils xfonts-base \
        python3-pip python3-venv pipx \
        git curl vim wget gedit terminator \
        python3-catkin-pkg python3-dotenv \
        libxcb-xinerama0 libxkbcommon-x11-0 libxcb-cursor0 \
        novnc websockify python3-websockify \
        libgl1 libglx-mesa0 libgl1-mesa-dri && \
    ln -s /usr/share/novnc/utils/novnc_proxy /usr/local/bin/novnc_proxy && \
    rm -rf /var/lib/apt/lists/*

##### Python 仮想環境 + requirements.txt #############################
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv $VENV_DIR && \
    . $VENV_DIR/bin/activate && \
    pip install --upgrade --break-system-packages pip setuptools wheel && \
    pip install --break-system-packages -r /tmp/requirements.txt

##### 追加の ROS 2 ツール ###########################################
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        ros-jazzy-rqt* ros-jazzy-rviz2 \
        ros-jazzy-demo-nodes-cpp ros-jazzy-demo-nodes-py && \
    rm -rf /var/lib/apt/lists/*

##### VNC パスワードと xstartup ######################################
# RUN mkdir -p /root/.vnc && \
#     echo "${VNC_PASSWORD}" | vncpasswd -f > /root/.vnc/passwd && \
#     chmod 600 /root/.vnc/passwd && \
#     echo "#!/bin/bash\nxrdb \$HOME/.Xresources\nxsetroot -solid grey\n/usr/bin/startxfce4 &" \
#         > /root/.vnc/xstartup && chmod +x /root/.vnc/xstartup

##### スタートアップスクリプト #######################################
COPY vnc_startup.sh /usr/local/bin/vnc_startup.sh
RUN chmod +x /usr/local/bin/vnc_startup.sh

##### ROS2 ワークスペース ###########################################
WORKDIR /workspace
RUN mkdir -p src && \
    echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc && \
    echo "cd /workspace" >> /root/.bashrc

##### ポート公開 & エントリポイント #################################
EXPOSE 5901 6080
CMD ["/usr/local/bin/vnc_startup.sh"]