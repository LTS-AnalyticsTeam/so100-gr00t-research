``` bash
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1

```

Dockerが反応しない場合のトラブルシューティング
``` bash
sudo systemctl daemon-reload
sudo systemctl stop docker
sudo systemctl start docker  # or: sudo systemctl restart docker
```
## テレオペ
``` bash
python -m lerobot.teleoperate \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.cameras="{ center_cam: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, right_cam: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
    --robot.id=white \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=black \
    --display_data=true
```


## GR00T操作実行
isaac-gr00t-so100リポジトリでサーバーを立ち上げる必要がある。
``` bash
# lerobot old version
python scripts/exe_policy_lerobot1.py

# lerobot new version (from ros container)
python example/exe_policy_lerobot.py \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=white \
    --robot.cameras="{ center_cam: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, right_cam: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
    --robot.calibration_dir=/workspace/calibration/robots/so100_follower \
    --policy_host=localhost \
    --lang_instruction="move blocks from tray to matching dishes."

# lerobot new version (from local)
python example/exe_policy_lerobot.py \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=white \
    --robot.cameras="{ center_cam: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, right_cam: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
    --robot.calibration_dir=/home/lts-data/Project/so100-gr00t-research/calibration/robots/so100_follower \
    --policy_host=localhost \
    --lang_instruction="move blocks from tray to matching dishes." 
```
