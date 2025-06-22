

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
## Record
``` bash
python -m lerobot.record \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.cameras="{ center_cam: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, right_cam: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
    --robot.id=white \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=black \
    --dataset.repo_id=lt-s/rsj2025_train_move_six_blocks_tray_to_matching_dishes \
    --dataset.fps=60 \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=4 \
    --dataset.num_episodes=20 \
    --dataset.single_task="move six blocks from tray to matching dishes" \
    --dataset.tags='["so100", "LeRobot", "RSJ2025", "train"]' \
    --display_data=true \
    --resume=true 
```
## Replay
``` bash
python -m lerobot.replay \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=white \
    --dataset.repo_id=lt-s/rsj2025_eval_move_red_block_tray_to_red_dish \
    --dataset.episode=2
```

## GR00T操作実行
isaac-gr00t-so100リポジトリでサーバーを立ち上げる必要がある。
``` bash
# lerobot old version
python scripts/exe_policy_lerobot1.py

# lerobot new version
python scripts/exe_policy_lerobot2.py \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=white \
    --robot.cameras="{ center_cam: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, right_cam: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
    --policy_host=localhost \
    --lang_instruction="move blocks from tray to matching dishes."
```



