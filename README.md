
# Isaac-GR00T環境構築
## jetson-containersの環境構築
- JetsonでEdge AI & Roboticsを行う場合の環境は、jetson-containersのgithubプロジェクトで用意されている。(https://github.com/dusty-nv/jetson-containers/tree/master)
- jetson-containersのセットアップから利用方法はこちらに記載(https://github.com/dusty-nv/jetson-containers/tree/master/docs)
- 詳細な作業手順とエラーハンドリングはnotinoに記載[notion: jetson-containersによるIsaac-GR00Tの環境構築](https://www.notion.so/jetson-containers-Isaac-GR00T-1f02c737cc9780f4b2a7d10f15e54b4b)

## lerobotの環境構築
git clone https://github.com/huggingface/lerobot.git
cd lerobot
conda create -y -n lerobot python=3.10
conda activate lerobot
conda install ffmpeg -c conda-forge
pip install -e .

# Issac-GR00Tの実行
## Issac-GR00Tの計算サーバー立ち上げ
```bash
docker run  \
  -it  \
  --rm \
  --runtime nvidia \
  --privileged \
  --network=host  \
  -v /mnt/suneast-ssd/ai-model:/mnt/suneast-ssd/ai-model \
  isaac-gr00t:r36.3.0-cu126-22.04 \
  python /opt/Isaac-GR00T/scripts/inference_service.py --server \
      --model_path /mnt/suneast-ssd/ai-model/gr00t/gr00t-checkpoint-7000-20250510 \
      --embodiment_tag new_embodiment \
      --data_config so100 \
      --denoising_steps 4 \
      --port 5555 \
      --host localhost
```

## lerobotによる操作実行

```bash
conda activate lerobot

sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1

# check robot connection
python ~/lerobot/lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --robot.cameras='{}' \
  --control.type=teleoperate

# check camera connection & setting
python scripts/eval_gr00t_so100.py --cam_idx 0

# manipulate so100 by GR000T
python scripts/eval_gr00t_so100.py \
 --use_policy \
 --host localhost \
 --port 5555 \
 --cam_idx 0

```