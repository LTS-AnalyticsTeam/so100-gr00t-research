## docker環境のセットアップ
イメージの作成
```bash
docker build --tag so100-gr00t-research .
```
コンテナの作成
```bash
docker run -it -d \
  --ipc=host \
  --name so100-gr00t-research \
  -v "$(pwd)":/workspace/so100-gr00t-research \
  --gpus all \
  so100-gr00t-research:latest \
  /bin/bash
```

# pythonの仮想環境の構築
conda create -y -n so100 python=3.10
conda init bash
source ~/.bashrc
conda activate so100

## Isaac-GR00Tのセットアップ
```bash
cd Isaac-GR00T
pip install --upgrade setuptools
pip install -e .
pip install --no-build-isolation flash-attn==2.7.1.post4
```

## lerobotのセットアップ
cd lerobot
conda install ffmpeg -c conda-forge
pip install -e .
