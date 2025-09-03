<div align="center">

# VLA Auto Recover
このリポジトリは、視覚言語モデル (VLM) による状態認識と、状態遷移管理(StateManager)、アクション実行(Controller) を連携させた自律 / 半自律タスク実行基盤を構築することを目的としています。

</div>


## 目次
1. 背景 / 目的
2. 全体アーキテクチャ概要
3. セットアップ
4. ビルド & 起動手順 (ROS2 ワークスペース)
5. ROS2 ワークスペース構造
6. ログ / デバッグ

---

## 2. 全体アーキテクチャ概要
| レイヤ | 説明 |
|--------|------|
| camera node | 入力 | カメラ / センサ / (将来: ロボット状態) |
| vlm_detector | 認識 | VLM によるシーン・状態推定 |
| state_manager | 状態管理 | StateManager が状態遷移図に基づき現在状態を更新 |
| vla_controller | 制御 | Controller が次アクションを決定・実行 |

アクタ間のノード & トピック構造:  
![Node構成](docs/nodes_graph.svg)

状態遷移モデル:  
<p align="center"><img src="docs/system_state.svg" alt="状態遷移図" width="520"></p>

---

## 3. セットアップ

### 3.1 Docker環境の立ち上げ
```bash
docker-compose build vla_auto_recover-cli
docker-compose up -d vla_auto_recover-cli
docker attach vla_auto_recover_cli
```


### 3.2 SO100 ARM
ハードウエアとしてSO-100 ARMを利用した。下記、サイトを参考にセットアップを行う。
- [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
- [SO-ARM101 Installation](https://huggingface.co/docs/lerobot/so101)


### 3.3 Lerobotのインストール
SO-100 ARMを動かすために、Lerobotを利用した。下記URLを参考にインストールを行う。
- [Lerobot Installation](https://huggingface.co/docs/lerobot/installation)
このプロジェクトは lerobot コミット `1ee2ca5c2627eab05940452472d876d0d4e73d1f` を利用。  
ROS からパッケージ解決される必要があるため `pip install -e .` は避ける。

### 3.4 GR00Tサーバの立ち上げ
このリポジトリに従ってGR00Tサーバーを立ち上げる。
- [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T/tree/main)

---

## 4. ビルド & 起動手順 (ROS2 ワークスペース)

環境変数ファイル作成 (`ros/config/.env`):
```bash
cat > ros/config/.env <<'EOF'
# Azure OpenAI設定
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
AZURE_OPENAI_API_VERSION=2024-02-15-preview
USE_AZURE_OPENAI=true
EOF
```

ノード一括起動:
```bash
cd ros
bash run_all_nodes.sh
```



---

## 5. ROS2 ワークスペース構造
```
ros/
	src/
		vla_auto_recover/        # 制御・リカバリロジック
		vla_interfaces/          # カスタムメッセージ / サービス
	run_all_nodes.sh             # ノード一括起動スクリプト
    write_graph_nodes.sh         # トピックグラフ生成スクリプト
	config/.env                  # 環境変数設定
	log_script/
		sub_rosout.py            # ログ購読ユーティリティ
```

---

## 6. ログ / デバッグ
個別ログ購読:
```bash
python ros/log_script/sub_rosout.py --node camera
python ros/log_script/sub_rosout.py --node vlm_detector
python ros/log_script/sub_rosout.py --node state_manager
python ros/log_script/sub_rosout.py --node vla_controller
```
