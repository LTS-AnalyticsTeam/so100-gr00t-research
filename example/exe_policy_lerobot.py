# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This is the new Gr00T policy eval script with so100, so101 robot arm. Based on:
https://github.com/huggingface/lerobot/pull/777

Example command:

```shell

python exe_policy_lerobot.py \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=white \
    --robot.cameras="{ center_cam: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, right_cam: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
    --policy_host=localhost \
    --lang_instruction="move blocks from tray to matching dishes."
```


First replay to ensure the robot is working:
```shell
python -m lerobot.replay \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=white \
    --dataset.repo_id=youliangtan/so100_strawberry_grape \
    --dataset.episode=2
```
"""

import json
import os
import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat
from pathlib import Path

import draccus
import matplotlib.pyplot as plt
import numpy as np
from lerobot.common.cameras.opencv.configuration_opencv import (  # noqa: F401
    OpenCVCameraConfig,
)
from lerobot.common.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.common.utils.utils import (
    init_logging,
    log_say,
)
import threading
import sys

# NOTE:
# Sometimes we would like to abstract different env, or run this on a separate machine
# User can just move this single python class method gr00t/eval/service.py
# to their code or do the following line below
# sys.path.append(os.path.expanduser("~/Isaac-GR00T/gr00t/eval/"))
from service import ExternalRobotInferenceClient

# from gr00t.eval.service import ExternalRobotInferenceClient

#################################################################################


class Gr00tRobotInferenceClient:
    """The exact keys used is defined in modality.json

    This currently only supports so100_follower, so101_follower
    modify this code to support other robots with other keys based on modality.json
    """

    def __init__(
        self,
        host="localhost",
        port=5555,
        camera_keys=[],
        robot_state_keys=[],
        show_images=False,
        action_chunk_size=16,  # アクションチャンクサイズのパラメータ追加
    ):
        self.policy = ExternalRobotInferenceClient(host=host, port=port)
        self.camera_keys = camera_keys
        self.robot_state_keys = robot_state_keys
        self.show_images = show_images
        self.action_chunk_size = action_chunk_size
        assert (
            len(robot_state_keys) == 6
        ), f"robot_state_keys should be size 6, but got {len(robot_state_keys)} "
        self.modality_keys = ["single_arm", "gripper"]

    def get_action(self, observation_dict, lang: str):
        # first add the images
        obs_dict = {f"video.{key}": observation_dict[key] for key in self.camera_keys}

        # show images
        if self.show_images:
            view_img(obs_dict)

        # Make all single float value of dict[str, float] state into a single array
        state = np.array([observation_dict[k] for k in self.robot_state_keys])
        obs_dict["state.single_arm"] = state[:5].astype(np.float64)
        obs_dict["state.gripper"] = state[5:6].astype(np.float64)
        obs_dict["annotation.human.task_description"] = lang

        # then add a dummy dimension of np.array([1, ...]) to all the keys (assume history is 1)
        for k in obs_dict:
            if isinstance(obs_dict[k], np.ndarray):
                obs_dict[k] = obs_dict[k][np.newaxis, ...]
            else:
                obs_dict[k] = [obs_dict[k]]

        # get the action chunk via the policy server
        # Example of obs_dict for single camera task:
        # obs_dict = {
        #     "video.front": np.zeros((1, 480, 640, 3), dtype=np.uint8),
        #     "video.wrist": np.zeros((1, 480, 640, 3), dtype=np.uint8),
        #     "state.single_arm": np.zeros((1, 5)),
        #     "state.gripper": np.zeros((1, 1)),
        #     "annotation.human.action.task_description": [self.language_instruction],
        # }
        action_chunk = self.policy.get_action(obs_dict)

        # convert the action chunk to a list of dict[str, float]
        lerobot_actions = []
        for i in range(
            self.action_chunk_size
        ):  # ハードコードされた値を self.action_chunk_size に変更
            action_dict = self._convert_to_lerobot_action(action_chunk, i)
            lerobot_actions.append(action_dict)
        return lerobot_actions

    def _convert_to_lerobot_action(
        self, action_chunk: dict[str, np.array], idx: int
    ) -> dict[str, float]:
        """
        This is a magic function that converts the action chunk to a dict[str, float]
        This is because the action chunk is a dict[str, np.array]
        and we want to convert it to a dict[str, float]
        so that we can send it to the robot
        """
        concat_action = np.concatenate(
            [
                np.atleast_1d(action_chunk[f"action.{key}"][idx])
                for key in self.modality_keys
            ],
            axis=0,
        )
        assert len(concat_action) == len(self.robot_state_keys), "this should be size 6"
        # convert the action to dict[str, float]
        action_dict = {
            key: concat_action[i] for i, key in enumerate(self.robot_state_keys)
        }
        return action_dict


#################################################################################


def view_img(img, overlay_img=None):
    """
    This is a matplotlib viewer since cv2.imshow can be flaky in lerobot env
    """
    if isinstance(img, dict):
        # stack the images horizontally
        img = np.concatenate([img[k] for k in img], axis=1)

    plt.imshow(img)
    plt.title("Camera View")
    plt.axis("off")
    plt.pause(0.001)  # Non-blocking show
    plt.clf()  # Clear the figure for the next frame


def print_yellow(text):
    print("\033[93m {}\033[00m".format(text))


@dataclass
class EvalConfig:
    robot: RobotConfig  # the robot to use
    policy_host: str = "localhost"  # host of the gr00t server
    policy_port: int = 5555  # port of the gr00t server
    action_horizon: int = 16  # number of actions to execute from the action chunk
    lang_instruction: str = "Grab pens and place into pen holder."
    play_sounds: bool = False  # whether to play sounds
    timeout: int = 60  # timeout in seconds for policy requests
    max_runtime: int = 0  # maximum total runtime in seconds (0 means no limit)
    show_images: bool = False  # whether to show images
    action_chunk_size: int = 16  # size of action chunks generated by policy
    max_chunks: int = 0  # maximum number of action chunks to process (0 means no limit)
    return_to_start: bool = True  # 動作終了後に開始位置に戻るかどうか


@draccus.wrap()
def eval(cfg: EvalConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # キー入力による終了フラグ
    stop_event = threading.Event()

    # キー入力を監視するスレッド
    def input_monitor():
        print(
            "\033[93m実行中に Ctrl+C または 'q' + Enter キーで安全に終了します\033[0m"
        )
        while not stop_event.is_set():
            try:
                user_input = input()
                if user_input.lower() == "q":
                    print(
                        "\033[91m中断コマンドを受け取りました。安全に終了します...\033[0m"
                    )
                    stop_event.set()
                    break
            except KeyboardInterrupt:
                print("\033[91mCtrl+C が押されました。安全に終了します...\033[0m")
                stop_event.set()
                break
            except Exception:
                pass  # 他の例外を無視

    # 入力モニタースレッドを開始
    input_thread = threading.Thread(target=input_monitor, daemon=True)
    input_thread.start()

    # Step 1: Initialize the robot
    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    # get camera keys from RobotConfig
    camera_keys = list(cfg.robot.cameras.keys())
    print("camera_keys: ", camera_keys)

    log_say("Initializing robot", cfg.play_sounds, blocking=True)

    language_instruction = cfg.lang_instruction

    # NOTE: for so100/so101, this should be:
    # ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']
    robot_state_keys = list(robot._motors_ft.keys())
    print("robot_state_keys: ", robot_state_keys)

    # 開始位置を記録
    start_position = None
    if cfg.return_to_start:
        # ロボットの現在位置を取得
        observation = robot.get_observation()
        start_position = {}
        for key in robot_state_keys:
            start_position[key] = observation[key]
        print(f"記録した開始位置: {start_position}")

    # Step 2: Initialize the policy
    policy = Gr00tRobotInferenceClient(
        host=cfg.policy_host,
        port=cfg.policy_port,
        camera_keys=camera_keys,
        robot_state_keys=robot_state_keys,
        show_images=cfg.show_images,
        action_chunk_size=cfg.action_chunk_size,
    )
    log_say(
        "Initializing policy client with language instruction: " + language_instruction,
        cfg.play_sounds,
        blocking=True,
    )

    # Step 3: Run the Eval Loop
    start_time = time.time()  # 実行開始時間を記録
    chunk_count = 0  # チャンクカウンターを追加

    try:
        while (
            not stop_event.is_set()
        ):  # keyboard.is_pressed を stop_event.is_set() に変更
            # 最大実行時間のチェック
            if cfg.max_runtime > 0 and time.time() - start_time > cfg.max_runtime:
                log_say(
                    f"Maximum runtime of {cfg.max_runtime} seconds reached",
                    cfg.play_sounds,
                )
                break

            # 最大チャンク数のチェック
            if cfg.max_chunks > 0 and chunk_count >= cfg.max_chunks:
                log_say(
                    f"Maximum number of chunks ({cfg.max_chunks}) reached",
                    cfg.play_sounds,
                )
                break

            # get the realtime image
            observation_dict = robot.get_observation()
            print("observation_dict", observation_dict.keys())

            # タイムアウトパラメータを使用してポリシーのアクション取得に制限を設ける場合はここで実装
            action_chunk = policy.get_action(observation_dict, language_instruction)

            # チャンクカウンターをインクリメント
            chunk_count += 1
            print(f"Processing chunk {chunk_count}")

            for i in range(cfg.action_horizon):
                action_dict = action_chunk[i]
                print("action_dict", action_dict.keys())
                robot.send_action(action_dict)
                time.sleep(0.02)  # Implicitly wait for the action to be executed
    finally:
        # 終了処理のため、スレッドを停止
        stop_event.set()

        # 開始位置に戻る処理
        if cfg.return_to_start and start_position:
            log_say("開始位置に戻ります", cfg.play_sounds)
            print("開始位置に移動中...")

            # 開始位置に移動
            robot.send_action(start_position)
            time.sleep(2.0)  # 十分な移動時間を確保

        # 後処理
        robot.disconnect()
        log_say("ロボット接続を終了しました", cfg.play_sounds)


if __name__ == "__main__":
    eval()
