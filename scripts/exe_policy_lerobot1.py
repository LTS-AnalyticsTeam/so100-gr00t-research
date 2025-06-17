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

# SO100 Real Robot
import time
from contextlib import contextmanager

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
from lerobot.common.robot_devices.robots.configs import So100RobotConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError

# NOTE:
# Sometimes we would like to abstract different env, or run this on a separate machine
# User can just move this single python class method gr00t/eval/service.py
# to their code or do the following line below
# sys.path.append(os.path.expanduser("~/Isaac-GR00T/gr00t/eval/"))
from service import ExternalRobotInferenceClient

# Import tqdm for progress bar
from tqdm import tqdm

#################################################################################


class SO100Robot:
    def __init__(self):
        self.config = So100RobotConfig()
        self.center_cam = 0
        self.right_cam = 2
        self.config.cameras = {
            "center_cam": OpenCVCameraConfig(self.center_cam, 30, 640, 480, "bgr"),
            "right_cam": OpenCVCameraConfig(self.right_cam, 30, 640, 480, "bgr")            
        }
        self.config.leader_arms = {}
        # Create the robot
        self.robot = make_robot_from_config(self.config)
        self.motor_bus = self.robot.follower_arms["main"]

    @contextmanager
    def activate(self):
        try:
            self.connect()
            self.move_to_initial_pose()
            yield
        finally:
            self.disconnect()

    def connect(self):
        if self.robot.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "ManipulatorRobot is already connected. Do not run `robot.connect()` twice."
            )

        # Connect the arms
        self.motor_bus.connect()

        # We assume that at connection time, arms are in a rest position, and torque can
        # be safely disabled to run calibration and/or set robot preset configurations.
        self.motor_bus.write("Torque_Enable", TorqueMode.DISABLED.value)

        # Calibrate the robot
        self.robot.activate_calibration()

        self.set_so100_robot_preset()

        # Enable torque on all motors of the follower arms
        self.motor_bus.write("Torque_Enable", TorqueMode.ENABLED.value)
        print("robot present position:", self.motor_bus.read("Present_Position"))
        self.robot.is_connected = True

        self.robot.cameras["center_cam"].connect()
        self.robot.cameras["right_cam"].connect()
        
        print("================> SO100 Robot is fully connected =================")

    def set_so100_robot_preset(self):
        # Mode=0 for Position Control
        self.motor_bus.write("Mode", 0)
        # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
        # self.motor_bus.write("P_Coefficient", 16)
        self.motor_bus.write("P_Coefficient", 10)
        # Set I_Coefficient and D_Coefficient to default value 0 and 32
        self.motor_bus.write("I_Coefficient", 0)
        self.motor_bus.write("D_Coefficient", 32)
        # Close the write lock so that Maximum_Acceleration gets written to EPROM address,
        # which is mandatory for Maximum_Acceleration to take effect after rebooting.
        self.motor_bus.write("Lock", 0)
        # Set Maximum_Acceleration to 254 to speedup acceleration and deceleration of
        # the motors. Note: this configuration is not in the official STS3215 Memory Table
        self.motor_bus.write("Maximum_Acceleration", 254)
        self.motor_bus.write("Acceleration", 254)

    def move_to_initial_pose(self):
        current_state = self.robot.capture_observation()["observation.state"]
        # print("current_state", current_state)
        # print all keys of the observation
        # print("observation keys:", self.robot.capture_observation().keys())
        current_state = torch.tensor([0, 90, 90, 90, -70, 30])
        self.robot.send_action(current_state)
        time.sleep(2)
        print("-------------------------------- moving to initial pose")

    def go_home(self):
        # [ 0, 156.7090, 135.6152,  83.7598, -89.1211,  16.5107]
        print("-------------------------------- moving to home pose")
        home_state = torch.tensor([0, 156.7090, 135.6152, 83.7598, -89.1211, 16.5107])
        self.set_target_state(home_state)
        time.sleep(2)

    def get_observation(self):
        return self.robot.capture_observation()

    def get_current_state(self):
        return self.get_observation()["observation.state"].data.numpy()

    def get_current_img(self):
        img_center_cam = self.get_observation()["observation.images.center_cam"].data.numpy()
        img_right_cam = self.get_observation()["observation.images.right_cam"].data.numpy()
        # convert bgr to rgb
        img_center_cam = cv2.cvtColor(img_center_cam, cv2.COLOR_BGR2RGB)
        img_right_cam = cv2.cvtColor(img_right_cam, cv2.COLOR_BGR2RGB)
        return img_center_cam, img_right_cam

    def set_target_state(self, target_state: torch.Tensor):
        self.robot.send_action(target_state)

    def enable(self):
        self.motor_bus.write("Torque_Enable", TorqueMode.ENABLED.value)

    def disable(self):
        self.motor_bus.write("Torque_Enable", TorqueMode.DISABLED.value)

    def disconnect(self):
        self.disable()
        self.robot.disconnect()
        self.robot.is_connected = False
        print("================> SO100 Robot disconnected")

    def __del__(self):
        self.disconnect()


#################################################################################


class Gr00tRobotInferenceClient:
    def __init__(
        self,
        host="localhost",
        port=5555,
        language_instruction="Pick up the fruits and place them on the plate.",
    ):
        self.language_instruction = language_instruction
        # 480, 640
        self.img_size = (480, 640)
        self.policy = ExternalRobotInferenceClient(host=host, port=port)

    def get_action(self, img_center_cam, img_right_cam, state):
        obs_dict = {
            "video.center_cam": img_center_cam[np.newaxis, :, :, :],
            "video.right_cam": img_right_cam[np.newaxis, :, :, :],
            "state.single_arm": state[:5][np.newaxis, :].astype(np.float64),
            "state.gripper": state[5:6][np.newaxis, :].astype(np.float64),
            "annotation.human.task_description": [self.language_instruction],
        }
        res = self.policy.get_action(obs_dict)
        # print("Inference query time taken", time.time() - start_time)
        return res

    def set_lang_instruction(self, lang_instruction):
        self.language_instruction = lang_instruction


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--action_horizon", type=int, default=12)
    parser.add_argument("--actions_to_execute", type=int, default=350)
    parser.add_argument(
        "--lang_instruction", type=str, default="move blocks from tray to matching dishes"
    )
    parser.add_argument("--record_imgs", action="store_true")
    args = parser.parse_args()

    # print lang_instruction
    print("lang_instruction: ", args.lang_instruction)

    ACTIONS_TO_EXECUTE = args.actions_to_execute
    ACTION_HORIZON = (
        args.action_horizon
    )  # we will execute only some actions from the action_chunk of 16
    MODALITY_KEYS = ["single_arm", "gripper"]

    client = Gr00tRobotInferenceClient(
        host=args.host,
        port=args.port,
        language_instruction=args.lang_instruction,
    )

    if args.record_imgs:
        # create a folder to save the images and delete all the images in the folder
        os.makedirs("eval_images", exist_ok=True)
        for file in os.listdir("eval_images"):
            os.remove(os.path.join("eval_images", file))

    robot = SO100Robot()
    image_count = 0
    with robot.activate():
        for i in tqdm(range(ACTIONS_TO_EXECUTE), desc="Executing actions"):
            img_center_cam, img_right_cam = robot.get_current_img()
            state = robot.get_current_state()
            action = client.get_action(img_center_cam, img_right_cam, state)
            start_time = time.time()
            for i in range(ACTION_HORIZON):
                concat_action = np.concatenate(
                    [np.atleast_1d(action[f"action.{key}"][i]) for key in MODALITY_KEYS],
                    axis=0,
                )
                assert concat_action.shape == (6,), concat_action.shape
                robot.set_target_state(torch.from_numpy(concat_action))
                time.sleep(0.02)

                if args.record_imgs:
                    # resize the image to 320x240
                    img_center_cam = cv2.resize(cv2.cvtColor(img_center_cam, cv2.COLOR_RGB2BGR), (320, 240))
                    cv2.imwrite(f"eval_images/center_cam/img_{image_count}.jpg", img_center_cam)
                    img_right_cam = cv2.resize(cv2.cvtColor(img_right_cam, cv2.COLOR_RGB2BGR), (320, 240))
                    cv2.imwrite(f"eval_images/right_cam/img_{image_count}.jpg", img_right_cam)
                    image_count += 1

                # 0.05*16 = 0.8 seconds
                print("executing action", i, "time taken", time.time() - start_time)
            print("Action chunk execution time taken", time.time() - start_time)
