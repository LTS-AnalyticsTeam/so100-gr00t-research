from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable, Dict
from pathlib import Path
import torch
import zmq

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

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

from vla_auto_recover.processing.config.prompt_settings import (
    RUNNING_ACTION_ID,
    LANGUAGE_INSTRUCTION_LIST,
)


class TorchSerializer:
    @staticmethod
    def to_bytes(data: dict) -> bytes:
        buffer = BytesIO()
        torch.save(data, buffer)
        return buffer.getvalue()

    @staticmethod
    def from_bytes(data: bytes) -> dict:
        buffer = BytesIO(data)
        obj = torch.load(buffer, weights_only=False)
        return obj


@dataclass
class EndpointHandler:
    handler: Callable
    requires_input: bool = True


class BaseInferenceServer:
    """
    An inference server that spin up a ZeroMQ socket and listen for incoming requests.
    Can add custom endpoints by calling `register_endpoint`.
    """

    def __init__(self, host: str = "*", port: int = 5555):
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")
        self._endpoints: dict[str, EndpointHandler] = {}

        # Register the ping endpoint by default
        self.register_endpoint("ping", self._handle_ping, requires_input=False)
        self.register_endpoint("kill", self._kill_server, requires_input=False)

    def _kill_server(self):
        """
        Kill the server.
        """
        self.running = False

    def _handle_ping(self) -> dict:
        """
        Simple ping handler that returns a success message.
        """
        return {"status": "ok", "message": "Server is running"}

    def register_endpoint(
        self, name: str, handler: Callable, requires_input: bool = True
    ):
        """
        Register a new endpoint to the server.

        Args:
            name: The name of the endpoint.
            handler: The handler function that will be called when the endpoint is hit.
            requires_input: Whether the handler requires input data.
        """
        self._endpoints[name] = EndpointHandler(handler, requires_input)

    def run(self):
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f"Server is ready and listening on {addr}")
        while self.running:
            try:
                message = self.socket.recv()
                request = TorchSerializer.from_bytes(message)
                endpoint = request.get("endpoint", "get_action")

                if endpoint not in self._endpoints:
                    raise ValueError(f"Unknown endpoint: {endpoint}")

                handler = self._endpoints[endpoint]
                result = (
                    handler.handler(request.get("data", {}))
                    if handler.requires_input
                    else handler.handler()
                )
                self.socket.send(TorchSerializer.to_bytes(result))
            except Exception as e:
                print(f"Error in server: {e}")
                import traceback

                print(traceback.format_exc())
                self.socket.send(b"ERROR")


class BaseInferenceClient:
    def __init__(
        self, host: str = "localhost", port: int = 5555, timeout_ms: int = 15000
    ):
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self._init_socket()

    def _init_socket(self):
        """Initialize or reinitialize the socket with current settings"""
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()  # Recreate socket for next attempt
            return False

    def kill_server(self):
        """
        Kill the server.
        """
        self.call_endpoint("kill", requires_input=False)

    def call_endpoint(
        self, endpoint: str, data: dict | None = None, requires_input: bool = True
    ) -> dict:
        """
        Call an endpoint on the server.

        Args:
            endpoint: The name of the endpoint.
            data: The input data for the endpoint.
            requires_input: Whether the endpoint requires input data.
        """
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data

        self.socket.send(TorchSerializer.to_bytes(request))
        message = self.socket.recv()
        if message == b"ERROR":
            raise RuntimeError("Server error")
        return TorchSerializer.from_bytes(message)

    def __del__(self):
        """Cleanup resources on destruction"""
        self.socket.close()
        self.context.term()


class ExternalRobotInferenceClient(BaseInferenceClient):
    """
    Client for communicating with the RealRobotServer
    """

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the action from the server.
        The exact definition of the observations is defined
        by the policy, which contains the modalities configuration.
        """
        return self.call_endpoint("get_action", observations)


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
    ):
        self.policy = ExternalRobotInferenceClient(host=host, port=port)
        self.camera_keys = camera_keys
        self.robot_state_keys = robot_state_keys
        self.show_images = show_images
        assert (
            len(robot_state_keys) == 6
        ), f"robot_state_keys should be size 6, but got {len(robot_state_keys)} "
        self.modality_keys = ["single_arm", "gripper"]

    def get_action(self, observation_dict, lang: str):
        # first add the images
        obs_dict = {f"video.{key}": observation_dict[key] for key in self.camera_keys}

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
        for i in range(16):
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




class GR00TExecuter:

    IS_LOCAL = False
    REMOTE_SERVER = "74.226.200.235"

    def __init__(self):
        init_logging()

        # Step 1: Initialize the robot configuration
        # RobotConfigを正しく初期化（typeパラメータは使用しない）
        from lerobot.common.robots.so100_follower import SO100FollowerConfig

        self.robot_config = SO100FollowerConfig(
            port="/dev/ttyACM1",
            id="white",
            cameras={},
            calibration_dir=Path("/workspace/calibration/robots/so100_follower"),
        )

        logging.info(f"Robot configuration: {self.robot_config}")

        # Step 2: Initialize the robot
        self.robot = make_robot_from_config(self.robot_config)
        self.robot.connect()

        # get camera keys from RobotConfig
        camera_keys = ["center_cam", "right_cam"]
        print("camera_keys: ", camera_keys)
        robot_state_keys = list(self.robot._motors_ft.keys())
        print("robot_state_keys: ", robot_state_keys)
        observation = self.robot.get_observation()
        self.start_position = {key: observation[key] for key in robot_state_keys}
        print(f"記録した開始位置: {self.start_position}")

        # Configuration parameters
        self.policy_host = "localhost" if self.IS_LOCAL else  self.REMOTE_SERVER
        self.policy_port = 5555
        self.action_horizon = 16
        self.play_sounds = False
        self.timeout = 60
        self.action_id = RUNNING_ACTION_ID

        self.policy = Gr00tRobotInferenceClient(
            host=self.policy_host,
            port=self.policy_port,
            camera_keys=camera_keys,
            robot_state_keys=robot_state_keys,
        )

    def act(self, observation_dict: Dict[str, np.ndarray]) -> None:

        # get the realtime image
        print("observation_dict", observation_dict.keys())
        action_chunk = self.policy.get_action(observation_dict, self.lang_instruction)

        for i in range(self.action_horizon):
            action_dict = action_chunk[i]
            print("action_dict", action_dict.keys())
            self.robot.send_action(action_dict)
            time.sleep(0.02)  # Implicitly wait for the action to be executed

    def go_back_start_position(self):
        """Return to the start position"""
        print(f"Going back to start position: {self.start_position}")
        self.robot.send_action(self.start_position)
        time.sleep(2.0)

    @property
    def lang_instruction(self):
        return LANGUAGE_INSTRUCTION_LIST[self.action_id]
