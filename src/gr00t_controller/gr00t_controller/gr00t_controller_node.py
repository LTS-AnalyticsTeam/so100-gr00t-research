#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subscribe /recovery_action and execute on SO-ARM100 via
GR00T inference server, with pause control.
"""
import time
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from vla_interfaces.msg import Action

from so100_robot import SO100Robot
from service import ExternalRobotInferenceClient


class GR00TController(Node):
    def __init__(self):
        super().__init__('gr00t_controller')
        # Pause flag
        self.paused = False
        self.create_subscription(Bool, '/vla_pause', self._cb_pause, 10)

        # Recovery action
        self.sub_act = self.create_subscription(
            Action, '/recovery_action', self._cb_action, 10)

        # GR00T policy client
        self.policy = ExternalRobotInferenceClient()
        self.get_logger().info('Connected to GR00T server')

        # Real robot
        self.robot = SO100Robot(calibrate=False, enable_camera=True, cam_idx=1)
        self.robot_ctx = self.robot.activate()
        self.robot_ctx.__enter__()

        self.get_logger().info('GR00TController ready')

    def _cb_pause(self, msg: Bool):
        self.paused = msg.data
        self.get_logger().info(f'Pause={self.paused}')

    def _cb_action(self, msg: Action):
        # Recovering: ignore normal VLA loop, execute recovery
        if msg.name and not self.paused:
            # Ideally should only be called when paused==True
            self.get_logger().info(f'Executing recovery action: {msg.name}')
        if msg.name and self.paused:
            img = self.robot.get_current_img()
            state = self.robot.get_current_state()
            action_dict = self.policy.get_action(img, state)
            # 16-step chunk
            MOD_KEYS = ['single_arm', 'gripper']
            for i in range(action_dict['action.single_arm'].shape[0]):
                concat = np.concatenate(
                    [action_dict[f'action.{k}'][i] for k in MOD_KEYS], axis=0)
                self.robot.set_target_state(torch.from_numpy(concat))
                time.sleep(0.02)
            self.get_logger().info('Recovery done')

    def destroy_node(self):
        super().destroy_node()
        self.robot_ctx.__exit__(None, None, None)


def main(args=None):
    rclpy.init(args=args)
    node = GR00TController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
