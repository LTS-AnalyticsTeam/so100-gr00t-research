#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Normal  ---[ /recovery_action ]--->  Recovering
Recovering ---[ timeout ]--------->  Normal
Publishes /vla_pause (std_msgs/Bool)
"""
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from vla_interfaces.msg import Action


class StateManager(Node):
    def __init__(self):
        super().__init__('state_manager')
        self.state = 'Normal'
        self.recover_timeout = 3.0  

        self.pub_pause = self.create_publisher(Bool, '/vla_pause', 10)
        self.sub_action = self.create_subscription(
            Action, '/recovery_action', self._cb_action, 10)

        self.last_recover = 0.0
        self.create_timer(0.2, self._timer)

        self._set_pause(False)
        self.get_logger().info('StateManager ready')

    def _cb_action(self, msg: Action):
        if self.state == 'Normal':
            self.state = 'Recovering'
            self.last_recover = time.time()
            self._set_pause(True)
            self.get_logger().info('→ Recovering (pause VLA)')

    def _timer(self):
        if self.state == 'Recovering' and time.time() - self.last_recover > self.recover_timeout:
            self.state = 'Normal'
            self._set_pause(False)
            self.get_logger().info('→ Normal (resume VLA)')

    def _set_pause(self, flag: bool):
        msg = Bool()
        msg.data = flag
        self.pub_pause.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = StateManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
