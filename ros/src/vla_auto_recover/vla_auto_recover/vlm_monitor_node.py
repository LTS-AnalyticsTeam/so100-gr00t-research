#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from vla_interfaces.msg import Action
from sensor_msgs.msg import Image
from .processing.vlm_monitor import State


class VLMMonitorNode(Node):
    def __init__(self):
        super().__init__("vlm_monitor")

        # ------ Publishers ------
        self.action_pub = self.create_publisher(Action, "/action", 10)

        # ------ Subscribers ------
        self.image_sub = self.create_subscription(
            Image, "/image", self._cb_watch_situation, 10
        )

        # ------ Timers ------
        self.timer = self.create_timer(3.0, self._timer_callback)

        self.current_state = State.NORMAL

    def _cb_watch_situation(self, msg: Image):
        """Handle camera data for VLM analysis"""

    def _timer_callback(self):
        """Periodic state publishing"""


def main(args=None):
    rclpy.init(args=args)
    node = VLMMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("VLMMonitorNode shutting down")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
