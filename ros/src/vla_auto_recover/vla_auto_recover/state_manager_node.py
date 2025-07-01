#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from .processing.vlm_detector import VLMDetector
from std_msgs.msg import Int32, String


class StateManagerNode(Node):

    def __init__(self):
        super().__init__("state_manager")

        # ------ Publishers ------
        self.state_change_pub = self.create_publisher(Int32, "/state_change", 10)
        self.action_id_pub = self.create_publisher(Int32, "/action_id", 10)

        # ------ Subscribers ------
        self.detection_result_sub = self.create_subscription(
            String, "/detection_result", self._cb_transition, 10
        )

        # ------ Timers ------
        self.timer = self.create_timer(3.0, self._timer_state_logger)

        self.vlm_detector = VLMDetector()

    def _cb_transition(self, msg: String): ...

    def _timer_state_logger(self): ...


def main(args=None):
    rclpy.init(args=args)
    node = StateManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("StateManagerNode shutting down")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
