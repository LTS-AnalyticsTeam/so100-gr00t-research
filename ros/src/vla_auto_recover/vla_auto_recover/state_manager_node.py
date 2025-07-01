#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from .processing.vlm_monitor import VLMMonitor
from vla_interfaces.msg import DetectionResult
from sensor_msgs.msg import Image
from std_msgs.msg import Int32


class StateManagerNode(Node):

    def __init__(self):
        super().__init__("state_manager")

        # ------ Publishers ------
        self.state_change_pub = self.create_publisher(Int32, "/state_change", 10)
        self.action_id_pub = self.create_publisher(Int32, "/action_id", 10)

        # ------ Subscribers ------
        self.detection_result_sub = self.create_subscription(
            DetectionResult, "/detection_result", self._cb_transition, 10
        )

        # ------ Timers ------
        self.timer = self.create_timer(3.0, self._timer_state_logger)

        self.vlm_monitor = VLMMonitor()

    def _cb_transition(self, msg: DetectionResult): ...

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
