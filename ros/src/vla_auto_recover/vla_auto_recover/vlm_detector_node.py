#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from vla_interfaces.msg import DetectionResult, State
from sensor_msgs.msg import Image
from .processing.vlm_detector import VLMDetector


class VLMDetectorNode(Node):

    def __init__(self):
        super().__init__("vlm_detector")

        # ------ Publishers ------
        self.detection_result_pub = self.create_publisher(
            DetectionResult, "/detection_result", 10
        )

        # ------ Subscribers ------
        self.image_sub = self.create_subscription(
            Image, "/image/vlm", self._cb_put_queue, 10
        )
        self.state_change_sub = self.create_subscription(
            State, "/state_change", self._cb_change_state, 10
        )

        # ------ Timers ------
        # No timers needed for this node

        self.vlm_detector = VLMDetector()

    def _cb_put_queue(self, msg: Image): ...

    def _cb_change_state(self, msg: State): ...

    def _detector_worker(self): ...


def main(args=None):
    rclpy.init(args=args)
    node = VLMDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("VLMDetectorNode shutting down")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
