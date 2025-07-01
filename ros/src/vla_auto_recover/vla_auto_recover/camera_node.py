#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vla_interfaces.msg import ImagePair
import time


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera")

        # ------ Publishers ------
        self.image_pub = self.create_publisher(ImagePair, "/image/vlm", 10)
        self.image_pub = self.create_publisher(ImagePair, "/image/vla", 10)

        # ------ Subscribers ------
        # No Subscriber

        # ------ Timers ------
        self.timer = self.create_timer(0.5, self._cb_publish_image)

    def _cb_publish_image(self): ...


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("CameraNode shutting down")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
