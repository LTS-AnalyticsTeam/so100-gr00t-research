#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from vla_interfaces.msg import Action, State
from sensor_msgs.msg import Image
from std_msgs.msg import String
import time


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera')

        # ------ Publishers ------
        self.camera_vlm_pub = self.create_publisher(Image, '/camera_vlm', 10)
        self.camera_vla_pub = self.create_publisher(Image, '/camera_vla', 10)

        # ------ Subscribers ------
        # No Subscriber

        # ------ Timers ------
        self.timer = self.create_timer(0.5, self._timer_callback)
        
    def _timer_callback(self):
        """Periodic camera data publishing"""

        
def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('CameraNode shutting down')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()