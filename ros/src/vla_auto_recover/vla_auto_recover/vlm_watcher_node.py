#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from vla_interfaces.msg import Action, State
from sensor_msgs.msg import Image
from .processing.vlm import CLIENT

class VLMWatcher(Node):
    def __init__(self):
        super().__init__('vlm_watcher')
        
        # ------ Publishers ------
        self.state_pub = self.create_publisher(State, '/state', 10)

        # ------ Subscribers ------
        self.image_sub = self.create_subscription(Image, '/image', self._cb_watch_situation, 10)

        # ------ Timers ------
        self.timer = self.create_timer(3.0, self._timer_callback)
        
    def _cb_watch_situation(self, msg: Image):
        """Handle camera data for VLM analysis"""
        
    def _timer_callback(self):
        """Periodic state publishing"""


def main(args=None):
    rclpy.init(args=args)
    node = VLMWatcher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('VLMWatcher shutting down')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()