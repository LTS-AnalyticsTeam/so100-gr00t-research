#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from vla_interfaces.msg import Action, State
from sensor_msgs.msg import Image

class GR00TController(Node):
    def __init__(self):
        super().__init__('gr00t_controller')

        # ------ Publishers ------
        # No Publisher

        # ------ Subscribers ------
        self.action_sub = self.create_subscription(Action, '/action', self._cb_action, 10)
        self.camera_vla_sub = self.create_subscription(Image, '/camera_vla', self._cb_camera_vla, 10)

        # ------ Timers ------
        self.timer = self.create_timer(1.0, self._timer_callback)
        
        # State variables
        self.is_executing_recovery = False
        self.vla_active = True

    def _cb_action(self, msg: Action):
        """Handle incoming recovery action requests"""
        
    def _cb_camera_vla(self, msg: Image):
        """Handle camera data for VLA"""
        
    def _timer_callback(self):
        """Periodic status publishing"""
        
        


def main(args=None):
    rclpy.init(args=args)
    node = GR00TController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('GR00TController shutting down')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()