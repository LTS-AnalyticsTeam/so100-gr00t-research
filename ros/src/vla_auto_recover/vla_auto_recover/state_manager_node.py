#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from vla_interfaces.msg import Action, State

class StateManager(Node):
    def __init__(self):
        super().__init__('state_manager')
        # ------ Publishers ------
        self.action_pub = self.create_publisher(Action, '/action', 10)

        # ------ Subscribers ------
        self.state_sub = self.create_subscription(State, '/state', self._cb_state, 10)

        # ------ Timers ------
        self.timer = self.create_timer(2.0, self._timer_callback)
        self.current_state = State.NORMAL
        
    def _cb_state(self, msg: State):
        """Handle state changes"""
        
        
    def _cb_recovery_status(self, msg: State):
        """Handle recovery status updates"""
        
        
    def _timer_callback(self):
        """Periodic action publishing"""
        


def main(args=None):
    rclpy.init(args=args)
    node = StateManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('StateManager shutting down')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()