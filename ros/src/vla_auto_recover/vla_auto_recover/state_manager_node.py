import rclpy
from rclpy.node import Node


class StateManager(Node):
    def __init__(self):
        super().__init__('state_manager')
        self.timer = self.create_timer(1, self.print_data)

    def print_data(self):
        self.get_logger().info('Printing data from StateManager...')

def main(args=None):
    rclpy.init(args=args)
    node = StateManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('StateManager shutting down')
        node._log_detailed('system_shutdown', {
            'reason': 'keyboard_interrupt',
            'final_stats': node.stats
        })
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
