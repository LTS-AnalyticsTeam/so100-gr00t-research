import rclpy
from rclpy.node import Node


class GR00TController(Node):
    def __init__(self):
        super().__init__('gr00t_controller')
        self.timer = self.create_timer(1, self.print_data)

    def print_data(self):
        self.get_logger().info('Printing data from GR00TController...')


def main(args=None):
    rclpy.init(args=args)
    node = GR00TController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('GR00TController shutting down')
        node._log_detailed('system_shutdown', {
            'reason': 'keyboard_interrupt',
            'final_stats': node.stats
        })
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
