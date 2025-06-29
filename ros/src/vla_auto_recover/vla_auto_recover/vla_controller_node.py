#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from vla_interfaces.msg import Action
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class VLAController(Node):
    def __init__(self):
        super().__init__("vla_controller")

        # ------ Publishers ------
        # No Publisher

        # ------ Subscribers ------
        self.action_sub = self.create_subscription(
            Action, "/action", self._cb_change_action, 10
        )
        self.image_sub = self.create_subscription(
            Image,
            "/image",
            self._cb_action_exec,
            qos_profile=QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,  # 欠けてもいいから最新を速く
                history=HistoryPolicy.KEEP_LAST,  # 直近だけ保持して古いのは捨てる
                depth=3,  # 最新の3つを保持
            ),
        )

        # ------ Timers ------
        self.timer = self.create_timer(1.0, self._timer_callback)

    def _cb_change_action(self, msg: Action):
        """Handle incoming recovery action requests"""

    def _cb_action_exec(self, msg: Image):
        """Handle camera data for VLA"""

    def _timer_callback(self):
        """Periodic status publishing"""


def main(args=None):
    rclpy.init(args=args)
    node = VLAController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("VLAController shutting down")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
