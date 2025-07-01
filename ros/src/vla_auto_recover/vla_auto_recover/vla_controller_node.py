#!/usr/bin/env python3

import rclpy
import queue
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class VLAControllerNode(Node):
    def __init__(self):
        super().__init__("vla_controller")

        # ------ Publishers ------
        # No Publisher

        # ------ Subscribers ------
        self.action_id_sub = self.create_subscription(
            Int32, "/action_id", self._cb_change_action, 10
        )
        self.image_sub = self.create_subscription(
            Image,
            "/image/vla",
            self._cb_save_image,
            qos_profile=QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,  # 欠けてもいいから最新を速く
                history=HistoryPolicy.KEEP_LAST,  # 直近だけ保持して古いのは捨てる
                depth=3,  # 最新の3つを保持
            ),
        )

        # ------ Timers ------
        self.timer = self.create_timer(1.0, self._timer_exec_action)

        self._img_queue = queue.Queue(maxsize=1)

    def _cb_change_action(self, msg: Int32):
        """Handle incoming recovery action requests"""
        # 1. _timer_exec_actionを止める
        # 2. Home Positionに戻す
        # 3. アクションの変数を変更する

    def _cb_save_image(self, msg: Image):
        """Handle camera data for VLA"""
        try:
            self._img_queue.put_nowait(msg)
        except queue.Full:
            pass  # 古い画像は捨てる

    def _timer_exec_action(self): ...


def main(args=None):
    rclpy.init(args=args)
    node = VLAControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("VLAControllerNode shutting down")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
