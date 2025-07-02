#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from vla_auto_recover.processing.state_manager import StateManager
from vla_auto_recover.processing.config.system_settings import get_DR
from vla_interfaces.msg import DetectionOutput
from std_msgs.msg import Int32, String


class StateManagerNode(Node):

    def __init__(self):
        super().__init__("state_manager")

        # ------ Publishers ------
        self.state_change_pub = self.create_publisher(String, "/state_change", 10)
        self.action_id_pub = self.create_publisher(Int32, "/action_id", 10)

        # ------ Subscribers ------
        self.detection_result_sub = self.create_subscription(
            DetectionOutput, "/detection_output", self._cb_state_transition, 10
        )

        # ------ Timers ------
        # No timers needed for this node

        self.state_manager = StateManager()

    def _cb_state_transition(self, msg: DetectionOutput):
        state_changed = self.state_manager.transition(get_DR(msg.detection_result))
        if state_changed:
            self.get_logger().info(f"State transitioned to: {self.state_manager.state}")
            # Publish the new state
            self.state_change_pub.publish(String(data=self.state_manager.state))
            self.action_id_pub.publish(Int32(data=msg.action_id))
        else:
            self.get_logger().info(
                f"No state transition because detection result is `{msg.detection_result}`"
            )


def main(args=None):
    rclpy.init(args=args)
    node = StateManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("StateManagerNode shutting down")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
