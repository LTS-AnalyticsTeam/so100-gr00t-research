#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from vla_auto_recover.processing.state_manager import StateManager
from vla_auto_recover.processing.config.system_settings import get_DR
from vla_interfaces.msg import DetectionOutput
from std_msgs.msg import Int32, String
from vla_auto_recover.processing.config.system_settings import State
from vla_interfaces.msg import SystemState
import traceback


class StateManagerNode(Node):

    def __init__(self):
        super().__init__("state_manager")

        # ------ Publishers ------
        self.state_change_pub = self.create_publisher(SystemState, "/state_change", 10)

        # ------ Subscribers ------
        self.detection_result_sub = self.create_subscription(
            DetectionOutput, "/detection_output", self._cb_state_transition, 10
        )

        # ------ Timers ------
        # No timers needed for this node

        self.state_manager = StateManager()
        # タイムスタンプベースの状態更新制御のための最新更新時間
        self.latest_update_timestamp = 0

    def _cb_state_transition(self, msg: DetectionOutput):
        try:
            # タイムスタンプが最新の更新時間より新しい場合のみ状態を更新
            if msg.timestamp <= self.latest_update_timestamp:
                self.get_logger().info(
                    f"Ignoring outdated detection result. "
                    f"Message timestamp: {msg.timestamp}, "
                    f"Latest update timestamp: {self.latest_update_timestamp}"
                )
                return

            # 最新の更新時間を更新
            self.latest_update_timestamp = msg.timestamp

            state_changed = self.state_manager.transition(get_DR(msg.detection_result))
            if state_changed:
                self.get_logger().info(
                    f"State transitioned to: {self.state_manager.state}"
                )
                # Publish the new state
                self.state_change_pub.publish(
                    SystemState(state=self.state_manager.state, action_id=msg.action_id)
                )
            else:
                self.get_logger().info(
                    f"No state transition because detection result is `{msg.detection_result}`"
                )

            if self.state_manager.state == State.END.value:
                self.get_logger().info("StateManagerNode shutting down")
                self.destroy_node()
                rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"Error in state transition callback: {e}")
            self.get_logger().debug(traceback.format_exc())


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
