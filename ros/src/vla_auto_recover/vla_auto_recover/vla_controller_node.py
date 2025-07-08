#!/usr/bin/env python3

import rclpy
import queue
import time
from rclpy.node import Node
from std_msgs.msg import Int32
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from vla_interfaces.msg import ImagePair
from vla_auto_recover.processing.vla_controller import GR00TExecuter
from vla_auto_recover.processing.utils.image_convert import imgmsg_to_ndarray
from vla_auto_recover.processing.config.prompt_settings import ACTION_END_ID
import traceback

class VLAControllerNode(Node):

    DRY_RUN = False

    def __init__(self):
        super().__init__("vla_controller")
        self.gr00t_executer = GR00TExecuter()
        self.q_image_pair = queue.Queue(maxsize=3)
        
        # ------ Publishers ------
        # No Publisher

        # ------ Subscribers ------
        self.image_sub = self.create_subscription(
            ImagePair,
            "/image/vla",
            self._cb_timer_save_image,
            qos_profile=QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,  # 欠けてもいいから最新を速く
                history=HistoryPolicy.KEEP_LAST,  # 直近だけ保持して古いのは捨てる
                depth=3,  # 最新の3つを保持
            ),
        )
        self.action_id_sub = self.create_subscription(
            Int32, "/action_id", self._cb_change_action, 10
        )

        # ------ Timers ------
        self.timer_exec_action = self.create_timer(1.0, self._timer_exec_action)

    def _cb_timer_save_image(self, msg: ImagePair):
        """Save incoming image pair to the queue"""
        # 画像ペアをキューに保存
        try:
            self.q_image_pair.put_nowait(msg)
        except queue.Full:
            self.q_image_pair.get()
            self.q_image_pair.put_nowait(msg)
        self.get_logger().info(
            f"Image pair added to queue. Queue size: {self.q_image_pair.qsize()}"
        )


    def _cb_change_action(self, msg: Int32):
        """Handle incoming recovery action requests"""
        if msg.data == ACTION_END_ID:
            self.get_logger().info("Received END_ID, shutting down VLAControllerNode")
            self.destroy_node()
            rclpy.shutdown()
            return
        else:        
            # アクションのIDを変更する
            self.action_id = msg.data
            # タイマーを停止
            self.timer_exec_action.cancel() 
            # Start Positionに戻す
            self.gr00t_executer.go_back_start_position()
            # タイマーをリセットして再開
            self.timer_exec_action.reset()  
            self.get_logger().info("Action ID changed to: {}".format(self.action_id))

    def _timer_exec_action(self):
        if self.DRY_RUN:
            self.get_logger().info("Executing action callback without real action.")
            return

        try:
            image_pair = self.q_image_pair.get_nowait()
        except queue.Empty:
            self.get_logger().info("No image pair available in the queue")
            return

        try:
            # ロボットの現在状態を取得
            robot_observation = self.gr00t_executer.robot.get_observation()
            
            # 画像データとロボット状態を結合
            observation_dict = {
                "center_cam": imgmsg_to_ndarray(image_pair.center_cam),
                "right_cam": imgmsg_to_ndarray(image_pair.right_cam),
            }
            
            # ロボットの関節位置を追加
            robot_state_keys = list(self.gr00t_executer.robot._motors_ft.keys())
            for key in robot_state_keys:
                observation_dict[key] = robot_observation[key]
            
            self.gr00t_executer.act(observation_dict)
            self.get_logger().info("Successfully executed action")
            
        except Exception as e:
            self.get_logger().error(f"Error during action execution: {e}")
            self.get_logger().error(traceback.format_exc())

        return None

    def destroy_node(self):
        # タイマーを停止
        try:
            if hasattr(self, 'timer_exec_action'):
                self.timer_exec_action.cancel()
        except Exception as e:
            self.get_logger().error(f"Error canceling timer: {e}")
        
        # ロボットを初期位置に戻す
        try:
            self.gr00t_executer.go_back_start_position()
        except Exception as e:
            self.get_logger().error(f"Error returning to start position: {e}")
        
        # ロボット接続を切断
        try:
            if hasattr(self.gr00t_executer, 'robot') and self.gr00t_executer.robot:
                self.gr00t_executer.robot.disconnect()
        except Exception as e:
            self.get_logger().error(f"Error disconnecting robot: {e}")
        
        # 親クラスのdestroy_nodeを呼び出し
        try:
            super().destroy_node()
        except Exception as e:
            self.get_logger().error(f"Error in parent destroy_node: {e}")


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
