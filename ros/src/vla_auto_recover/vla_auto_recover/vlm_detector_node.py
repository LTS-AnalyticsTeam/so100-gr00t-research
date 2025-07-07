#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from vla_auto_recover.processing.config.system_settings import CB_InputIF
from vla_interfaces.msg import ImagePair, DetectionOutput
from vla_auto_recover.processing.vlm_detector import VLMDetector
from vla_auto_recover.processing.config.system_settings import State
from vla_auto_recover.processing.config.prompt_settings import RUNNING_ACTION_ID
from std_msgs.msg import String
import threading
import queue
import traceback

class VLMDetectorNode(Node):

    def __init__(self):
        super().__init__("vlm_detector")

        # ------ Parameters ------
        self.declare_parameters(
            namespace="",
            parameters=[
                ("fps", 1/5),
                ("worker_num", 4),
            ],
        )

        # ------ Publishers ------
        self.detection_result_pub = self.create_publisher(
            DetectionOutput, "/detection_output", 10
        )

        # ------ Subscribers ------
        self.image_sub = self.create_subscription(
            ImagePair, "/image/vlm", self._cb_sub_put_queue, 10
        )
        self.state_change_sub = self.create_subscription(
            String, "/state_change", self._cb_sub_change_state, 10
        )

        # ------ Timers ------
        self._timer_detector_worker = self.create_timer(
            1 / self.get_parameter("fps").value, self._cb_timer_start_detector_worker
        )
        self._timer_publish_result = self.create_timer(
            1.0, self._cb_timer_publish_result
        )

        # ------ Property ------
        self.vlm_detector = VLMDetector()
        self.state = State.RUNNING
        self.action_id = RUNNING_ACTION_ID

        # Thread-safe queues
        self.q_image_pair = queue.Queue(maxsize=1)
        self.q_detection_output = queue.Queue()

        # Worker pool management
        self.worker_pool = []
        self.max_workers = self.get_parameter("worker_num").value
        self.worker_lock = threading.Lock()

    def _cb_sub_put_queue(self, msg: ImagePair):
        """Callback for the image subscription to put images into the queue."""
        try:
            self.q_image_pair.put_nowait(msg)
        except queue.Full:
            self.q_image_pair.get()
            self.q_image_pair.put_nowait(msg)
        self.get_logger().info(
            f"Image pair added to queue. Queue size: {self.q_image_pair.qsize()}"
        )

    def _cb_sub_change_state(self, msg: String):
        old_state = self.state
        self.state = State(msg.data)
        self.get_logger().info(f"State changed: {old_state} -> {self.state}")
        if self.state == State.END:
            self.get_logger().info("Shutting down VLMDetectorNode")
            self.destroy_node()
            rclpy.shutdown()
            

    def _cb_timer_start_detector_worker(self):
        """Start detection worker if queue has items and workers are available."""
        if self.q_image_pair.empty():
            return

        with self.worker_lock:

            try:
                # Remove finished workers from pool
                self.worker_pool = [w for w in self.worker_pool if w.is_alive()]

                # Start new worker if under limit
                if len(self.worker_pool) < self.max_workers:
                    worker = threading.Thread(
                        target=self._detection_worker,
                        daemon=True,
                        name=f"VLMWorker-{len(self.worker_pool)}",
                    )
                    worker.start()
                    self.worker_pool.append(worker)
                    self.get_logger().info(f"Started new worker: {worker.name}")
            except Exception as e:
                self.get_logger().error(f"Failed to start worker: {e}")

    def _detection_worker(self):
        """Worker thread to process images from the queue."""
        processed_count = 0

        try:
            # Get image from queue with timeout
            image_pair = self.q_image_pair.get(timeout=0.1)
        except queue.Empty:
            self.get_logger().info("No image in queue to process.")
            return
        except Exception as e:
            self.get_logger().error(f"Queue get failed: {e}")
            return

        try:
            input_data = CB_InputIF.from_msg(
                msg=image_pair,
                action_id=self.action_id,
            )
            output_data = self.vlm_detector.call_CB(self.state, input_data)
            self.q_detection_output.put(output_data)
            self.get_logger().info(
                f"Get Detection Output: {output_data.detection_result.value}, "
                f"Action ID: {output_data.action_id}, "
                f"Reason: {output_data.reason}"
            )

        except Exception as e:
            self.get_logger().error(
                f"Exception in detection worker: {traceback.format_exc()}"
            )
            self.get_logger().error(f"Detection processing failed: {e}")
            return

    def _cb_timer_publish_result(self):
        """Callback to publish detection results at a fixed rate."""
        if not self.q_detection_output.empty():
            output = self.q_detection_output.get()
            self.detection_result_pub.publish(
                DetectionOutput(
                    detection_result=output.detection_result.value,
                    action_id=output.action_id,
                    reason=output.reason,
                )
            )
            self.get_logger().info("Published DetectionOutput message")
            self.get_logger().info(
                f"Detection Result: {output.detection_result.value}, "
                f"Action ID: {output.action_id}, "
                f"Reason: {output.reason}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = VLMDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("VLMDetectorNode shutting down")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
