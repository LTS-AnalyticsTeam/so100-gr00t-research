#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple HSV-histogram-based anomaly detector.
Publishes vla_interfaces/Anomaly every 500 ms when rule is triggered.
"""
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vla_interfaces.msg import Anomaly
from cv_bridge import CvBridge


class AnomalyDetector(Node):
    def __init__(self) -> None:
        super().__init__('anomaly_detector')
        self.declare_parameter('hue_threshold', 10.0)
        self.hue_threshold = float(self.get_parameter('hue_threshold').value)

        self.sub = self.create_subscription(
            Image, '/rgb_image', self._cb_image, 10)

        self.pub = self.create_publisher(Anomaly, '/detected_anomaly', 10)
        self.bridge = CvBridge()

        self.last_pub_time = self.get_clock().now()
        self.pub_interval = 0.5  # seconds

        self.get_logger().info('AnomalyDetector ready')

    def _cb_image(self, msg: Image):
        now = self.get_clock().now()
        if (now - self.last_pub_time).nanoseconds < self.pub_interval * 1e9:
            return

        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        mean_h = float(np.mean(hsv[:, :, 0]))

        if mean_h < self.hue_threshold:
            anomaly = Anomaly()
            anomaly.type = 'pose_error'
            anomaly.image_path = ''
            self.pub.publish(anomaly)
            self.last_pub_time = now
            self.get_logger().info('Anomaly published: pose_error')


def main(args=None):
    rclpy.init(args=args)
    node = AnomalyDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
