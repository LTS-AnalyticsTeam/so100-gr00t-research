#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vla_interfaces.msg import ImagePair
import time
import os
import glob
import cv2
from vla_auto_recover.processing.utils.image_convert import numpy_to_imgmsg
import threading
from vla_auto_recover.processing.camera import CAMRERA_DIR

class CameraNode(Node):
    def __init__(self):
        super().__init__("camera")
        
        
        self.file_lock = threading.Lock()

        # ------ Publishers ------
        self.image_vlm_pub = self.create_publisher(ImagePair, "/image/vlm", 10)
        self.image_vla_pub = self.create_publisher(ImagePair, "/image/vla", 10)

        # ------ Subscribers ------
        # No Subscriber

        # ------ Timers ------
        self.timer_vlm_camera = self.create_timer(0.5, self._cb_publish_image_vlm)
        self.timer_vla_camera = self.create_timer(0.5, self._cb_publish_image_vla)

    def _get_latest_image_pair(self):
        """Get the latest image pair from camera directories with thread safety"""
        with self.file_lock:
            try:
                # Get latest images from both cameras
                center_cam_dir = os.path.join(CAMRERA_DIR, "center_cam")
                right_cam_dir = os.path.join(CAMRERA_DIR, "right_cam")
                
                if not os.path.exists(center_cam_dir) or not os.path.exists(right_cam_dir):
                    self.get_logger().warn("Camera directories not found")
                    return None, None
                
                # Get latest files from each directory
                center_files = glob.glob(os.path.join(center_cam_dir, "*.jpg"))
                right_files = glob.glob(os.path.join(right_cam_dir, "*.jpg"))
                
                if not center_files or not right_files:
                    self.get_logger().warn("No image files found in camera directories")
                    return None, None
                
                # Sort by filename (timestamp) and get the latest
                center_latest = sorted(center_files)[-1]
                right_latest = sorted(right_files)[-1]
                
                # Read images
                center_img = cv2.imread(center_latest)
                right_img = cv2.imread(right_latest)
                
                if center_img is None or right_img is None:
                    self.get_logger().warn("Failed to read image files")
                    return None, None
                    
                return center_img, right_img
                
            except Exception as e:
                self.get_logger().error(f"Error reading images: {e}")
                return None, None

    def _cb_publish_image_vlm(self):
        """Publish image pair for VLM processing"""
        try:
            center_img, right_img = self._get_latest_image_pair()
            
            if center_img is None or right_img is None:
                return
            
            # Convert OpenCV images to ROS Image messages
            center_msg = numpy_to_imgmsg(center_img, encoding="bgr8")
            right_msg = numpy_to_imgmsg(right_img, encoding="bgr8")
            
            # Create ImagePair message
            image_pair = ImagePair()
            image_pair.center_cam = center_msg
            image_pair.right_cam = right_msg
            image_pair.center_cam.header.stamp = self.get_clock().now().to_msg()
            image_pair.right_cam.header.stamp = self.get_clock().now().to_msg()
            
            # Publish
            self.image_vlm_pub.publish(image_pair)
            
        except Exception as e:
            self.get_logger().error(f"Error in VLM image callback: {e}")

    def _cb_publish_image_vla(self):
        """Publish image pair for VLA processing"""
        try:
            center_img, right_img = self._get_latest_image_pair()
            
            if center_img is None or right_img is None:
                return
            
            # Convert OpenCV images to ROS Image messages
            center_msg = numpy_to_imgmsg(center_img, encoding="bgr8")
            right_msg = numpy_to_imgmsg(right_img, encoding="bgr8")
            
            # Create ImagePair message
            image_pair = ImagePair()
            image_pair.center_cam = center_msg
            image_pair.right_cam = right_msg
            image_pair.center_cam.header.stamp = self.get_clock().now().to_msg()
            image_pair.right_cam.header.stamp = self.get_clock().now().to_msg()
            
            # Publish
            self.image_vla_pub.publish(image_pair)
            
        except Exception as e:
            self.get_logger().error(f"Error in VLA image callback: {e}")

    

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("CameraNode shutting down")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
