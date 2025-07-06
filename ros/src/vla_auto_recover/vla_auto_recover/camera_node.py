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
        
        # ------ Parameters ------
        self.declare_parameters(
            namespace="",
            parameters=[
                ("center_cam_id", 0),
                ("right_cam_id", 2),
                ("fps", 30),
            ],
        )
        
        # Get camera parameters
        self.center_cam_id = self.get_parameter("center_cam_id").value
        self.right_cam_id = self.get_parameter("right_cam_id").value
        self.fps = self.get_parameter("fps").value
        
        self.frame_lock = threading.Lock()
        self.latest_center_frame = None
        self.latest_right_frame = None
        
        # Initialize cameras
        self.cameras = {}
        if not self._initialize_cameras():
            self.get_logger().error("Failed to initialize cameras")
            return

        # ------ Publishers ------
        self.image_vlm_pub = self.create_publisher(ImagePair, "/image/vlm", 10)
        self.image_vla_pub = self.create_publisher(ImagePair, "/image/vla", 10)

        # ------ Subscribers ------
        # No Subscriber

        # ------ Timers ------
        self.timer_vlm_camera = self.create_timer(0.5, self._cb_publish_image_vlm)
        self.timer_vla_camera = self.create_timer(0.5, self._cb_publish_image_vla)
        
        # Start camera capture thread
        self.capture_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        self.get_logger().info(f"Camera node initialized with center_cam_id={self.center_cam_id}, right_cam_id={self.right_cam_id}")

    def _initialize_cameras(self) -> bool:
        """Initialize cameras"""
        try:
            # Initialize center camera
            center_cap = cv2.VideoCapture(self.center_cam_id)
            if not center_cap.isOpened():
                self.get_logger().error(f"Cannot open center_cam (ID: {self.center_cam_id})")
                return False
            
            # Set center camera FPS and properties
            center_cap.set(cv2.CAP_PROP_FPS, self.fps)
            center_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            center_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            center_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to get latest frame
            
            actual_fps_center = center_cap.get(cv2.CAP_PROP_FPS)
            self.get_logger().info(f"Center cam FPS set to: {actual_fps_center}")
            
            self.cameras['center_cam'] = center_cap
            
            # Initialize right camera
            right_cap = cv2.VideoCapture(self.right_cam_id)
            if not right_cap.isOpened():
                self.get_logger().error(f"Cannot open right_cam (ID: {self.right_cam_id})")
                center_cap.release()
                return False
            
            # Set right camera FPS and properties
            right_cap.set(cv2.CAP_PROP_FPS, self.fps)
            right_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            right_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            right_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to get latest frame
            
            actual_fps_right = right_cap.get(cv2.CAP_PROP_FPS)
            self.get_logger().info(f"Right cam FPS set to: {actual_fps_right}")
            
            self.cameras['right_cam'] = right_cap
            
            self.get_logger().info("Cameras initialized successfully")
            return True
            
        except Exception as e:
            self.get_logger().error(f"Error initializing cameras: {e}")
            return False

    def _capture_loop(self):
        """Continuous camera capture loop running in separate thread"""
        while self.capture_running:
            try:
                center_frame = None
                right_frame = None
                
                # Capture from center camera
                if 'center_cam' in self.cameras:
                    ret_center, center_frame = self.cameras['center_cam'].read()
                    if not ret_center:
                        center_frame = None
                        self.get_logger().warn("Failed to read from center camera")
                        
                # Capture from right camera
                if 'right_cam' in self.cameras:
                    ret_right, right_frame = self.cameras['right_cam'].read()
                    if not ret_right:
                        right_frame = None
                        self.get_logger().warn("Failed to read from right camera")
                
                # Update latest frames with thread safety
                with self.frame_lock:
                    if center_frame is not None:
                        self.latest_center_frame = center_frame.copy()
                    if right_frame is not None:
                        self.latest_right_frame = right_frame.copy()
                
                # Small delay to prevent excessive CPU usage
                time.sleep(1.0 / self.fps)
                
            except Exception as e:
                self.get_logger().error(f"Error in camera capture loop: {e}")
                time.sleep(0.1)

    def _get_latest_image_pair(self):
        """Get the latest image pair from cameras with thread safety"""
        with self.frame_lock:
            try:
                if self.latest_center_frame is None or self.latest_right_frame is None:
                    self.get_logger().warn("No camera frames available")
                    return None, None
                
                # Return copies to avoid modification issues
                return self.latest_center_frame.copy(), self.latest_right_frame.copy()
                
            except Exception as e:
                self.get_logger().error(f"Error getting latest frames: {e}")
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
            self.get_logger().info("VLM image pair published successfully")
            
            
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
            self.get_logger().info("VLA image pair published successfully")
            
        except Exception as e:
            self.get_logger().error(f"Error in VLA image callback: {e}")

    def destroy_node(self):
        """Clean up resources when node is destroyed"""
        # Stop capture thread
        self.capture_running = False
        if hasattr(self, 'capture_thread') and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        # Release cameras
        for name, camera in self.cameras.items():
            if camera.isOpened():
                camera.release()
                self.get_logger().info(f"Released {name}")
        self.cameras.clear()
        
        # Call parent cleanup
        super().destroy_node()
    

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
