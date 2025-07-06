#!/usr/bin/env python3
"""
python sub_rosout.py --node *
"""

import rclpy, sys
from rclpy.node import Node
from rcl_interfaces.msg import Log
from typing import Literal 

NODES = ["camera", "vlm_detector", "state_manager", "vla_controller"]

def main(target_node_name):
    rclpy.init()
    ros_node = Node('rosout_tap')
    
    def cb(m: Log):
        if m.name == target_node_name:
            print(m.msg)
    
    ros_node.create_subscription(Log, '/rosout', cb, 10)
    rclpy.spin(ros_node)
    ros_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="ROS Node Logger")
    parser.add_argument(
        "--node",
        choices=NODES,
        help="Name of the node to log messages from",
    )
    args = parser.parse_args()
    
    main(args.node)