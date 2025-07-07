#!/usr/bin/env python3
"""
python ros/log_script/sub_rosout.py --node *
"""

import rclpy, sys
from rclpy.node import Node
from rcl_interfaces.msg import Log
from typing import Literal 

NODES = ["camera", "vlm_detector", "state_manager", "vla_controller"]

# Log level mapping from rcl_interfaces/msg/Log
LOG_LEVELS = {
    10: "DEBUG",
    20: "INFO", 
    30: "WARN",
    40: "ERROR",
    50: "FATAL"
}

def main(target_node_name):
    rclpy.init()
    ros_node = Node('rosout_tap')
    
    def cb(m: Log):
        if m.name == target_node_name:
            level_name = LOG_LEVELS.get(m.level, f"UNKNOWN({m.level})")
            print(f"[{level_name}] {m.msg}")
    
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