#!/usr/bin/env python3

"""Test utilities and fixtures for the VLA & VLM Auto Recovery system tests"""

import tempfile
import json
import os
import cv2
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path


class TestFixtures:
    """Common test fixtures and utilities"""
    
    @staticmethod
    def create_test_image(width=640, height=480, channels=3):
        """Create a test image for testing"""
        return np.zeros((height, width, channels), dtype=np.uint8)
    
    @staticmethod
    def create_test_action_list():
        """Create a test action list for testing"""
        return [
            {"action": "Move the red block from the tray to the right dish.", "target_id": "block_red"},
            {"action": "Move the green block from the tray to the left dish.", "target_id": "block_green"},
            {"action": "Move the blue block from the tray to the center dish.", "target_id": "block_blue"},
            {"action": "Reposition the red block to the center of the right dish.", "target_id": "block_red"},
            {"action": "Stack the green block on top of the blue block.", "target_id": "block_green"}
        ]
    
    @staticmethod
    def create_test_prompt():
        """Create a test prompt for testing"""
        return """Analyze this image for anomalies in a block sorting task.

IMPORTANT: Respond ONLY with valid JSON:
{
  "anomaly": "normal|pose_error|missing_block|extra_block",
  "action": "action_name_from_list",
  "target_id": "target_id_from_list",
  "confidence": 0.85,
  "reasoning": "Brief explanation"
}"""
    
    @staticmethod
    def setup_test_files(temp_dir):
        """Set up test configuration files in a temporary directory"""
        # Create prompt file
        prompt_file = os.path.join(temp_dir, 'prompt.txt')
        with open(prompt_file, 'w') as f:
            f.write(TestFixtures.create_test_prompt())
        
        # Create action list file
        action_list_file = os.path.join(temp_dir, 'action_list.jsonl')
        actions = TestFixtures.create_test_action_list()
        with open(action_list_file, 'w') as f:
            for action in actions:
                f.write(json.dumps(action) + '\n')
        
        # Create image directory and test image
        image_dir = os.path.join(temp_dir, 'images')
        os.makedirs(image_dir)
        test_image = os.path.join(image_dir, 'test_image.jpg')
        img = TestFixtures.create_test_image()
        cv2.imwrite(test_image, img)
        
        return {
            'prompt_file': prompt_file,
            'action_list_file': action_list_file,
            'image_dir': image_dir,
            'test_image': test_image
        }


class ROSMockHelper:
    """Helper for mocking ROS2 components"""
    
    @staticmethod
    def setup_ros_mocks():
        """Set up common ROS2 mocks"""
        import sys
        
        # Mock ROS2 core
        sys.modules['rclpy'] = Mock()
        sys.modules['rclpy.node'] = Mock()
        
        # Mock standard messages
        sys.modules['std_msgs'] = Mock()
        sys.modules['std_msgs.msg'] = Mock()
        sys.modules['std_msgs.msg'].Bool = Mock()
        sys.modules['std_msgs.msg'].String = Mock()
        
        # Mock custom interfaces
        sys.modules['vla_interfaces'] = Mock()
        sys.modules['vla_interfaces.msg'] = Mock()
        sys.modules['vla_interfaces.msg'].Action = Mock()
        sys.modules['vla_interfaces.msg'].RecoveryStatus = Mock()
        sys.modules['vla_interfaces.msg'].VerificationRequest = Mock()
        sys.modules['vla_interfaces.msg'].RecoveryVerification = Mock()
        
        return sys.modules
    
    @staticmethod
    def create_mock_node():
        """Create a mock ROS2 node"""
        mock_node = Mock()
        mock_node.declare_parameter = Mock()
        mock_node.get_parameter = Mock()
        mock_node.create_publisher = Mock()
        mock_node.create_subscription = Mock()
        mock_node.create_timer = Mock()
        mock_node.get_logger = Mock()
        mock_node.get_logger.return_value = Mock()
        
        return mock_node
    
    @staticmethod
    def create_mock_message(msg_type, **kwargs):
        """Create a mock ROS2 message"""
        mock_msg = Mock()
        for key, value in kwargs.items():
            setattr(mock_msg, key, value)
        return mock_msg


class RobotMockHelper:
    """Helper for mocking robot components"""
    
    @staticmethod
    def setup_robot_mocks():
        """Set up robot-related mocks"""
        import sys
        
        # Mock robot dependencies
        sys.modules['so100_robot'] = Mock()
        sys.modules['service'] = Mock()
        sys.modules['torch'] = Mock()
        
        # Mock external libraries
        sys.modules['openai'] = Mock()
        sys.modules['dotenv'] = Mock()
        sys.modules['cv2'] = Mock()
        sys.modules['numpy'] = Mock()
        
        return sys.modules
    
    @staticmethod
    def create_mock_robot():
        """Create a mock SO100Robot"""
        mock_robot = Mock()
        mock_robot.activate.return_value.__enter__ = Mock(return_value=mock_robot)
        mock_robot.activate.return_value.__exit__ = Mock(return_value=None)
        mock_robot.get_current_img.return_value = TestFixtures.create_test_image()
        mock_robot.get_current_state.return_value = np.zeros(7)
        mock_robot.set_target_state = Mock()
        
        return mock_robot
    
    @staticmethod
    def create_mock_policy_client():
        """Create a mock policy inference client"""
        mock_client = Mock()
        mock_client.get_action.return_value = {
            'action.single_arm': np.random.random((16, 7)),
            'action.gripper': np.random.random((16, 1))
        }
        
        return mock_client


class AzureMockHelper:
    """Helper for mocking Azure OpenAI components"""
    
    @staticmethod
    def setup_azure_mocks():
        """Set up Azure OpenAI mocks"""
        import sys
        
        sys.modules['openai'] = Mock()
        sys.modules['dotenv'] = Mock()
        
        return sys.modules
    
    @staticmethod
    def create_mock_azure_client(response_data=None):
        """Create a mock Azure OpenAI client"""
        if response_data is None:
            response_data = {
                "anomaly": "missing_block",
                "action": "Move the red block",
                "target_id": "block_red",
                "confidence": 0.85,
                "reasoning": "Red block is missing from expected position"
            }
        
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = json.dumps(response_data)
        
        mock_client.chat.completions.create.return_value = mock_response
        
        return mock_client
    
    @staticmethod
    def setup_mock_environment():
        """Set up mock environment variables for Azure"""
        return {
            'AZURE_OPENAI_API_KEY': 'test_api_key_12345',
            'AZURE_OPENAI_ENDPOINT': 'https://test-resource.openai.azure.com/',
            'AZURE_OPENAI_DEPLOYMENT_NAME': 'test-gpt4-deployment',
            'AZURE_OPENAI_API_VERSION': '2024-02-15-preview',
            'USE_AZURE_OPENAI': 'true'
        }


def pytest_configure(config):
    """Pytest configuration hook"""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "requires_hardware: Tests requiring robot hardware")
    config.addinivalue_line("markers", "requires_api: Tests requiring external API access")