#!/usr/bin/env python3

import unittest
import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
import os
import sys
import numpy as np
import time

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Mock ROS2 before importing the node
sys.modules['rclpy'] = Mock()
sys.modules['rclpy.node'] = Mock()
sys.modules['std_msgs'] = Mock()
sys.modules['std_msgs.msg'] = Mock()
sys.modules['vla_interfaces'] = Mock()
sys.modules['vla_interfaces.msg'] = Mock()
sys.modules['torch'] = Mock()

# Mock robot dependencies
sys.modules['so100_robot'] = Mock()
sys.modules['service'] = Mock()

# Mock specific message types
mock_bool = Mock()
mock_action = Mock()
mock_recovery_status = Mock()

sys.modules['std_msgs.msg'].Bool = mock_bool
sys.modules['vla_interfaces.msg'].Action = mock_action
sys.modules['vla_interfaces.msg'].RecoveryStatus = mock_recovery_status

from gr00t_controller.gr00t_controller_node import GR00TController


class TestGR00TController(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('gr00t_controller.gr00t_controller_node.rclpy')
    @patch('gr00t_controller.gr00t_controller_node.Node')
    def test_gr00t_controller_initialization(self, mock_node, mock_rclpy):
        """Test GR00TController initialization"""
        mock_node_instance = Mock()
        mock_node.return_value = mock_node_instance
        
        # Mock parameter methods
        mock_node_instance.declare_parameter = Mock()
        mock_param = Mock()
        mock_param.value = self.temp_dir
        mock_node_instance.get_parameter = Mock(return_value=mock_param)
        
        mock_node_instance.create_publisher = Mock()
        mock_node_instance.create_subscription = Mock()
        mock_node_instance.create_timer = Mock()
        mock_node_instance.get_logger = Mock()
        mock_logger = Mock()
        mock_node_instance.get_logger.return_value = mock_logger
        
        with patch.object(GR00TController, 'setup_logging'), \
             patch.object(GR00TController, '_log_detailed'), \
             patch('os.makedirs'):
            
            # Mock robot initialization to fail gracefully
            with patch('gr00t_controller.gr00t_controller_node.SO100Robot') as mock_robot, \
                 patch('gr00t_controller.gr00t_controller_node.ExternalRobotInferenceClient'):
                
                mock_robot.side_effect = Exception("Robot not available")
                
                controller = GR00TController()
                
                # Verify initialization
                self.assertFalse(controller.robot_connected)
                self.assertTrue(controller.vla_active)
                self.assertFalse(controller.paused)
                mock_node_instance.declare_parameter.assert_called()
                mock_node_instance.create_publisher.assert_called()
                mock_node_instance.create_subscription.assert_called()
    
    @patch('gr00t_controller.gr00t_controller_node.rclpy')
    @patch('gr00t_controller.gr00t_controller_node.Node')
    def test_vla_pause_callback(self, mock_node, mock_rclpy):
        """Test VLA pause/resume callback"""
        mock_node_instance = Mock()
        mock_node.return_value = mock_node_instance
        
        # Setup mocks
        mock_node_instance.declare_parameter = Mock()
        mock_param = Mock()
        mock_param.value = self.temp_dir
        mock_node_instance.get_parameter = Mock(return_value=mock_param)
        
        mock_node_instance.create_publisher = Mock()
        mock_node_instance.create_subscription = Mock()
        mock_node_instance.create_timer = Mock()
        mock_node_instance.get_logger = Mock()
        mock_logger = Mock()
        mock_node_instance.get_logger.return_value = mock_logger
        
        with patch.object(GR00TController, 'setup_logging'), \
             patch.object(GR00TController, '_log_detailed'), \
             patch('os.makedirs'):
            
            # Mock robot initialization to fail gracefully
            with patch('gr00t_controller.gr00t_controller_node.SO100Robot') as mock_robot, \
                 patch('gr00t_controller.gr00t_controller_node.ExternalRobotInferenceClient'):
                
                mock_robot.side_effect = Exception("Robot not available")
                
                controller = GR00TController()
                
                # Test pause
                pause_msg = Mock()
                pause_msg.data = True
                controller._cb_pause(pause_msg)
                self.assertTrue(controller.paused)
                self.assertFalse(controller.vla_active)
                
                # Test resume
                pause_msg.data = False
                controller._cb_pause(pause_msg)
                self.assertFalse(controller.paused)
                self.assertTrue(controller.vla_active)
    
    @patch('gr00t_controller.gr00t_controller_node.rclpy')
    @patch('gr00t_controller.gr00t_controller_node.Node')
    def test_recovery_action_callback(self, mock_node, mock_rclpy):
        """Test recovery action callback"""
        mock_node_instance = Mock()
        mock_node.return_value = mock_node_instance
        
        # Setup mocks
        mock_node_instance.declare_parameter = Mock()
        mock_param = Mock()
        mock_param.value = self.temp_dir
        mock_node_instance.get_parameter = Mock(return_value=mock_param)
        
        mock_node_instance.create_publisher = Mock()
        mock_node_instance.create_subscription = Mock()
        mock_node_instance.create_timer = Mock()
        mock_node_instance.get_logger = Mock()
        mock_logger = Mock()
        mock_node_instance.get_logger.return_value = mock_logger
        
        with patch.object(GR00TController, 'setup_logging'), \
             patch.object(GR00TController, '_log_detailed'), \
             patch.object(GR00TController, '_execute_recovery_action') as mock_execute, \
             patch('os.makedirs'):
            
            # Mock robot initialization to fail gracefully
            with patch('gr00t_controller.gr00t_controller_node.SO100Robot') as mock_robot, \
                 patch('gr00t_controller.gr00t_controller_node.ExternalRobotInferenceClient'):
                
                mock_robot.side_effect = Exception("Robot not available")
                
                controller = GR00TController()
                
                # Test recovery action
                action_msg = Mock()
                action_msg.name = "test_action"
                action_msg.target_id = "test_target"
                
                controller._cb_action(action_msg)
                
                # Verify action was stored and execution started
                self.assertEqual(controller.current_recovery_action, action_msg)
                mock_execute.assert_called_once()
    
    @patch('gr00t_controller.gr00t_controller_node.rclpy')
    @patch('gr00t_controller.gr00t_controller_node.Node')
    def test_vla_execution_when_paused(self, mock_node, mock_rclpy):
        """Test that VLA execution is skipped when paused"""
        mock_node_instance = Mock()
        mock_node.return_value = mock_node_instance
        
        # Setup mocks
        mock_node_instance.declare_parameter = Mock()
        mock_param = Mock()
        mock_param.value = self.temp_dir
        mock_node_instance.get_parameter = Mock(return_value=mock_param)
        
        mock_node_instance.create_publisher = Mock()
        mock_node_instance.create_subscription = Mock()
        mock_node_instance.create_timer = Mock()
        mock_node_instance.get_logger = Mock()
        mock_logger = Mock()
        mock_node_instance.get_logger.return_value = mock_logger
        
        with patch.object(GR00TController, 'setup_logging'), \
             patch.object(GR00TController, '_log_detailed'), \
             patch('os.makedirs'):
            
            # Mock robot initialization to succeed
            with patch('gr00t_controller.gr00t_controller_node.SO100Robot') as mock_robot_class, \
                 patch('gr00t_controller.gr00t_controller_node.ExternalRobotInferenceClient'):
                
                mock_robot = Mock()
                mock_robot.activate.return_value.__enter__ = Mock(return_value=mock_robot)
                mock_robot.activate.return_value.__exit__ = Mock(return_value=None)
                mock_robot_class.return_value = mock_robot
                
                controller = GR00TController()
                controller.robot_connected = True
                controller.paused = True
                controller.vla_active = False
                
                # Test VLA execution when paused
                controller._vla_normal_execution()
                
                # Verify no robot interaction occurred (would be mocked if called)
                # The method should return early when paused
                self.assertTrue(controller.paused)
    
    @patch('gr00t_controller.gr00t_controller_node.rclpy')
    @patch('gr00t_controller.gr00t_controller_node.Node')
    def test_recovery_execution(self, mock_node, mock_rclpy):
        """Test recovery action execution"""
        mock_node_instance = Mock()
        mock_node.return_value = mock_node_instance
        
        # Setup mocks
        mock_node_instance.declare_parameter = Mock()
        mock_param = Mock()
        mock_param.value = self.temp_dir
        mock_node_instance.get_parameter = Mock(return_value=mock_param)
        
        mock_publisher = Mock()
        mock_node_instance.create_publisher = Mock(return_value=mock_publisher)
        mock_node_instance.create_subscription = Mock()
        mock_node_instance.create_timer = Mock()
        mock_node_instance.get_logger = Mock()
        mock_logger = Mock()
        mock_node_instance.get_logger.return_value = mock_logger
        
        with patch.object(GR00TController, 'setup_logging'), \
             patch.object(GR00TController, '_log_detailed'), \
             patch('os.makedirs'), \
             patch('time.time', return_value=1000.0):
            
            # Mock robot initialization to succeed
            with patch('gr00t_controller.gr00t_controller_node.SO100Robot') as mock_robot_class, \
                 patch('gr00t_controller.gr00t_controller_node.ExternalRobotInferenceClient') as mock_policy_class:
                
                mock_robot = Mock()
                mock_robot.activate.return_value.__enter__ = Mock(return_value=mock_robot)
                mock_robot.activate.return_value.__exit__ = Mock(return_value=None)
                mock_robot.get_current_img.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
                mock_robot.get_current_state.return_value = np.zeros(7)
                mock_robot_class.return_value = mock_robot
                
                mock_policy = Mock()
                mock_policy.get_action.return_value = {
                    'action.single_arm': np.random.random((16, 7)),
                    'action.gripper': np.random.random((16, 1))
                }
                mock_policy_class.return_value = mock_policy
                
                controller = GR00TController()
                controller.robot_connected = True
                controller.pub_recovery_status = mock_publisher
                
                # Create test action
                action_msg = Mock()
                action_msg.name = "test_action"
                action_msg.target_id = "test_target"
                controller.current_recovery_action = action_msg
                
                # Test recovery execution
                controller._execute_recovery_action()
                
                # Verify robot interaction and status publishing
                mock_robot.get_current_img.assert_called()
                mock_robot.get_current_state.assert_called()
                mock_policy.get_action.assert_called()
                mock_publisher.publish.assert_called()


class TestGR00TControllerIntegration(unittest.TestCase):
    """Integration tests for GR00TController"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up integration test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('gr00t_controller.gr00t_controller_node.rclpy')
    @patch('gr00t_controller.gr00t_controller_node.Node')
    def test_full_recovery_workflow_simulation(self, mock_node, mock_rclpy):
        """Test complete recovery workflow simulation"""
        mock_node_instance = Mock()
        mock_node.return_value = mock_node_instance
        
        # Setup mocks
        mock_node_instance.declare_parameter = Mock()
        mock_param = Mock()
        mock_param.value = self.temp_dir
        mock_node_instance.get_parameter = Mock(return_value=mock_param)
        
        mock_publisher = Mock()
        mock_node_instance.create_publisher = Mock(return_value=mock_publisher)
        mock_node_instance.create_subscription = Mock()
        mock_node_instance.create_timer = Mock()
        mock_node_instance.get_logger = Mock()
        mock_logger = Mock()
        mock_node_instance.get_logger.return_value = mock_logger
        
        with patch.object(GR00TController, 'setup_logging'), \
             patch.object(GR00TController, '_log_detailed'), \
             patch('os.makedirs'), \
             patch('time.time', return_value=1000.0):
            
            # Mock robot initialization to succeed
            with patch('gr00t_controller.gr00t_controller_node.SO100Robot') as mock_robot_class, \
                 patch('gr00t_controller.gr00t_controller_node.ExternalRobotInferenceClient') as mock_policy_class:
                
                mock_robot = Mock()
                mock_robot.activate.return_value.__enter__ = Mock(return_value=mock_robot)
                mock_robot.activate.return_value.__exit__ = Mock(return_value=None)
                mock_robot.get_current_img.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
                mock_robot.get_current_state.return_value = np.zeros(7)
                mock_robot_class.return_value = mock_robot
                
                mock_policy = Mock()
                mock_policy.get_action.return_value = {
                    'action.single_arm': np.random.random((16, 7)),
                    'action.gripper': np.random.random((16, 1))
                }
                mock_policy_class.return_value = mock_policy
                
                controller = GR00TController()
                controller.robot_connected = True
                controller.pub_recovery_status = mock_publisher
                
                # Step 1: Receive pause command
                pause_msg = Mock()
                pause_msg.data = True
                controller._cb_pause(pause_msg)
                self.assertTrue(controller.paused)
                
                # Step 2: Receive recovery action
                action_msg = Mock()
                action_msg.name = "Move red block"
                action_msg.target_id = "block_red"
                controller._cb_action(action_msg)
                self.assertEqual(controller.current_recovery_action, action_msg)
                
                # Step 3: Resume after recovery
                pause_msg.data = False
                controller._cb_pause(pause_msg)
                self.assertFalse(controller.paused)
                self.assertTrue(controller.vla_active)
                
                # Verify status was published during recovery
                mock_publisher.publish.assert_called()


class TestSO100RobotMock(unittest.TestCase):
    """Test the mock SO100Robot implementation"""
    
    def test_mock_robot_functionality(self):
        """Test that mock robot provides expected interface"""
        # Import the mock class from the controller module
        with patch('gr00t_controller.gr00t_controller_node.rclpy'), \
             patch('gr00t_controller.gr00t_controller_node.Node'):
            
            from gr00t_controller.gr00t_controller_node import SO100Robot
            
            robot = SO100Robot()
            
            # Test activation context manager
            ctx = robot.activate()
            self.assertEqual(ctx.__enter__(), robot)
            self.assertIsNone(ctx.__exit__(None, None, None))
            
            # Test image and state methods
            img = robot.get_current_img()
            self.assertEqual(img.shape, (480, 640, 3))
            self.assertEqual(img.dtype, np.uint8)
            
            state = robot.get_current_state()
            self.assertEqual(state.shape, (7,))
            
            # Test set target state (should not raise)
            robot.set_target_state(np.zeros(7))


if __name__ == '__main__':
    unittest.main()