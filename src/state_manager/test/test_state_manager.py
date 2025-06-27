#!/usr/bin/env python3

import unittest
import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
import os
import time
import sys
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Mock ROS2 before importing the node
sys.modules['rclpy'] = Mock()
sys.modules['rclpy.node'] = Mock()
sys.modules['std_msgs'] = Mock()
sys.modules['std_msgs.msg'] = Mock()
sys.modules['vla_interfaces'] = Mock()
sys.modules['vla_interfaces.msg'] = Mock()

# Mock specific message types
mock_bool = Mock()
mock_string = Mock()
mock_action = Mock()
mock_recovery_status = Mock()
mock_verification_request = Mock()
mock_recovery_verification = Mock()

sys.modules['std_msgs.msg'].Bool = mock_bool
sys.modules['std_msgs.msg'].String = mock_string
sys.modules['vla_interfaces.msg'].Action = mock_action
sys.modules['vla_interfaces.msg'].RecoveryStatus = mock_recovery_status
sys.modules['vla_interfaces.msg'].VerificationRequest = mock_verification_request
sys.modules['vla_interfaces.msg'].RecoveryVerification = mock_recovery_verification

from state_manager.state_manager_node import StateManager, SystemState


class TestStateManager(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('state_manager.state_manager_node.rclpy')
    @patch('state_manager.state_manager_node.Node')
    def test_state_manager_initialization(self, mock_node, mock_rclpy):
        """Test StateManager initialization"""
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
        
        with patch.object(StateManager, 'setup_logging'), \
             patch.object(StateManager, '_log_detailed'), \
             patch('os.makedirs'):
            
            state_manager = StateManager()
            
            # Verify initialization
            self.assertEqual(state_manager.current_state, SystemState.NORMAL)
            self.assertEqual(state_manager.previous_state, SystemState.NORMAL)
            mock_node_instance.declare_parameter.assert_called()
            mock_node_instance.create_publisher.assert_called()
            mock_node_instance.create_subscription.assert_called()
            mock_logger.info.assert_called()
    
    def test_system_state_enum(self):
        """Test SystemState enumeration"""
        self.assertEqual(SystemState.NORMAL.value, "Normal")
        self.assertEqual(SystemState.ANOMALY_DETECTED.value, "AnomalyDetected")
        self.assertEqual(SystemState.RECOVERING.value, "Recovering")
        self.assertEqual(SystemState.VERIFICATION_PENDING.value, "VerificationPending")
    
    @patch('state_manager.state_manager_node.rclpy')
    @patch('state_manager.state_manager_node.Node')
    def test_state_transition_to_recovery(self, mock_node, mock_rclpy):
        """Test state transition from Normal to Recovering"""
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
        
        # Mock publishers
        mock_pause_publisher = Mock()
        mock_verification_publisher = Mock()
        mock_node_instance.create_publisher.side_effect = [mock_pause_publisher, mock_verification_publisher]
        
        with patch.object(StateManager, 'setup_logging'), \
             patch.object(StateManager, '_log_detailed'), \
             patch('os.makedirs'), \
             patch('time.time', return_value=1000.0):
            
            state_manager = StateManager()
            state_manager.pub_pause = mock_pause_publisher
            
            # Test state transition
            state_manager._start_recovery("test_action", "test_target")
            
            # Verify state change
            self.assertEqual(state_manager.current_state, SystemState.RECOVERING)
            self.assertEqual(state_manager.recovery_info['action_name'], "test_action")
            self.assertEqual(state_manager.recovery_info['target_id'], "test_target")
            
            # Verify VLA pause message was published
            mock_pause_publisher.publish.assert_called()
    
    @patch('state_manager.state_manager_node.rclpy')
    @patch('state_manager.state_manager_node.Node')
    def test_anomaly_callback(self, mock_node, mock_rclpy):
        """Test anomaly detection callback"""
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
        
        with patch.object(StateManager, 'setup_logging'), \
             patch.object(StateManager, '_log_detailed'), \
             patch.object(StateManager, '_start_recovery') as mock_start_recovery, \
             patch('os.makedirs'):
            
            state_manager = StateManager()
            
            # Create mock anomaly message
            mock_msg = Mock()
            mock_msg.name = "test_action"
            mock_msg.target_id = "test_target"
            
            # Test callback
            state_manager._cb_anomaly(mock_msg)
            
            # Verify recovery was started
            mock_start_recovery.assert_called_once_with("test_action", "test_target")
    
    @patch('state_manager.state_manager_node.rclpy')
    @patch('state_manager.state_manager_node.Node')
    def test_recovery_status_callback(self, mock_node, mock_rclpy):
        """Test recovery status callback"""
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
        
        with patch.object(StateManager, 'setup_logging'), \
             patch.object(StateManager, '_log_detailed'), \
             patch.object(StateManager, '_start_verification') as mock_start_verification, \
             patch('os.makedirs'):
            
            state_manager = StateManager()
            state_manager.current_state = SystemState.RECOVERING
            state_manager.recovery_info['action_name'] = "test_action"
            
            # Create mock recovery status message
            mock_msg = Mock()
            mock_msg.completed = True
            mock_msg.failed = False
            mock_msg.action_name = "test_action"
            mock_msg.progress = 1.0
            mock_msg.status = "completed"
            
            # Test callback
            state_manager._cb_recovery_status(mock_msg)
            
            # Verify verification was started
            mock_start_verification.assert_called_once()
    
    @patch('state_manager.state_manager_node.rclpy')
    @patch('state_manager.state_manager_node.Node')
    def test_verification_result_callback_success(self, mock_node, mock_rclpy):
        """Test verification result callback for successful recovery"""
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
        
        with patch.object(StateManager, 'setup_logging'), \
             patch.object(StateManager, '_log_detailed'), \
             patch.object(StateManager, '_end_recovery') as mock_end_recovery, \
             patch('os.makedirs'):
            
            state_manager = StateManager()
            state_manager.current_state = SystemState.VERIFICATION_PENDING
            
            # Create mock verification result message
            mock_msg = Mock()
            mock_msg.is_resolved = True
            mock_msg.confidence = 0.95
            mock_msg.description = "Problem resolved"
            
            # Test callback
            state_manager._cb_verification_result(mock_msg)
            
            # Verify recovery was ended successfully
            mock_end_recovery.assert_called_once_with(success=True, reason="VLM confirmed resolution: Problem resolved")
    
    @patch('state_manager.state_manager_node.rclpy')
    @patch('state_manager.state_manager_node.Node')
    def test_verification_timeout(self, mock_node, mock_rclpy):
        """Test verification timeout handling"""
        mock_node_instance = Mock()
        mock_node.return_value = mock_node_instance
        
        # Setup mocks
        mock_node_instance.declare_parameter = Mock()
        mock_param = Mock()
        mock_param.value = 1.0  # Short timeout for testing
        mock_node_instance.get_parameter = Mock(return_value=mock_param)
        
        mock_node_instance.create_publisher = Mock()
        mock_node_instance.create_subscription = Mock()
        mock_timer = Mock()
        mock_node_instance.create_timer = Mock(return_value=mock_timer)
        mock_node_instance.get_logger = Mock()
        mock_logger = Mock()
        mock_node_instance.get_logger.return_value = mock_logger
        
        with patch.object(StateManager, 'setup_logging'), \
             patch.object(StateManager, '_log_detailed'), \
             patch.object(StateManager, '_end_recovery') as mock_end_recovery, \
             patch('os.makedirs'), \
             patch('time.time') as mock_time:
            
            # Set up time progression to simulate timeout
            mock_time.side_effect = [1000.0, 1000.0, 1002.0]  # 2 second gap
            
            state_manager = StateManager()
            state_manager.current_state = SystemState.VERIFICATION_PENDING
            state_manager.verification_active = True
            state_manager.recovery_info['verification_start_time'] = 1000.0
            state_manager.verification_timeout = 1.0  # 1 second timeout
            
            # Test timeout check
            state_manager._verification_timer_callback()
            
            # Verify timeout was handled
            mock_end_recovery.assert_called_once_with(success=False, reason="VLM verification timeout (1.0s)")


class TestStateManagerIntegration(unittest.TestCase):
    """Integration tests for StateManager"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up integration test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('state_manager.state_manager_node.rclpy')
    @patch('state_manager.state_manager_node.Node')
    def test_full_recovery_workflow(self, mock_node, mock_rclpy):
        """Test complete recovery workflow from anomaly to resolution"""
        mock_node_instance = Mock()
        mock_node.return_value = mock_node_instance
        
        # Setup mocks
        mock_node_instance.declare_parameter = Mock()
        mock_param = Mock()
        mock_param.value = self.temp_dir
        mock_node_instance.get_parameter = Mock(return_value=mock_param)
        
        mock_pause_publisher = Mock()
        mock_verification_publisher = Mock()
        mock_node_instance.create_publisher = Mock(side_effect=[mock_pause_publisher, mock_verification_publisher])
        mock_node_instance.create_subscription = Mock()
        mock_node_instance.create_timer = Mock()
        mock_node_instance.get_logger = Mock()
        mock_logger = Mock()
        mock_node_instance.get_logger.return_value = mock_logger
        
        with patch.object(StateManager, 'setup_logging'), \
             patch.object(StateManager, '_log_detailed'), \
             patch('os.makedirs'), \
             patch('time.time', return_value=1000.0):
            
            state_manager = StateManager()
            state_manager.pub_pause = mock_pause_publisher
            state_manager.pub_verification_request = mock_verification_publisher
            
            # Step 1: Anomaly detected
            anomaly_msg = Mock()
            anomaly_msg.name = "test_action"
            anomaly_msg.target_id = "test_target"
            
            state_manager._cb_anomaly(anomaly_msg)
            self.assertEqual(state_manager.current_state, SystemState.RECOVERING)
            
            # Step 2: Recovery completed
            recovery_msg = Mock()
            recovery_msg.completed = True
            recovery_msg.failed = False
            recovery_msg.action_name = "test_action"
            recovery_msg.progress = 1.0
            recovery_msg.status = "completed"
            
            state_manager._cb_recovery_status(recovery_msg)
            self.assertEqual(state_manager.current_state, SystemState.VERIFICATION_PENDING)
            
            # Step 3: Verification result - success
            verification_msg = Mock()
            verification_msg.is_resolved = True
            verification_msg.confidence = 0.95
            verification_msg.description = "Problem resolved"
            
            state_manager._cb_verification_result(verification_msg)
            self.assertEqual(state_manager.current_state, SystemState.NORMAL)
            
            # Verify VLA was paused and resumed
            self.assertEqual(mock_pause_publisher.publish.call_count, 2)


if __name__ == '__main__':
    unittest.main()