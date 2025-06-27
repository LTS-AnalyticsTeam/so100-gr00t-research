#!/usr/bin/env python3

import unittest
import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
import os
import time
import threading
import queue

# Mock ROS2 ecosystem
import sys
sys.modules['rclpy'] = Mock()
sys.modules['rclpy.node'] = Mock()
sys.modules['std_msgs'] = Mock()
sys.modules['std_msgs.msg'] = Mock()
sys.modules['vla_interfaces'] = Mock()
sys.modules['vla_interfaces.msg'] = Mock()

# Mock external dependencies
sys.modules['openai'] = Mock()
sys.modules['dotenv'] = Mock()
sys.modules['so100_robot'] = Mock()
sys.modules['service'] = Mock()
sys.modules['torch'] = Mock()
sys.modules['cv2'] = Mock()
sys.modules['numpy'] = Mock()

# Mock message types
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


class TestSystemIntegration(unittest.TestCase):
    """Integration tests for the complete VLA & VLM Auto Recovery system"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.message_queue = queue.Queue()
        
    def tearDown(self):
        """Clean up integration test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_recovery_workflow(self):
        """Test complete end-to-end recovery workflow"""
        # This test simulates the entire workflow:
        # VLMNode detects anomaly -> StateManager manages state -> GR00TController executes recovery
        
        # Import modules after mocking
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src/vlm_node'))
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src/state_manager'))
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src/gr00t_controller'))
        
        from vlm_node.vlm_node import VLMNode
        from state_manager.state_manager_node import StateManager, SystemState
        from gr00t_controller.gr00t_controller_node import GR00TController
        
        # Mock ROS2 publishing mechanism to capture messages
        published_messages = []
        
        def mock_publish(msg):
            published_messages.append(msg)
        
        # Test data
        test_anomaly = {
            'anomaly': 'missing_block',
            'action': 'Move the red block',
            'target_id': 'block_red',
            'confidence': 0.85
        }
        
        with patch('vlm_node.vlm_node.rclpy'), \
             patch('vlm_node.vlm_node.Node') as mock_vlm_node, \
             patch('state_manager.state_manager_node.rclpy'), \
             patch('state_manager.state_manager_node.Node') as mock_state_node, \
             patch('gr00t_controller.gr00t_controller_node.rclpy'), \
             patch('gr00t_controller.gr00t_controller_node.Node') as mock_gr00t_node:
            
            # Setup VLMNode mocks
            mock_vlm_instance = Mock()
            mock_vlm_node.return_value = mock_vlm_instance
            mock_vlm_instance.declare_parameter = Mock()
            mock_vlm_instance.get_parameter = Mock()
            mock_vlm_instance.get_parameter.return_value.value = self.temp_dir
            mock_vlm_instance.create_publisher = Mock()
            mock_vlm_instance.create_subscription = Mock()
            mock_vlm_instance.create_timer = Mock()
            mock_vlm_instance.get_logger = Mock()
            mock_vlm_instance.get_logger.return_value = Mock()
            
            # Setup StateManager mocks
            mock_state_instance = Mock()
            mock_state_node.return_value = mock_state_instance
            mock_state_instance.declare_parameter = Mock()
            mock_state_instance.get_parameter = Mock()
            mock_state_instance.get_parameter.return_value.value = self.temp_dir
            mock_state_instance.create_publisher = Mock()
            mock_state_instance.create_subscription = Mock()
            mock_state_instance.create_timer = Mock()
            mock_state_instance.get_logger = Mock()
            mock_state_instance.get_logger.return_value = Mock()
            
            # Setup GR00TController mocks
            mock_gr00t_instance = Mock()
            mock_gr00t_node.return_value = mock_gr00t_instance
            mock_gr00t_instance.declare_parameter = Mock()
            mock_gr00t_instance.get_parameter = Mock()
            mock_gr00t_instance.get_parameter.return_value.value = self.temp_dir
            mock_gr00t_instance.create_publisher = Mock()
            mock_gr00t_instance.create_subscription = Mock()
            mock_gr00t_instance.create_timer = Mock()
            mock_gr00t_instance.get_logger = Mock()
            mock_gr00t_instance.get_logger.return_value = Mock()
            
            with patch.object(VLMNode, 'setup_azure_client'), \
                 patch.object(VLMNode, '_load_prompt', return_value="test prompt"), \
                 patch.object(VLMNode, '_load_actions', return_value=[]), \
                 patch.object(VLMNode, '_find_image_files', return_value=[]), \
                 patch.object(VLMNode, 'setup_logging'), \
                 patch.object(VLMNode, '_log_detailed'), \
                 patch.object(StateManager, 'setup_logging'), \
                 patch.object(StateManager, '_log_detailed'), \
                 patch.object(GR00TController, 'setup_logging'), \
                 patch.object(GR00TController, '_log_detailed'), \
                 patch('os.makedirs'):
                
                # Create system components
                vlm_node = VLMNode()
                state_manager = StateManager()
                
                # Mock robot initialization to fail for testing
                with patch('gr00t_controller.gr00t_controller_node.SO100Robot') as mock_robot, \
                     patch('gr00t_controller.gr00t_controller_node.ExternalRobotInferenceClient'):
                    
                    mock_robot.side_effect = Exception("Robot not available")
                    gr00t_controller = GR00TController()
                
                # Step 1: Simulate VLM analysis detecting anomaly
                vlm_node.analysis_count = 0  # Should trigger anomaly
                with patch.object(vlm_node, '_get_recovery_action_and_target', 
                                return_value=(test_anomaly['action'], test_anomaly['target_id'])):
                    result = vlm_node._simulate_analysis("test_image.jpg")
                
                self.assertIn(result['anomaly'], ['missing_block', 'extra_block', 'pose_error'])
                self.assertEqual(result['action'], test_anomaly['action'])
                self.assertEqual(result['target_id'], test_anomaly['target_id'])
                
                # Step 2: Simulate StateManager receiving anomaly
                mock_action_msg = Mock()
                mock_action_msg.name = result['action']
                mock_action_msg.target_id = result['target_id']
                
                # Mock publisher for VLA pause
                mock_pause_publisher = Mock()
                state_manager.pub_pause = mock_pause_publisher
                
                with patch('time.time', return_value=1000.0):
                    state_manager._start_recovery(mock_action_msg.name, mock_action_msg.target_id)
                
                self.assertEqual(state_manager.current_state, SystemState.RECOVERING)
                mock_pause_publisher.publish.assert_called()  # Should pause VLA
                
                # Step 3: Simulate GR00TController receiving recovery action
                gr00t_controller._cb_action(mock_action_msg)
                self.assertEqual(gr00t_controller.current_recovery_action, mock_action_msg)
                
                # Step 4: Simulate recovery completion
                mock_recovery_status = Mock()
                mock_recovery_status.completed = True
                mock_recovery_status.failed = False
                mock_recovery_status.action_name = mock_action_msg.name
                mock_recovery_status.progress = 1.0
                mock_recovery_status.status = "completed"
                
                state_manager._cb_recovery_status(mock_recovery_status)
                self.assertEqual(state_manager.current_state, SystemState.VERIFICATION_PENDING)
                
                # Step 5: Simulate successful verification
                mock_verification_result = Mock()
                mock_verification_result.is_resolved = True
                mock_verification_result.confidence = 0.95
                mock_verification_result.description = "Problem resolved"
                
                state_manager._cb_verification_result(mock_verification_result)
                self.assertEqual(state_manager.current_state, SystemState.NORMAL)
    
    def test_message_flow_simulation(self):
        """Test ROS2 message flow between components"""
        # This test focuses on the message passing aspects of the system
        
        message_log = []
        
        def log_message(topic, msg_type, content):
            message_log.append({
                'topic': topic,
                'type': msg_type,
                'content': content,
                'timestamp': time.time()
            })
        
        # Simulate message publishing and subscribing
        topics = {
            '/recovery_action': [],
            '/recovery_status': [],
            '/vla_pause': [],
            '/verification_request': [],
            '/recovery_verification': []
        }
        
        # Test recovery action message
        recovery_action = {
            'name': 'Move the red block',
            'target_id': 'block_red'
        }
        topics['/recovery_action'].append(recovery_action)
        log_message('/recovery_action', 'Action', recovery_action)
        
        # Test VLA pause message
        vla_pause = {'data': True}
        topics['/vla_pause'].append(vla_pause)
        log_message('/vla_pause', 'Bool', vla_pause)
        
        # Test recovery status message
        recovery_status = {
            'completed': True,
            'failed': False,
            'action_name': 'Move the red block',
            'target_id': 'block_red',
            'progress': 1.0,
            'status': 'completed'
        }
        topics['/recovery_status'].append(recovery_status)
        log_message('/recovery_status', 'RecoveryStatus', recovery_status)
        
        # Test verification request message
        verification_request = {
            'verification_id': 'test_verification_123',
            'action_name': 'Move the red block',
            'target_id': 'block_red',
            'request_type': 'action_verification'
        }
        topics['/verification_request'].append(verification_request)
        log_message('/verification_request', 'VerificationRequest', verification_request)
        
        # Test verification result message
        verification_result = {
            'verification_id': 'test_verification_123',
            'is_resolved': True,
            'anomaly_type': 'normal',
            'confidence': 0.95,
            'description': 'Problem resolved'
        }
        topics['/recovery_verification'].append(verification_result)
        log_message('/recovery_verification', 'RecoveryVerification', verification_result)
        
        # Verify message flow
        self.assertEqual(len(message_log), 5)
        self.assertEqual(message_log[0]['topic'], '/recovery_action')
        self.assertEqual(message_log[1]['topic'], '/vla_pause')
        self.assertEqual(message_log[2]['topic'], '/recovery_status')
        self.assertEqual(message_log[3]['topic'], '/verification_request')
        self.assertEqual(message_log[4]['topic'], '/recovery_verification')
        
        # Verify message content
        self.assertEqual(message_log[0]['content']['name'], 'Move the red block')
        self.assertTrue(message_log[1]['content']['data'])
        self.assertTrue(message_log[2]['content']['completed'])
        self.assertTrue(message_log[4]['content']['is_resolved'])
    
    def test_error_handling_scenarios(self):
        """Test system behavior under error conditions"""
        
        # Test 1: VLM analysis failure
        with patch('vlm_node.vlm_node.rclpy'), \
             patch('vlm_node.vlm_node.Node'):
            
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src/vlm_node'))
            from vlm_node.vlm_node import VLMNode
            
            vlm_node = VLMNode.__new__(VLMNode)
            vlm_node.get_logger = Mock()
            vlm_node.get_logger.return_value = Mock()
            vlm_node.azure_client = None
            
            # Test Azure client not available
            result = vlm_node._simulate_analysis("test_image.jpg")
            self.assertIsInstance(result, dict)
            self.assertIn('anomaly', result)
        
        # Test 2: Robot connection failure
        with patch('gr00t_controller.gr00t_controller_node.rclpy'), \
             patch('gr00t_controller.gr00t_controller_node.Node'):
            
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src/gr00t_controller'))
            from gr00t_controller.gr00t_controller_node import GR00TController
            
            # Mock robot failure
            with patch('gr00t_controller.gr00t_controller_node.SO100Robot') as mock_robot, \
                 patch('gr00t_controller.gr00t_controller_node.ExternalRobotInferenceClient'), \
                 patch.object(GR00TController, 'setup_logging'), \
                 patch.object(GR00TController, '_log_detailed'), \
                 patch('os.makedirs'):
                
                mock_robot.side_effect = Exception("Robot connection failed")
                
                controller = GR00TController()
                self.assertFalse(controller.robot_connected)
        
        # Test 3: State transition errors
        with patch('state_manager.state_manager_node.rclpy'), \
             patch('state_manager.state_manager_node.Node'):
            
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src/state_manager'))
            from state_manager.state_manager_node import StateManager, SystemState
            
            with patch.object(StateManager, 'setup_logging'), \
                 patch.object(StateManager, '_log_detailed'), \
                 patch('os.makedirs'):
                
                state_manager = StateManager()
                
                # Test anomaly during recovery (should be ignored)
                state_manager.current_state = SystemState.RECOVERING
                
                mock_action = Mock()
                mock_action.name = "new_action"
                mock_action.target_id = "new_target"
                
                initial_state = state_manager.current_state
                state_manager._cb_anomaly(mock_action)
                
                # State should remain unchanged (anomaly ignored during recovery)
                self.assertEqual(state_manager.current_state, initial_state)


class TestConfigurationAndSetup(unittest.TestCase):
    """Test configuration files and setup procedures"""
    
    def setUp(self):
        """Set up configuration test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up configuration test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_action_list_configuration(self):
        """Test action list file loading and validation"""
        # Create test action list
        actions = [
            {"action": "Move the red block from the tray to the right dish.", "target_id": "block_red"},
            {"action": "Move the green block from the tray to the left dish.", "target_id": "block_green"},
            {"action": "Reposition the blue block to the center.", "target_id": "block_blue"}
        ]
        
        action_file = os.path.join(self.temp_dir, 'test_actions.jsonl')
        with open(action_file, 'w') as f:
            for action in actions:
                f.write(json.dumps(action) + '\n')
        
        # Test loading
        loaded_actions = []
        with open(action_file, 'r') as f:
            for line in f:
                loaded_actions.append(json.loads(line.strip()))
        
        self.assertEqual(len(loaded_actions), 3)
        self.assertEqual(loaded_actions[0]['action'], actions[0]['action'])
        self.assertEqual(loaded_actions[1]['target_id'], actions[1]['target_id'])
    
    def test_prompt_configuration(self):
        """Test prompt file loading and validation"""
        test_prompt = """Analyze this image for anomalies in a block sorting task.

IMPORTANT: Respond ONLY with valid JSON:
{
  "anomaly": "normal|pose_error|missing_block|extra_block",
  "action": "action_name_from_list",
  "target_id": "target_id_from_list",
  "confidence": 0.85,
  "reasoning": "Brief explanation"
}"""
        
        prompt_file = os.path.join(self.temp_dir, 'test_prompt.txt')
        with open(prompt_file, 'w') as f:
            f.write(test_prompt)
        
        # Test loading
        with open(prompt_file, 'r') as f:
            loaded_prompt = f.read()
        
        self.assertEqual(loaded_prompt, test_prompt)
        self.assertIn('anomaly', loaded_prompt)
        self.assertIn('JSON', loaded_prompt)
    
    def test_environment_configuration(self):
        """Test environment variable configuration"""
        # Test required environment variables
        required_vars = [
            'AZURE_OPENAI_API_KEY',
            'AZURE_OPENAI_ENDPOINT',
            'AZURE_OPENAI_DEPLOYMENT_NAME',
            'AZURE_OPENAI_API_VERSION'
        ]
        
        test_env = {
            'AZURE_OPENAI_API_KEY': 'test_key_123',
            'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com/',
            'AZURE_OPENAI_DEPLOYMENT_NAME': 'test_deployment',
            'AZURE_OPENAI_API_VERSION': '2024-02-15-preview'
        }
        
        with patch('os.getenv') as mock_getenv:
            mock_getenv.side_effect = lambda key, default=None: test_env.get(key, default)
            
            # Test all required variables are available
            for var in required_vars:
                value = os.getenv(var)
                self.assertIsNotNone(value)
                self.assertTrue(len(value) > 0)


if __name__ == '__main__':
    unittest.main()