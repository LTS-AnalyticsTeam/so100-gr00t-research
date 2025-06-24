#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from vla_interfaces.msg import Action, RecoveryStatus, VerificationRequest, RecoveryVerification
import time
import json
import os
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
import asyncio
from threading import Thread, Lock
import queue

# Azure OpenAI imports
try:
    from openai import AzureOpenAI
    from dotenv import load_dotenv
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    print("Warning: Azure OpenAI dependencies not available")

class VLMNode(Node):
    def __init__(self):
        super().__init__('vlm_node')
        
        # Configuration parameters
        self.declare_parameter('input_dir', '/workspace/src/data')
        self.declare_parameter('prompt_file', '/workspace/config/prompt.txt')
        self.declare_parameter('action_list_file', '/workspace/config/action_list.jsonl')
        self.declare_parameter('output_file', '/tmp/vlm_results.txt')
        self.declare_parameter('analysis_interval', 1.0)
        self.declare_parameter('log_directory', '/tmp')
        
        self.input_dir = self.get_parameter('input_dir').value
        self.prompt_file = self.get_parameter('prompt_file').value
        self.action_list_file = self.get_parameter('action_list_file').value
        self.output_file = self.get_parameter('output_file').value
        self.analysis_interval = self.get_parameter('analysis_interval').value
        self.log_dir = self.get_parameter('log_directory').value
        
        # Azure OpenAI setup
        self.setup_azure_client()
        
        # Load prompt and actions
        self.prompt = self._load_prompt()
        self.actions = self._load_actions()
        
        # Find image files
        self.image_files = self._find_image_files()
        
        # Analysis state
        self.current_image_index = 0
        self.analysis_count = 0
        self.last_anomaly_action = None
        self.recovery_active = False
        
        # Verification tracking
        self.verification_queue = queue.Queue()
        self.verification_lock = Lock()
        
        # Publishers
        self.pub_anomaly = self.create_publisher(Action, '/recovery_action', 10)
        self.pub_recovery_status = self.create_publisher(RecoveryStatus, '/recovery_status', 10)
        self.pub_verification_result = self.create_publisher(RecoveryVerification, '/recovery_verification', 10)
        
        # Subscribers  
        self.sub_verification_request = self.create_subscription(
            VerificationRequest, '/verification_request', self._cb_verification_request, 10)
        
        # Analysis timer
        self.analysis_timer = self.create_timer(self.analysis_interval, self._analysis_timer_callback)
        
        # Verification worker thread
        self.verification_thread = Thread(target=self._verification_worker, daemon=True)
        self.verification_thread.start()
        
        # Logging setup
        self.setup_logging()
        
        self.get_logger().info(f'VLMNode initialized: {len(self.image_files)} images, {len(self.actions)} actions')
        self._log_detailed('system_startup', {
            'image_count': len(self.image_files),
            'action_count': len(self.actions),
            'analysis_interval': self.analysis_interval,
            'azure_available': AZURE_AVAILABLE,
            'available_actions': [action.get('action', '') for action in self.actions]
        })
    
    def setup_azure_client(self):
        """Setup Azure OpenAI client"""
        if not AZURE_AVAILABLE:
            self.azure_client = None
            self.get_logger().warning('Azure OpenAI not available - using simulation mode')
            return
        
        try:
            load_dotenv('/workspace/.env')
            self.azure_client = AzureOpenAI(
                api_key=os.getenv('AZURE_OPENAI_API_KEY'),
                api_version=os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-01'),
                azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')
            )
            self.get_logger().info('Azure OpenAI client initialized')
        except Exception as e:
            self.azure_client = None
            self.get_logger().error(f'Failed to initialize Azure OpenAI: {e}')
    
    def setup_logging(self):
        """Setup detailed logging system"""
        self.log_file = os.path.join(
            self.log_dir, 
            f"vlm_node_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        self.get_logger().info(f'Detailed logging to: {self.log_file}')
    
    def _log_detailed(self, event_type: str, details: dict):
        """Write detailed log entry"""
        timestamp = time.time()
        log_entry = {
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp).isoformat(),
            'event_type': event_type,
            'recovery_active': self.recovery_active,
            'current_image_index': self.current_image_index,
            'analysis_count': self.analysis_count,
            'last_anomaly_action': self.last_anomaly_action,
            'details': details
        }
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, indent=None) + '\n')
        except Exception as e:
            self.get_logger().error(f'Failed to write log: {e}')
    
    def _load_prompt(self) -> str:
        """Load system prompt from file"""
        try:
            with open(self.prompt_file, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
            self.get_logger().info(f'Loaded prompt from {self.prompt_file} ({len(prompt)} chars)')
            return prompt
        except Exception as e:
            self.get_logger().error(f'Failed to load prompt: {e}')
            return "Analyze the image and detect any anomalies."
    
    def _load_actions(self) -> list:
        """Load action definitions from JSONL file"""
        actions = []
        try:
            with open(self.action_list_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        actions.append(json.loads(line.strip()))
            self.get_logger().info(f'Loaded {len(actions)} actions from action_list.jsonl:')
            for i, action in enumerate(actions):
                self.get_logger().info(f'  [{i}] {action.get("action", "NO_ACTION")}')
            return actions
        except Exception as e:
            self.get_logger().error(f'Failed to load actions: {e}')
            return []
    
    def _find_image_files(self) -> list:
        """Find all image files in input directory"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        input_path = Path(self.input_dir)
        if input_path.exists():
            for file_path in input_path.iterdir():
                if file_path.suffix.lower() in image_extensions:
                    image_files.append(str(file_path))
        
        # Sort for consistent ordering
        image_files.sort()
        
        for i, filepath in enumerate(image_files):
            filename = Path(filepath).name
            self.get_logger().info(f'  [{i}] {filename}')
        
        return image_files
    
    def _analysis_timer_callback(self):
        """Main analysis timer - continuously analyze images"""
        if not self.image_files:
            return
        
        # Get current image
        image_path = self.image_files[self.current_image_index]
        
        self._log_detailed('analysis_start', {
            'image_path': image_path,
            'image_index': self.current_image_index
        })
        
        self.get_logger().info(f'Starting analysis: {Path(image_path).name}')
        
        # Perform VLM analysis
        analysis_start_time = time.time()
        result = self._analyze_image(image_path)
        analysis_duration = time.time() - analysis_start_time
        
        self.analysis_count += 1
        
        self._log_detailed('analysis_completed', {
            'image_path': image_path,
            'duration': analysis_duration,
            'analysis_count': self.analysis_count,
            'result': result
        })
        
        self.get_logger().info(f'Analysis completed: {Path(image_path).name} ({analysis_duration:.2f}s)')
        
        # 🔧 FIXED: Process result - check for any anomaly except 'normal'
        if result and result.get('anomaly') and result.get('anomaly') != 'normal':
            self._handle_anomaly_detected(result, image_path)
        else:
            self.get_logger().info(f'No anomaly detected in {Path(image_path).name} (anomaly: {result.get("anomaly", "unknown") if result else "no_result"})')
        
        # Move to next image (cycle through all images)
        self.current_image_index = (self.current_image_index + 1) % len(self.image_files)
    
    def _analyze_image(self, image_path: str) -> dict:
        """Analyze image using Azure OpenAI Vision"""
        if self.azure_client is None:
            # Simulation mode - alternate between anomalies for testing
            return self._simulate_analysis(image_path)
        
        try:
            # Load and encode image
            image = cv2.imread(image_path)
            if image is None:
                self.get_logger().error(f'Failed to load image: {image_path}')
                return None
            
            # Convert to base64 for API
            _, buffer = cv2.imencode('.jpg', image)
            import base64
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Prepare messages
            messages = [
                {
                    "role": "system",
                    "content": self.prompt
                },
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this image and determine if there are any anomalies that require recovery actions."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]
            
            # Call Azure OpenAI
            response = self.azure_client.chat.completions.create(
                model="gpt-4o",  # Use appropriate model name
                messages=messages,
                max_tokens=500,
                temperature=0.1
            )
            
            # Parse response
            content = response.choices[0].message.content
            result = self._parse_llm_response(content)
            
            return result
            
        except Exception as e:
            self.get_logger().error(f'Azure OpenAI analysis failed: {e}')
            return self._simulate_analysis(image_path)  # Fallback to simulation
    
    def _simulate_analysis(self, image_path: str) -> dict:
        """Simulate VLM analysis for testing"""
        # Simulate different anomaly types based on analysis count
        anomaly_types = ['missing_block', 'extra_block', 'pose_error', 'normal']
        
        # For testing: cycle through anomalies, with occasional normal
        if self.analysis_count % 8 == 7:  # Every 8th analysis is normal
            return {
                'anomaly': 'normal',
                'action': '',
                'target_id': '',
                'confidence': 0.95,
                'reasoning': 'No anomalies detected - scene appears normal'
            }
        else:
            anomaly_type = anomaly_types[self.analysis_count % 3]  # Cycle through anomaly types
            action_name, target_id = self._get_recovery_action_and_target(anomaly_type)
            return {
                'anomaly': anomaly_type,
                'action': action_name,
                'target_id': target_id,
                'confidence': 0.85 + (self.analysis_count % 3) * 0.05,
                'reasoning': f'Detected {anomaly_type} requiring recovery action'
            }
    
    def _parse_llm_response(self, content: str) -> dict:
        """Parse LLM response to extract anomaly information"""
        try:
            # Try to parse as JSON first
            if content.strip().startswith('{'):
                parsed = json.loads(content)
                
                # Ensure required fields exist
                if 'anomaly' not in parsed:
                    parsed['anomaly'] = 'unknown'
                if 'action' not in parsed:
                    parsed['action'] = ''
                if 'target_id' not in parsed:
                    parsed['target_id'] = ''
                if 'confidence' not in parsed:
                    parsed['confidence'] = 0.5
                if 'reasoning' not in parsed:
                    parsed['reasoning'] = 'LLM response parsing'
                
                return parsed
            
            # Simple text parsing fallback
            content_lower = content.lower()
            
            if any(word in content_lower for word in ['normal', 'no anomaly', 'no problem']):
                return {
                    'anomaly': 'normal',
                    'action': '',
                    'target_id': '',
                    'confidence': 0.9,
                    'reasoning': 'Scene appears normal'
                }
            
            # Detect specific anomaly types
            if 'missing' in content_lower or 'not found' in content_lower:
                anomaly_type = 'missing_block'
            elif 'extra' in content_lower or 'additional' in content_lower:
                anomaly_type = 'extra_block'
            elif 'pose' in content_lower or 'position' in content_lower or 'orientation' in content_lower:
                anomaly_type = 'pose_error'
            else:
                anomaly_type = 'unknown'
            
            action_name, target_id = self._get_recovery_action_and_target(anomaly_type)
            
            return {
                'anomaly': anomaly_type,
                'action': action_name,
                'target_id': target_id,
                'confidence': 0.8,
                'reasoning': content[:100]  # First 100 chars
            }
            
        except Exception as e:
            self.get_logger().warning(f'Failed to parse LLM response: {e}')
            return {
                'anomaly': 'unknown',
                'action': '',
                'target_id': '',
                'confidence': 0.5,
                'reasoning': 'Failed to parse response'
            }
    
    def _handle_anomaly_detected(self, result: dict, image_path: str):
        """Handle detected anomaly by publishing action"""
        anomaly_type = result.get('anomaly', 'unknown')
        confidence = result.get('confidence', 0.0)
        reasoning = result.get('reasoning', '')
        
        action_name, target_id = self._get_recovery_action_and_target(anomaly_type)
        
        if not action_name:
            self.get_logger().warning(f'No recovery action defined for anomaly: {anomaly_type}')
            self._log_detailed('no_action_for_anomaly', {
                'anomaly_type': anomaly_type,
                'image_path': image_path,
                'result': result,
                'available_actions': [action.get('action', '') for action in self.actions]
            })
            return
        
        # Update state
        self.last_anomaly_action = action_name
        
        # Publish anomaly action
        action_msg = Action()
        action_msg.name = action_name
        action_msg.target_id = target_id
        action_msg.timestamp = int(time.time() * 1e9)
        
        self.pub_anomaly.publish(action_msg)
        
        self._log_detailed('anomaly_detected_and_published', {
            'image_path': image_path,
            'anomaly_type': anomaly_type,
            'confidence': confidence,
            'action_name': action_name,
            'target_id': target_id,
            'reasoning': reasoning,
            'message_published': True
        })
        
        self.get_logger().info(f'🚨 Anomaly detected in {Path(image_path).name}: {anomaly_type} → "{action_name}" (published to /recovery_action)')
    
    def _get_recovery_action_and_target(self, anomaly_type: str) -> tuple:
        """Get recovery action name and target ID for anomaly type using action_list.jsonl"""
        
        # Map anomaly types to specific actions from action_list.jsonl
        anomaly_action_mapping = {
            'missing_block': 'Move the red block from the tray to the right dish.',
            'extra_block': 'Transfer the green block from the right dish to the correct dish.',
            'pose_error': 'Transfer the green block from the right dish to the correct dish.',
            'color_mismatch': 'Transfer the red block from the left dish to the correct dish.',
        }
        
        # Get the action name from mapping
        action_name = anomaly_action_mapping.get(anomaly_type, '')
        
        # If action name found, look up target_id from action_list.jsonl
        target_id = ''
        if action_name:
            for action_item in self.actions:
                if action_item.get('action', '') == action_name:
                    target_id = action_item.get('target_id', '')
                    break
        
        # Fallback: if not found in action_list, try first available action
        if not action_name and self.actions:
            first_action = self.actions[0]
            action_name = first_action.get('action', '')
            target_id = first_action.get('target_id', '')
            
            self.get_logger().warning(f'Using fallback action for {anomaly_type}: {action_name}')
        
        self._log_detailed('action_mapping', {
            'anomaly_type': anomaly_type,
            'mapped_action': action_name,
            'mapped_target_id': target_id,
            'available_actions': len(self.actions)
        })
        
        return action_name, target_id
    
    def _cb_verification_request(self, msg: VerificationRequest):
        """Handle verification request from StateManager"""
        self._log_detailed('verification_request_received', {
            'verification_id': msg.verification_id,
            'action_name': msg.action_name,
            'target_id': msg.target_id,
            'request_type': msg.request_type
        })
        
        self.get_logger().info(f'📋 Verification requested: {msg.request_type} for {msg.action_name}')
        
        # Add to verification queue for processing
        self.verification_queue.put(msg)
    
    def _verification_worker(self):
        """Worker thread for handling verification requests"""
        while True:
            try:
                # Get verification request from queue
                request = self.verification_queue.get(timeout=1.0)
                
                # Process verification
                self._process_verification_request(request)
                
                self.verification_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Verification worker error: {e}')
    
    def _process_verification_request(self, request: VerificationRequest):
        """Process a verification request by analyzing current scene"""
        verification_start_time = time.time()
        
        # Analyze current image to check if problem is resolved
        if not self.image_files:
            self._send_verification_result(request, False, 'unknown', 0.0, 'No images available')
            return
        
        # Use current image for verification
        current_image = self.image_files[self.current_image_index]
        
        self._log_detailed('verification_analysis_start', {
            'verification_id': request.verification_id,
            'image_path': current_image,
            'action_name': request.action_name
        })
        
        # Perform VLM analysis specifically for verification
        result = self._analyze_image(current_image)
        verification_duration = time.time() - verification_start_time
        
        if not result:
            self._send_verification_result(request, False, 'unknown', 0.0, 'Analysis failed')
            return
        
        # Determine if the original problem is resolved
        is_resolved = result.get('anomaly') == 'normal'
        anomaly_type = result.get('anomaly', 'unknown')
        confidence = result.get('confidence', 0.0)
        reasoning = result.get('reasoning', '')
        
        # For verification, we want to check if the specific problem is resolved
        if is_resolved:
            description = f'Problem resolved - scene appears normal (confidence: {confidence:.2f})'
        else:
            description = f'{anomaly_type} still detected (confidence: {confidence:.2f}) - {reasoning}'
        
        self._log_detailed('verification_analysis_completed', {
            'verification_id': request.verification_id,
            'duration': verification_duration,
            'is_resolved': is_resolved,
            'anomaly_type': anomaly_type,
            'confidence': confidence,
            'description': description
        })
        
        # Send verification result
        self._send_verification_result(request, is_resolved, anomaly_type, confidence, description)
    
    def _send_verification_result(self, request: VerificationRequest, is_resolved: bool, 
                                 anomaly_type: str, confidence: float, description: str):
        """Send verification result back to StateManager"""
        result_msg = RecoveryVerification()
        result_msg.verification_id = request.verification_id
        result_msg.action_name = request.action_name
        result_msg.target_id = request.target_id
        result_msg.is_resolved = is_resolved
        result_msg.anomaly_type = anomaly_type
        result_msg.confidence = confidence
        result_msg.description = description
        result_msg.timestamp = int(time.time() * 1e9)
        
        self.pub_verification_result.publish(result_msg)
        
        self._log_detailed('verification_result_sent', {
            'verification_id': request.verification_id,
            'is_resolved': is_resolved,
            'anomaly_type': anomaly_type,
            'confidence': confidence,
            'description': description
        })
        
        status_icon = '✅' if is_resolved else '🔄'
        self.get_logger().info(f'{status_icon} Verification result: {anomaly_type} (confidence: {confidence:.2f}) - {description[:50]}')
    
    def destroy_node(self):
        """Clean shutdown"""
        self._log_detailed('system_shutdown', {
            'reason': 'normal_shutdown',
            'analysis_count': self.analysis_count,
            'last_anomaly_action': self.last_anomaly_action,
            'recovery_active': self.recovery_active
        })
        
        # Stop verification worker
        try:
            self.verification_queue.put(None)  # Signal to stop
        except:
            pass
        
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = VLMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('VLMNode shutting down')
        node._log_detailed('system_shutdown', {
            'reason': 'keyboard_interrupt',
            'final_analysis_count': node.analysis_count,
            'final_image_index': node.current_image_index
        })
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()