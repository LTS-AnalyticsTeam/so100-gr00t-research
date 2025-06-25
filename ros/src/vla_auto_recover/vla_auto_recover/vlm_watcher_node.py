#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vla_interfaces.msg import Action, Anomaly
import cv2
from cv_bridge import CvBridge
import numpy as np
import time
import json
import threading
import queue
from queue import Empty
import base64
import openai
import os
import traceback
from datetime import datetime
from dotenv import load_dotenv

def _init_openai_client():
    """Initialize OpenAI client (Azure or OpenAI)"""
    
    load_dotenv('/workspace/config/.env')

    use_azure = os.getenv('USE_AZURE_OPENAI', 'false').lower() == 'true'
    
    try:
        if use_azure:
            from openai import AzureOpenAI
            client = AzureOpenAI(
                api_key=os.getenv('AZURE_OPENAI_API_KEY'),
                api_version=os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview'),
                azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')
            )
            print("Azure OpenAI client initialized")
        else:
            from openai import OpenAI
            client = OpenAI(
                api_key=os.getenv('OPENAI_API_KEY')
            )
            print("OpenAI client initialized")
        
        return client, use_azure
        
    except Exception as e:
        print(f"Warning: OpenAI client not available: {e}")
        return None, use_azure

# Initialize OpenAI client
_client, _use_azure = _init_openai_client()

class VLMWatcher(Node):
    def __init__(self):
        super().__init__('vlm_watcher')
        
        # Configuration parameters
        self.declare_parameter('fps', 5.0)
        self.declare_parameter('num_workers', 4)
        self.declare_parameter('api_timeout', 15.0)
        self.declare_parameter('max_failures', 3)
        self.declare_parameter('failure_reset_time', 300.0)  # 5 minutes
        self.declare_parameter('log_directory', '/tmp/vlm_logs')
        self.declare_parameter('save_anomaly_images', True)
        self.declare_parameter('prompt_file', '/config/prompt.txt')
        self.declare_parameter('action_list_file', '/config/action_list.jsonl')
        
        self.fps = self.get_parameter('fps').value
        self.num_workers = self.get_parameter('num_workers').value
        self.api_timeout = self.get_parameter('api_timeout').value
        self.max_failures = self.get_parameter('max_failures').value
        self.failure_reset_time = self.get_parameter('failure_reset_time').value
        self.log_dir = self.get_parameter('log_directory').value
        self.save_anomaly_images = self.get_parameter('save_anomaly_images').value
        self.prompt_file = self.get_parameter('prompt_file').value
        self.action_list_file = self.get_parameter('action_list_file').value
        
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        if self.save_anomaly_images:
            os.makedirs(os.path.join(self.log_dir, 'images'), exist_ok=True)
        
        # Processing control
        self.min_interval = 1.0 / self.fps
        self.last_sent = 0
        self.bridge = CvBridge()
        
        # API failure tracking
        self.api_failure_count = 0
        self.last_failure_time = 0
        self.last_successful_call = time.time()
        self.safe_mode = False
        
        # Statistics tracking
        self.stats = {
            'total_images_received': 0,
            'total_images_processed': 0,
            'total_api_calls': 0,
            'successful_api_calls': 0,
            'failed_api_calls': 0,
            'anomalies_detected': 0,
            'last_reset_time': time.time()
        }
        
        # Load configuration
        self.prompt = self._load_prompt()
        self.actions = self._load_actions()
        
        # Thread-safe queues
        self.q_req = queue.Queue(maxsize=20)
        self.q_res = queue.Queue()
        
        # Publishers
        self.pub_act = self.create_publisher(Action, '/recovery_action', 10)
        self.pub_anomaly = self.create_publisher(Anomaly, '/anomaly_detected', 10)
        
        # Subscribers
        self.sub_img = self.create_subscription(Image, '/camera/image', self._cb_img, 10)
        
        # Start worker threads
        self.workers = []
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker, daemon=True, name=f'VLMWorker-{i}')
            worker.start()
            self.workers.append(worker)
        
        # Timer for processing results
        self.timer_res = self.create_timer(0.1, self._timer_res)
        
        # Timer for periodic statistics and health checks
        self.timer_stats = self.create_timer(30.0, self._timer_stats)
        
        # Logging setup
        self.setup_logging()
        
        self.get_logger().info(f'VLMWatcher initialized with {self.num_workers} workers at {self.fps} FPS')
        self._log_detailed('system_startup', {
            'fps': self.fps,
            'num_workers': self.num_workers,
            'actions_loaded': len(self.actions),
            'prompt_length': len(self.prompt)
        })
    
    def setup_logging(self):
        """Setup detailed logging system"""
        self.log_file = os.path.join(
            self.log_dir, 
            f"vlm_watcher_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        self.get_logger().info(f'Logging to: {self.log_file}')
    
    def _log_detailed(self, event_type: str, details: dict):
        """Write detailed log entry"""
        timestamp = time.time()
        log_entry = {
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp).isoformat(),
            'event_type': event_type,
            'safe_mode': self.safe_mode,
            'api_failure_count': self.api_failure_count,
            'queue_sizes': {
                'request': self.q_req.qsize(),
                'response': self.q_res.qsize()
            },
            'stats': self.stats.copy(),
            'details': details
        }
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, indent=None) + '\n')
        except Exception as e:
            self.get_logger().error(f'Failed to write log: {e}')
    
    def _load_prompt(self) -> str:
        """Load prompt from file with error handling"""
        try:
            with open(self.prompt_file, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
            self.get_logger().info(f'Loaded prompt from {self.prompt_file} ({len(prompt)} chars)')
            return prompt
        except Exception as e:
            self.get_logger().error(f'Failed to load prompt from {self.prompt_file}: {e}')
            return "Analyze this image for any anomalies and suggest recovery actions."
    
    def _load_actions(self) -> list:
        """Load action list from file with error handling"""
        actions = []
        try:
            with open(self.action_list_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        j = json.loads(line.strip())
                        name = j.get('action')
                        tid = j.get('target_id', '')
                        if name:
                            actions.append({'action': name, 'target_id': tid})
                    except json.JSONDecodeError as e:
                        self.get_logger().warning(f'Invalid JSON in {self.action_list_file} line {line_num}: {e}')
            
            self.get_logger().info(f'Loaded {len(actions)} actions from {self.action_list_file}')
            self._log_detailed('actions_loaded', {
                'action_count': len(actions),
                'action_file': self.action_list_file,
                'actions': actions
            })
            
        except Exception as e:
            self.get_logger().error(f'Failed to load actions from {self.action_list_file}: {e}')
        
        return actions
    
    def _cb_img(self, msg: Image):
        """Handle incoming camera images"""
        self.stats['total_images_received'] += 1
        
        if self.safe_mode:
            # Skip processing in safe mode
            return
        
        now = time.time()
        if now - self.last_sent < self.min_interval:
            return
        
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # Encode as JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            result, jpg_data = cv2.imencode('.jpg', cv_image, encode_param)
            
            if not result:
                self.get_logger().warning('Failed to encode image as JPEG')
                return
            
            jpg_bytes = jpg_data.tobytes()
            
            # Try to add to processing queue
            try:
                self.q_req.put_nowait(jpg_bytes)
                self.last_sent = now
                self.stats['total_images_processed'] += 1
                
                # Log queue status periodically
                if self.stats['total_images_processed'] % 50 == 0:
                    self._log_detailed('processing_status', {
                        'images_processed': self.stats['total_images_processed'],
                        'queue_size': self.q_req.qsize(),
                        'processing_rate': self.stats['total_images_processed'] / (now - self.stats['last_reset_time'])
                    })
                    
            except queue.Full:
                self.get_logger().warning('Processing queue full, dropping image')
                self._log_detailed('queue_full', {
                    'queue_size': self.q_req.qsize(),
                    'max_size': self.q_req.maxsize
                })
        
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
            self._log_detailed('image_processing_error', {
                'error_type': type(e).__name__,
                'error_message': str(e)
            })
    
    def _worker(self):
        """Worker thread for gpt-4.1 API calls"""
        thread_name = threading.current_thread().name
        
        while True:
            try:
                # Get image from queue with timeout
                jpg = self.q_req.get(timeout=1.0)
            except Empty:
                continue
            
            # Process image
            start_time = time.time()
            success = self._process_image(jpg, thread_name)
            processing_time = time.time() - start_time
            
            if success:
                self.stats['successful_api_calls'] += 1
                self.last_successful_call = time.time()
                # Reset failure count on success
                if self.api_failure_count > 0:
                    self._log_detailed('api_recovery', {
                        'previous_failure_count': self.api_failure_count,
                        'worker_thread': thread_name
                    })
                    self.api_failure_count = 0
                    
            else:
                self.stats['failed_api_calls'] += 1
                self._handle_api_failure(thread_name)
            
            self.stats['total_api_calls'] += 1
            
            # Log processing metrics
            if self.stats['total_api_calls'] % 10 == 0:
                self._log_detailed('api_metrics', {
                    'processing_time': processing_time,
                    'success_rate': self.stats['successful_api_calls'] / self.stats['total_api_calls'],
                    'worker_thread': thread_name
                })
    
    def _process_image(self, jpg: bytes, worker_name: str) -> bool:
        """Process single image with gpt-4.1 API"""
        try:
            # Encode image to base64
            b64 = base64.b64encode(jpg).decode('utf-8')
            
            # Prepare action list text
            actions_text = "Available actions:\\n" + "\\n".join(
                f"- {a['action']} (target_id={a['target_id']})" 
                for a in self.actions
            )
            
            # Prepare messages
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": self.prompt},
                    {"type": "text", "text": actions_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]}
            ]
            
            # Make API call with timeout
            response = _client.chat.completions.create(
                model="gpt-4.1",
                messages=messages,
                timeout=self.api_timeout
            )
            
            content = response.choices[0].message.content.strip()
            
            # Add to results queue
            self.q_res.put({
                'content': content,
                'timestamp': time.time(),
                'worker': worker_name,
                'image_size': len(jpg)
            })
            
            return True
            
        except openai.APITimeoutError:
            self.get_logger().warning(f'gpt-4.1 API timeout in {worker_name}')
            self._log_detailed('api_timeout', {
                'worker': worker_name,
                'timeout_duration': self.api_timeout
            })
            return False
            
        except openai.RateLimitError:
            self.get_logger().warning(f'gpt-4.1 rate limit exceeded in {worker_name}')
            self._log_detailed('api_rate_limit', {
                'worker': worker_name
            })
            time.sleep(1.0)  # Brief pause on rate limit
            return False
            
        except openai.APIError as e:
            self.get_logger().error(f'gpt-4.1 API error in {worker_name}: {e}')
            self._log_detailed('api_error', {
                'worker': worker_name,
                'error_type': type(e).__name__,
                'error_message': str(e)
            })
            return False
            
        except Exception as e:
            self.get_logger().error(f'Unexpected error in {worker_name}: {e}')
            self._log_detailed('worker_error', {
                'worker': worker_name,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc()
            })
            return False
    
    def _handle_api_failure(self, worker_name: str):
        """Handle API failure with progressive backoff"""
        self.api_failure_count += 1
        self.last_failure_time = time.time()
        
        self._log_detailed('api_failure', {
            'failure_count': self.api_failure_count,
            'max_failures': self.max_failures,
            'worker': worker_name,
            'time_since_last_success': self.last_failure_time - self.last_successful_call
        })
        
        if self.api_failure_count >= self.max_failures:
            self._enter_safe_mode()
    
    def _enter_safe_mode(self):
        """Enter safe mode to prevent further API calls"""
        if not self.safe_mode:
            self.safe_mode = True
            self.get_logger().error(f'Entering safe mode after {self.api_failure_count} consecutive failures')
            
            # Publish system error anomaly
            anomaly = Anomaly()
            anomaly.type = "system_error"
            anomaly.severity = "critical"
            anomaly.message = f"VLM system entering safe mode after {self.api_failure_count} API failures"
            self.pub_anomaly.publish(anomaly)
            
            self._log_detailed('safe_mode_entered', {
                'failure_count': self.api_failure_count,
                'time_since_last_success': time.time() - self.last_successful_call
            })
    
    def _timer_res(self):
        """Process API results"""
        processed_count = 0
        
        while not self.q_res.empty() and processed_count < 5:  # Limit per cycle
            try:
                result = self.q_res.get_nowait()
                content = result['content']
                timestamp = result['timestamp']
                
                self._log_detailed('api_response_received', {
                    'response_length': len(content),
                    'worker': result['worker'],
                    'processing_delay': time.time() - timestamp
                })
                
                try:
                    j = json.loads(content)
                    anomaly_type = j.get('anomaly', 'normal')
                    
                    if anomaly_type != 'normal':
                        self.stats['anomalies_detected'] += 1
                        action_name = j.get('action', '')
                        target_id = j.get('target_id', '')
                        confidence = j.get('confidence', 0.0)
                        reasoning = j.get('reasoning', '')
                        
                        self._log_detailed('anomaly_detected', {
                            'anomaly_type': anomaly_type,
                            'action_name': action_name,
                            'target_id': target_id,
                            'confidence': confidence,
                            'reasoning': reasoning,
                            'response_content': content
                        })
                        
                        # Save anomaly image if enabled
                        if self.save_anomaly_images:
                            self._save_anomaly_image(anomaly_type, action_name)
                        
                        # Publish recovery action
                        action = Action()
                        action.name = action_name
                        action.target_id = target_id
                        self.pub_act.publish(action)
                        
                        # Publish anomaly notification
                        anomaly = Anomaly()
                        anomaly.type = anomaly_type
                        anomaly.severity = "medium"
                        anomaly.message = f"Detected {anomaly_type}: {action_name}"
                        self.pub_anomaly.publish(anomaly)
                        
                        self.get_logger().info(f'Anomaly detected: {anomaly_type} → {action_name}')
                    
                except json.JSONDecodeError as e:
                    self.get_logger().warning(f'Invalid JSON from gpt-4.1: {content[:100]}...')
                    self._log_detailed('invalid_json_response', {
                        'error': str(e),
                        'content_preview': content[:200],
                        'content_length': len(content)
                    })
                
                processed_count += 1
                
            except Empty:
                break
    
    def _save_anomaly_image(self, anomaly_type: str, action_name: str):
        """Save current camera image when anomaly is detected"""
        # This would need access to the latest camera image
        # For now, we log the intent
        self._log_detailed('anomaly_image_save_requested', {
            'anomaly_type': anomaly_type,
            'action_name': action_name,
            'timestamp': time.time()
        })
    
    def _timer_stats(self):
        """Periodic statistics and health monitoring"""
        current_time = time.time()
        uptime = current_time - self.stats['last_reset_time']
        
        # Check if we should exit safe mode
        if self.safe_mode and (current_time - self.last_failure_time) > self.failure_reset_time:
            self.safe_mode = False
            self.api_failure_count = 0
            self.get_logger().info('Exiting safe mode - system recovered')
            self._log_detailed('safe_mode_exited', {
                'time_since_failure': current_time - self.last_failure_time,
                'reset_threshold': self.failure_reset_time
            })
        
        # Calculate rates
        processing_rate = self.stats['total_images_processed'] / uptime if uptime > 0 else 0
        api_rate = self.stats['total_api_calls'] / uptime if uptime > 0 else 0
        success_rate = (self.stats['successful_api_calls'] / self.stats['total_api_calls'] 
                       if self.stats['total_api_calls'] > 0 else 1.0)
        
        self._log_detailed('periodic_health_check', {
            'uptime': uptime,
            'processing_rate': processing_rate,
            'api_rate': api_rate,
            'success_rate': success_rate,
            'queue_status': {
                'request_size': self.q_req.qsize(),
                'response_size': self.q_res.qsize()
            },
            'safe_mode': self.safe_mode,
            'time_since_last_success': current_time - self.last_successful_call
        })
        
        self.get_logger().info(
            f'Stats: {self.stats["total_images_processed"]} images, '
            f'{self.stats["anomalies_detected"]} anomalies, '
            f'{success_rate:.1%} API success rate'
        )

def main(args=None):
    rclpy.init(args=args)
    node = VLMWatcher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('VLMWatcher shutting down')
        node._log_detailed('system_shutdown', {
            'reason': 'keyboard_interrupt',
            'final_stats': node.stats
        })
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()