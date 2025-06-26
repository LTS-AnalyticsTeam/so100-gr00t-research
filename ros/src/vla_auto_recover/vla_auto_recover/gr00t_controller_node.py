#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from vla_interfaces.msg import Action, RecoveryStatus
import time
import json
import os
import traceback
import numpy as np
import torch
from datetime import datetime

# Import your existing classes (adjust import paths as needed)
try:
    from so100_robot import SO100Robot
    from service import ExternalRobotInferenceClient
except ImportError as e:
    print(f"Warning: Could not import robot interfaces: {e}")
    # For development/testing without actual robot
    class SO100Robot:
        def __init__(self, calibrate=False, enable_camera=True, cam_idx=1):
            pass
        def activate(self):
            return self
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def get_current_img(self): 
            return np.zeros((480, 640, 3), dtype=np.uint8)
        def get_current_state(self): 
            return np.zeros(7)
        def set_target_state(self, state): 
            pass
    
    class ExternalRobotInferenceClient:
        def get_action(self, img, state): 
            return {'action.single_arm': np.random.random((16, 7)), 'action.gripper': np.random.random((16, 1))}

class GR00TController(Node):
    def __init__(self):
        super().__init__('gr00t_controller')
        
        # Configuration parameters
        self.declare_parameter('policy_server_url', 'http://localhost:8000')
        self.declare_parameter('execution_delay', 0.02)
        self.declare_parameter('vla_execution_frequency', 10.0)  # Hz
        self.declare_parameter('log_directory', 'run_logs/gr00t_logs')
        self.declare_parameter('save_recovery_images', True)
        
        self.policy_server_url = self.get_parameter('policy_server_url').value
        self.execution_delay = self.get_parameter('execution_delay').value
        self.vla_frequency = self.get_parameter('vla_execution_frequency').value
        self.log_dir = self.get_parameter('log_directory').value
        self.save_images = self.get_parameter('save_recovery_images').value
        
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        if self.save_images:
            os.makedirs(os.path.join(self.log_dir, 'images'), exist_ok=True)
        
        # VLA System state
        self.vla_active = True          # VLA通常実行フラグ
        self.paused = False             # リカバリー用ポーズフラグ
        self.current_vla_action = None  # 現在のVLAアクション
        
        # Recovery state
        self.current_recovery_action = None
        self.recovery_start_time = 0
        self.is_executing_recovery = False
        
        # Robot interface initialization
        try:
            self.robot = SO100Robot(calibrate=False, enable_camera=True, cam_idx=1)
            self.robot_ctx = self.robot.activate()
            self.robot_ctx.__enter__()
            
            # GR00T policy client (for recovery actions)
            self.recovery_policy = ExternalRobotInferenceClient()
            
            # VLA policy client (for normal operations)
            # TODO: Initialize VLA policy here
            # self.vla_policy = YourVLAPolicyClient()
            self.vla_policy = ExternalRobotInferenceClient()  # Temporary - replace with actual VLA
            
            self.robot_connected = True
            self.get_logger().info(f'Robot interfaces initialized successfully')
            
        except Exception as e:
            self.robot_connected = False
            self.get_logger().error(f'Failed to initialize robot interface: {e}')
            self._log_detailed('initialization_error', {
                'error': str(e),
                'policy_server_url': self.policy_server_url
            })
        
        # Publishers
        self.pub_recovery_status = self.create_publisher(RecoveryStatus, '/recovery_status', 10)
        
        # Subscribers
        self.sub_pause = self.create_subscription(Bool, '/vla_pause', self._cb_pause, 10)
        self.sub_action = self.create_subscription(Action, '/recovery_action', self._cb_action, 10)
        
        # VLA execution timer (runs during normal operation)
        vla_timer_period = 1.0 / self.vla_frequency
        self.vla_timer = self.create_timer(vla_timer_period, self._vla_normal_execution)
        
        # Logging setup
        self.setup_logging()
        
        self.get_logger().info('GR00TController (Pattern1) initialized')
        self._log_detailed('system_startup', {
            'robot_connected': self.robot_connected,
            'vla_frequency': self.vla_frequency,
            'pattern': 'Pattern1_VLA_integrated'
        })
    
    def setup_logging(self):
        """Setup detailed logging system"""
        self.log_file = os.path.join(
            self.log_dir, 
            f"gr00t_controller_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        self.get_logger().info(f'Logging to: {self.log_file}')
    
    def _log_detailed(self, event_type: str, details: dict):
        """Write detailed log entry"""
        timestamp = time.time()
        log_entry = {
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp).isoformat(),
            'event_type': event_type,
            'vla_active': self.vla_active,
            'paused': self.paused,
            'is_executing_recovery': self.is_executing_recovery,
            'current_recovery_action': self.current_recovery_action,
            'current_vla_action': self.current_vla_action,
            'details': details
        }
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, indent=None) + '\n')
        except Exception as e:
            self.get_logger().error(f'Failed to write log: {e}')
    
    def _cb_pause(self, msg: Bool):
        """Handle VLA pause/resume commands"""
        old_paused = self.paused
        self.paused = msg.data
        
        self._log_detailed('vla_pause_received', {
            'old_paused': old_paused,
            'new_paused': self.paused,
            'state_change': old_paused != self.paused
        })
        
        if self.paused != old_paused:
            if self.paused:
                self.get_logger().info('VLA paused for recovery mode')
                # VLAの通常実行を停止
                self.vla_active = False
                self._stop_current_vla_action()
                
            else:
                self.get_logger().info('VLA resumed from recovery mode')
                # VLAの通常実行を再開
                self.vla_active = True
                self._resume_vla_execution()
                
                # Reset recovery state when resuming
                if self.current_recovery_action:
                    self._log_detailed('recovery_session_end', {
                        'action_name': self.current_recovery_action,
                        'duration': time.time() - self.recovery_start_time
                    })
                    self.current_recovery_action = None
                    self.is_executing_recovery = False
    
    def _vla_normal_execution(self):
        """Normal VLA execution loop - runs at specified frequency"""
        if not self.vla_active or self.paused or not self.robot_connected:
            return
        
        if self.is_executing_recovery:
            # Skip VLA execution during recovery
            return
        
        try:
            # Get current robot state
            img = self.robot.get_current_img()
            state = self.robot.get_current_state()
            
            # Get action from VLA policy (normal task execution)
            # TODO: Implement your VLA task specification here
            # For example: task_specification = "sort colored blocks"
            vla_action_dict = self.vla_policy.get_action(img, state)
            
            if 'action.single_arm' in vla_action_dict:
                # Execute VLA action (single step or small sequence)
                action_sequence = vla_action_dict['action.single_arm']
                gripper_sequence = vla_action_dict.get('action.gripper', np.zeros((action_sequence.shape[0], 1)))
                
                # For VLA, typically execute just the first step or a few steps
                current_step = action_sequence[0] if len(action_sequence.shape) > 1 else action_sequence
                current_gripper = gripper_sequence[0] if len(gripper_sequence.shape) > 1 else gripper_sequence
                
                # Concatenate action components
                MOD_KEYS = ['single_arm', 'gripper']
                concat = np.concatenate([current_step, current_gripper], axis=0)
                
                # Send to robot
                self.robot.set_target_state(torch.from_numpy(concat))
                
                # Update current action for logging
                self.current_vla_action = f"VLA_step_{int(time.time()*10)%1000}"
                
                # Log VLA execution periodically
                if int(time.time() * 10) % 50 == 0:  # Every 5 seconds at 10Hz
                    self._log_detailed('vla_execution', {
                        'action_shape': action_sequence.shape,
                        'gripper_shape': gripper_sequence.shape,
                        'execution_step': self.current_vla_action
                    })
            
        except Exception as e:
            self.get_logger().error(f'VLA execution error: {e}')
            self._log_detailed('vla_execution_error', {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc()
            })
    
    def _stop_current_vla_action(self):
        """Stop current VLA action safely"""
        try:
            # Hold current position
            current_state = self.robot.get_current_state()
            self.robot.set_target_state(torch.from_numpy(current_state))
            
            self._log_detailed('vla_stopped', {
                'previous_action': self.current_vla_action,
                'current_state_shape': current_state.shape if current_state is not None else None
            })
            
            self.current_vla_action = None
            self.get_logger().info('VLA stopped - robot holding current position')
            
        except Exception as e:
            self.get_logger().error(f'Error stopping VLA: {e}')
            self._log_detailed('vla_stop_error', {
                'error_type': type(e).__name__,
                'error_message': str(e)
            })
    
    def _resume_vla_execution(self):
        """Resume VLA execution"""
        self._log_detailed('vla_resumed', {
            'resume_time': time.time(),
            'was_executing_recovery': self.is_executing_recovery
        })
        
        self.get_logger().info('VLA execution resumed')
        # The actual resumption happens in the next _vla_normal_execution() call
    
    def _cb_action(self, msg: Action):
        """Handle recovery action execution"""
        if not msg.name:
            self.get_logger().warning('Received empty action name')
            return
        
        self._log_detailed('recovery_action_received', {
            'action_name': msg.name,
            'target_id': msg.target_id,
            'paused': self.paused,
            'vla_active': self.vla_active,
            'robot_connected': self.robot_connected
        })
        
        if not self.paused:
            self.get_logger().warning(f'Received recovery action but not paused: {msg.name}')
            return
        
        if self.is_executing_recovery:
            self.get_logger().warning(f'Already executing recovery, ignoring: {msg.name}')
            return
        
        if not self.robot_connected:
            self._publish_recovery_status('failed', 0.0, msg.name, msg.target_id, 
                                        'Robot interface not connected')
            return
        
        # Start recovery execution
        self.current_recovery_action = msg.name
        self.recovery_start_time = time.time()
        self.is_executing_recovery = True
        
        self.get_logger().info(f'Starting recovery action: {msg.name}')
        self._publish_recovery_status('executing', 0.0, msg.name, msg.target_id)
        
        # Execute recovery in separate thread to avoid blocking
        self._execute_recovery_action(msg.name, msg.target_id)
    
    def _execute_recovery_action(self, action_name: str, target_id: str):
        """Execute the actual recovery action"""
        try:
            self._log_detailed('recovery_execution_start', {
                'action_name': action_name,
                'target_id': target_id,
                'start_time': self.recovery_start_time,
                'vla_was_active': self.vla_active
            })
            
            # Save initial state image if enabled
            if self.save_images:
                self._save_recovery_image('start', action_name)
            
            # Get current robot state
            img = self.robot.get_current_img()
            state = self.robot.get_current_state()
            
            self._log_detailed('robot_state_captured', {
                'image_shape': img.shape if img is not None else None,
                'state_shape': state.shape if state is not None else None,
                'action_name': action_name
            })
            
            # Get action from GR00T recovery policy server
            self._publish_recovery_status('executing', 0.1, action_name, target_id)
            
            # Use recovery policy (different from VLA policy)
            action_dict = self.recovery_policy.get_action(img, state)
            
            if 'action.single_arm' not in action_dict:
                raise ValueError("Invalid action_dict: missing 'action.single_arm' key")
            
            action_sequence = action_dict['action.single_arm']
            gripper_sequence = action_dict.get('action.gripper', np.zeros((action_sequence.shape[0], 1)))
            total_steps = action_sequence.shape[0]
            
            self._log_detailed('recovery_sequence_generated', {
                'total_steps': total_steps,
                'sequence_shape': action_sequence.shape,
                'gripper_shape': gripper_sequence.shape,
                'action_name': action_name
            })
            
            self.get_logger().info(f'Executing {total_steps} steps for recovery action')
            
            # Execute action sequence step by step
            MOD_KEYS = ['single_arm', 'gripper']
            for i in range(total_steps):
                if not self.is_executing_recovery:  # Check if cancelled
                    self._log_detailed('recovery_cancelled', {
                        'step': i,
                        'total_steps': total_steps,
                        'action_name': action_name
                    })
                    return
                
                # Create target state by concatenating components
                concat = np.concatenate([
                    action_sequence[i],
                    gripper_sequence[i] if i < len(gripper_sequence) else gripper_sequence[-1]
                ], axis=0)
                
                # Execute step
                self.robot.set_target_state(torch.from_numpy(concat))
                
                # Update progress
                progress = (i + 1) / total_steps
                if i % 4 == 0 or i == total_steps - 1:  # Update every 4 steps or at end
                    self._publish_recovery_status('executing', progress, action_name, target_id)
                
                # Log progress
                if i % 8 == 0 or i == total_steps - 1:  # Log every 8 steps or at end
                    self._log_detailed('recovery_progress', {
                        'step': i + 1,
                        'total_steps': total_steps,
                        'progress': progress,
                        'action_name': action_name
                    })
                
                time.sleep(self.execution_delay)
            
            # Save final state image if enabled
            if self.save_images:
                self._save_recovery_image('end', action_name)
            
            execution_time = time.time() - self.recovery_start_time
            
            self._log_detailed('recovery_execution_completed', {
                'action_name': action_name,
                'target_id': target_id,
                'execution_time': execution_time,
                'total_steps': total_steps
            })
            
            # ロボットアクション完了 - StateManagerに「executing」ステータス（progress=1.0）を送信
            # StateManagerがVLM検証を開始する
            self._publish_recovery_status('executing', 1.0, action_name, target_id)
            self.get_logger().info(f'Recovery action sequence completed: {action_name} ({execution_time:.2f}s)')
            self.get_logger().info('Robot action completed - StateManager will start VLM verification...')
            
        except Exception as e:
            execution_time = time.time() - self.recovery_start_time
            error_msg = f'{type(e).__name__}: {str(e)}'
            
            self._log_detailed('recovery_execution_failed', {
                'action_name': action_name,
                'target_id': target_id,
                'execution_time': execution_time,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc()
            })
            
            # Report failure
            self._publish_recovery_status('failed', 0.0, action_name, target_id, error_msg)
            self.get_logger().error(f'Recovery action failed: {action_name} - {error_msg}')
            
            # Reset recovery state only on failure
            self.is_executing_recovery = False
    
    def _publish_recovery_status(self, status: str, progress: float, action_name: str = '', 
                               target_id: str = '', error_message: str = ''):
        """Publish recovery status to StateManager"""
        msg = RecoveryStatus()
        msg.completed = (status == 'completed')
        msg.failed = (status == 'failed')
        msg.action_name = action_name
        msg.target_id = target_id
        msg.progress = max(0.0, min(1.0, progress))  # Clamp to [0, 1]
        msg.status = status
        msg.error_message = error_message
        msg.start_time = int(self.recovery_start_time * 1e9) if self.recovery_start_time > 0 else 0
        msg.completion_time = int(time.time() * 1e9) if status in ['completed', 'failed'] else 0
        
        self.pub_recovery_status.publish(msg)
        
        self._log_detailed('recovery_status_published', {
            'status': status,
            'progress': progress,
            'action_name': action_name,
            'target_id': target_id,
            'error_message': error_message
        })
    
    def _save_recovery_image(self, phase: str, action_name: str):
        """Save recovery image for debugging"""
        try:
            img = self.robot.get_current_img()
            if img is not None:
                timestamp = int(time.time() * 1000)
                filename = f"recovery_{phase}_{timestamp}_{action_name.replace(' ', '_')[:30]}.jpg"
                filepath = os.path.join(self.log_dir, 'images', filename)
                
                # Convert numpy array to image and save
                import cv2
                cv2.imwrite(filepath, img)
                
                self._log_detailed('recovery_image_saved', {
                    'phase': phase,
                    'action_name': action_name,
                    'filepath': filepath,
                    'image_shape': img.shape
                })
                
        except Exception as e:
            self.get_logger().warning(f'Failed to save recovery image: {e}')
    
    def destroy_node(self):
        """Clean shutdown"""
        self._log_detailed('system_shutdown_start', {
            'vla_active': self.vla_active,
            'is_executing_recovery': self.is_executing_recovery
        })
        
        try:
            # Stop VLA execution
            self.vla_active = False
            
            # Stop any ongoing recovery
            self.is_executing_recovery = False
            
            # Clean up robot context
            if hasattr(self, 'robot_ctx'):
                self.robot_ctx.__exit__(None, None, None)
                
        except Exception as e:
            self.get_logger().error(f'Error during shutdown: {e}')
        finally:
            super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = GR00TController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('GR00TController shutting down')
        node._log_detailed('system_shutdown', {
            'reason': 'keyboard_interrupt',
            'was_executing_recovery': node.is_executing_recovery,
            'vla_was_active': node.vla_active
        })
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()