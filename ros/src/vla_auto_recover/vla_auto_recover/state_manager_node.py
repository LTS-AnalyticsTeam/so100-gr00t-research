#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
from vla_interfaces.msg import Action, RecoveryStatus, Anomaly
import time
import json
import os
from datetime import datetime
from enum import Enum

class SystemState(Enum):
    NORMAL = "Normal"
    ANOMALY_DETECTED = "AnomalyDetected" 
    RECOVERING = "Recovering"

class StateManager(Node):
    def __init__(self):
        super().__init__('state_manager')
        
        # Configuration parameters
        self.declare_parameter('log_directory', 'run_logs/state_manager_logs')
        
        self.log_dir = self.get_parameter('log_directory').value
        
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # State management
        self.current_state = SystemState.NORMAL
        self.previous_state = SystemState.NORMAL
        self.state_change_time = time.time()
        
        # Recovery tracking
        self.recovery_info = {
            'action_name': '',
            'target_id': '',
            'start_time': 0,
            'status': '',
            'progress': 0.0,
            'last_update': 0
        }
        
        # Publishers
        self.pub_pause = self.create_publisher(Bool, '/vla_pause', 10)
        
        # Subscribers
        self.sub_anomaly = self.create_subscription(Action, '/anomaly_detected', self._cb_anomaly, 10)
        self.sub_recovery_status = self.create_subscription(RecoveryStatus, '/recovery_status', self._cb_recovery_status, 10)
        
        # Health monitoring timer
        self.health_timer = self.create_timer(30.0, self._health_check)
        
        # Logging setup
        self.setup_logging()
        
        self.get_logger().info('StateManager initialized')
        self._log_detailed('system_startup', {
            'initial_state': self.current_state.value
        })
    
    def setup_logging(self):
        """Setup detailed logging system"""
        self.log_file = os.path.join(
            self.log_dir, 
            f"state_manager_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        self.get_logger().info(f'Logging to: {self.log_file}')
    
    def _log_detailed(self, event_type: str, details: dict):
        """Write detailed log entry"""
        timestamp = time.time()
        log_entry = {
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp).isoformat(),
            'event_type': event_type,
            'current_state': self.current_state.value,
            'details': details,
            'recovery_info': self.recovery_info.copy()
        }
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, indent=None) + '\n')
        except Exception as e:
            self.get_logger().error(f'Failed to write log: {e}')
    
    def _change_state(self, new_state: SystemState, reason: str):
        """Change system state with logging"""
        if new_state != self.current_state:
            self.previous_state = self.current_state
            self.current_state = new_state
            self.state_change_time = time.time()
            
            self._log_detailed('state_change', {
                'old_state': self.previous_state.value,
                'new_state': new_state.value,
                'reason': reason,
                'transition_time': self.state_change_time
            })
            
            self.get_logger().info(f'State: {self.previous_state.value} → {new_state.value} ({reason})')
    
    def _cb_anomaly(self, msg: Action):
        """Handle anomaly detection from VLM"""
        self._log_detailed('anomaly_received', {
            'action_name': msg.name,
            'target_id': msg.target_id,
            'current_state': self.current_state.value,
            'time_since_last_recovery': time.time() - self.recovery_info.get('start_time', 0)
        })
        
        if self.current_state == SystemState.RECOVERING:
            # Already handling recovery - ignore new anomalies
            self._log_detailed('anomaly_ignored', {
                'ignored_action': msg.name,
                'ignored_target': msg.target_id,
                'reason': 'already_recovering',
                'current_action': self.recovery_info.get('action_name', 'none')
            })
            self.get_logger().warning(f'Anomaly ignored (already recovering): {msg.name}')
            return
        
        # Start new recovery process
        self._start_recovery(msg.name, msg.target_id)
    
    def _start_recovery(self, action_name: str, target_id: str):
        """Start recovery process"""
        self._change_state(SystemState.RECOVERING, f'Anomaly detected: {action_name}')
        
        # Update recovery info
        self.recovery_info.update({
            'action_name': action_name,
            'target_id': target_id,
            'start_time': time.time(),
            'status': 'requested',
            'progress': 0.0,
            'last_update': time.time(),
            'verification_start_time': 0,
            'verification_attempts': 0
        })
        
        # Pause VLA
        pause_msg = Bool()
        pause_msg.data = True
        self.pub_pause.publish(pause_msg)
        
        self._log_detailed('vla_pause_command', {
            'pause_state': True,
            'reason': 'recovery_start',
            'action_name': action_name
        })
        
        self.get_logger().info(f'🚨 Recovery started: {action_name}')
        self.get_logger().info('📋 VLA paused for recovery mode')
    
    def _cb_recovery_status(self, msg: RecoveryStatus):
        """Handle recovery status updates from GR00T"""
        self._log_detailed('recovery_status_received', {
            'status': msg.status,
            'action_name': msg.action_name,
            'target_id': msg.target_id,
            'progress': msg.progress,
            'completed': msg.completed,
            'failed': msg.failed,
            'error_message': msg.error_message,
            'execution_time': time.time() - self.recovery_info.get('start_time', time.time())
        })
        
        # Update recovery info
        self.recovery_info.update({
            'action_name': msg.action_name,
            'target_id': msg.target_id,
            'status': msg.status,
            'progress': msg.progress,
            'last_update': time.time()
        })
        
        if msg.failed:
            # Recovery failed - resume VLA
            self._end_recovery(success=False, reason=f'Recovery failed: {msg.error_message}')
            
        elif msg.status == 'executing' and msg.progress >= 1.0:
            # Robot action sequence completed - recovery successful
            self._end_recovery(success=True, reason='Robot action completed successfully')
    
    
    def _end_recovery(self, success: bool, reason: str):
        """End recovery process and resume VLA"""
        # Update recovery status
        final_status = 'completed' if success else 'failed'
        self.recovery_info['status'] = final_status
        
        # Resume VLA
        pause_msg = Bool()
        pause_msg.data = False
        self.pub_pause.publish(pause_msg)
        
        # Change state back to normal
        self._change_state(SystemState.NORMAL, reason)
        
        recovery_duration = time.time() - self.recovery_info['start_time']
        
        self._log_detailed('recovery_completed', {
            'success': success,
            'reason': reason,
            'total_duration': recovery_duration,
            'action_name': self.recovery_info['action_name'],
            'final_status': final_status
        })
        
        self._log_detailed('vla_pause_command', {
            'pause_state': False,
            'reason': 'recovery_end',
            'success': success,
            'current_action': self.recovery_info['action_name']
        })
        
        if success:
            self.get_logger().info(f'✅ Recovery completed successfully: {self.recovery_info["action_name"]} ({recovery_duration:.1f}s)')
        else:
            self.get_logger().error(f'❌ Recovery failed: {self.recovery_info["action_name"]} - {reason}')
        
        self.get_logger().info('📋 VLA resumed from recovery mode')
    
    def _health_check(self):
        """Periodic health check and system status logging"""
        uptime = time.time() - self.state_change_time
        
        self._log_detailed('system_health', {
            'current_state': self.current_state.value,
            'uptime': uptime,
            'last_recovery_time': self.recovery_info.get('start_time', 0),
            'recovery_info': self.recovery_info.copy()
        })
    
    def destroy_node(self):
        """Clean shutdown"""
        self._log_detailed('system_shutdown', {
            'reason': 'normal_shutdown',
            'final_state': self.current_state.value
        })
        
        # Ensure VLA is not left paused
        if self.current_state != SystemState.NORMAL:
            pause_msg = Bool()
            pause_msg.data = False
            self.pub_pause.publish(pause_msg)
            self.get_logger().info('Emergency VLA resume on shutdown')
        
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = StateManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('StateManager shutting down')
        node._log_detailed('system_shutdown', {
            'reason': 'keyboard_interrupt',
            'final_state': node.current_state.value
        })
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()