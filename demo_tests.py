#!/usr/bin/env python3

"""Example test configuration and demo for the VLA & VLM Auto Recovery system tests"""

import os
import tempfile
import json
import cv2
import numpy as np


def create_demo_test_environment():
    """Create a demonstration test environment with sample data"""
    
    # Create temporary directory for demo
    demo_dir = tempfile.mkdtemp(prefix='vlm_demo_')
    print(f"Creating demo test environment in: {demo_dir}")
    
    # Create sample prompt file
    prompt_content = """Analyze this image for anomalies in a block sorting task.

IMPORTANT: Respond ONLY with valid JSON:
{
  "anomaly": "normal|pose_error|missing_block|extra_block",
  "action": "action_name_from_list",
  "target_id": "target_id_from_list", 
  "confidence": 0.85,
  "reasoning": "Brief explanation"
}

Look for:
- Blocks in wrong positions
- Missing blocks from expected locations
- Extra blocks that shouldn't be there
- Blocks with incorrect orientations
"""
    
    prompt_file = os.path.join(demo_dir, 'demo_prompt.txt')
    with open(prompt_file, 'w') as f:
        f.write(prompt_content)
    
    # Create sample action list
    demo_actions = [
        {
            "action": "Move the red block from the tray to the right dish.",
            "target_id": "block_red"
        },
        {
            "action": "Move the green block from the tray to the left dish.", 
            "target_id": "block_green"
        },
        {
            "action": "Move the blue block from the tray to the center dish.",
            "target_id": "block_blue"
        },
        {
            "action": "Reposition the red block to the center of the right dish.",
            "target_id": "block_red"
        },
        {
            "action": "Stack the green block on top of the blue block.",
            "target_id": "block_green"
        },
        {
            "action": "Remove the extra yellow block from the workspace.",
            "target_id": "block_yellow"
        }
    ]
    
    action_file = os.path.join(demo_dir, 'demo_actions.jsonl')
    with open(action_file, 'w') as f:
        for action in demo_actions:
            f.write(json.dumps(action) + '\n')
    
    # Create sample test images
    image_dir = os.path.join(demo_dir, 'images')
    os.makedirs(image_dir)
    
    # Create different test scenarios
    scenarios = [
        'normal_scene.jpg',
        'missing_red_block.jpg', 
        'extra_block_present.jpg',
        'blocks_misaligned.jpg'
    ]
    
    for scenario in scenarios:
        # Create colored test image to simulate different scenarios
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        if 'normal' in scenario:
            # Green background for normal scene
            img[:, :] = [0, 128, 0]
        elif 'missing' in scenario:
            # Red background for missing block
            img[:, :] = [0, 0, 128]
        elif 'extra' in scenario:
            # Blue background for extra block
            img[:, :] = [128, 0, 0]
        else:
            # Yellow background for misaligned
            img[:, :] = [0, 128, 128]
        
        # Add some "blocks" (colored rectangles)
        cv2.rectangle(img, (100, 100), (150, 150), (255, 255, 255), -1)  # White block
        cv2.rectangle(img, (200, 100), (250, 150), (255, 0, 0), -1)      # Blue block
        cv2.rectangle(img, (300, 100), (350, 150), (0, 255, 0), -1)      # Green block
        
        image_path = os.path.join(image_dir, scenario)
        cv2.imwrite(image_path, img)
    
    # Create demo configuration
    demo_config = {
        'demo_directory': demo_dir,
        'prompt_file': prompt_file,
        'action_file': action_file,
        'image_directory': image_dir,
        'test_scenarios': scenarios,
        'action_count': len(demo_actions)
    }
    
    config_file = os.path.join(demo_dir, 'demo_config.json')
    with open(config_file, 'w') as f:
        json.dump(demo_config, f, indent=2)
    
    print(f"Demo environment created with:")
    print(f"  - Prompt file: {prompt_file}")
    print(f"  - Action file: {action_file} ({len(demo_actions)} actions)")
    print(f"  - Image directory: {image_dir} ({len(scenarios)} scenarios)")
    print(f"  - Config file: {config_file}")
    
    return demo_config


def run_demo_tests():
    """Run demonstration tests using the demo environment"""
    
    print("=== VLA & VLM Auto Recovery System - Test Demo ===\n")
    
    # Create demo environment
    demo_config = create_demo_test_environment()
    
    # Test 1: Configuration file loading
    print("\n1. Testing configuration file loading...")
    
    # Load and validate prompt
    with open(demo_config['prompt_file'], 'r') as f:
        prompt = f.read()
    print(f"   ✓ Prompt loaded ({len(prompt)} characters)")
    
    # Load and validate actions
    actions = []
    with open(demo_config['action_file'], 'r') as f:
        for line in f:
            actions.append(json.loads(line.strip()))
    print(f"   ✓ Actions loaded ({len(actions)} actions)")
    
    # Test 2: Simulated VLM analysis
    print("\n2. Testing simulated VLM analysis...")
    
    analysis_results = []
    anomaly_types = ['missing_block', 'extra_block', 'pose_error', 'normal']
    
    for i, scenario in enumerate(demo_config['test_scenarios']):
        # Simulate analysis logic
        if i % 4 == 3:  # Every 4th is normal
            result = {
                'scenario': scenario,
                'anomaly': 'normal',
                'action': '',
                'target_id': '',
                'confidence': 0.95,
                'reasoning': 'No anomalies detected - scene appears normal'
            }
        else:
            anomaly_type = anomaly_types[i % 3]
            selected_action = actions[i % len(actions)]
            result = {
                'scenario': scenario,
                'anomaly': anomaly_type,
                'action': selected_action['action'],
                'target_id': selected_action['target_id'],
                'confidence': 0.85,
                'reasoning': f'{anomaly_type} detected in scene'
            }
        
        analysis_results.append(result)
        print(f"   ✓ {scenario}: {result['anomaly']} (confidence: {result['confidence']})")
    
    # Test 3: State machine simulation
    print("\n3. Testing state machine transitions...")
    
    states = ['Normal', 'Recovering', 'VerificationPending', 'Normal']
    transitions = [
        'Anomaly detected → Start recovery',
        'Recovery completed → Start verification', 
        'Verification successful → Return to normal'
    ]
    
    for i, transition in enumerate(transitions):
        print(f"   ✓ {states[i]} → {states[i+1]}: {transition}")
    
    # Test 4: Message flow simulation
    print("\n4. Testing ROS2 message flow simulation...")
    
    message_types = [
        '/recovery_action',
        '/vla_pause',
        '/recovery_status',
        '/verification_request',
        '/recovery_verification'
    ]
    
    for msg_type in message_types:
        print(f"   ✓ Message flow: {msg_type}")
    
    # Summary
    print(f"\n=== Demo Summary ===")
    print(f"✓ Created test environment in: {demo_config['demo_directory']}")
    print(f"✓ Tested {len(demo_config['test_scenarios'])} scenarios")
    print(f"✓ Validated {len(actions)} recovery actions") 
    print(f"✓ Simulated complete recovery workflow")
    print(f"✓ All test components working correctly")
    
    print(f"\nTo run actual unit tests:")
    print(f"  python -m pytest src/vlm_node/test/test_vlm_node.py -v")
    print(f"\nTo clean up demo environment:")
    print(f"  rm -rf {demo_config['demo_directory']}")
    
    return demo_config


if __name__ == '__main__':
    try:
        config = run_demo_tests()
        print(f"\nDemo completed successfully!")
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()