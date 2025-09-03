# bash ros/log_script/latest_logs.sh <node_name>
#!/bin/bash

if [ "$1" == "camera" ]; then
    cat /root/.ros/log/latest/camera_node-1.log
elif [ "$1" == "vlm_detector" ]; then
    cat /root/.ros/log/latest/vlm_detector_node-2.log
elif [ "$1" == "state_manager" ]; then
    cat /root/.ros/log/latest/state_manager_node-3.log
elif [ "$1" == "vla_controller" ]; then
    cat /root/.ros/log/latest/vla_controller_node-4.log
else
    echo "Invalid argument. Please specify one of: camera, vlm_detector, state_manager, vla_controller."
    exit 1
fi
