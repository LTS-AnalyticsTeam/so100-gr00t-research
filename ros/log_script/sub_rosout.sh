# bash ros/log_script/sub_rosout.sh <node_name>
# <node_name> is ["camera", "vlm_detector", "state_manager", "vla_controller"]

ros2 topic echo /rosout \
  --filter "m.name == '$1'" \
  --flow-style \
| grep -oP '(?<=msg: ).*'
