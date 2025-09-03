import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pathlib import Path

TEST_DATA_DIR = Path("/workspace/ros/src/vla_auto_recover/test/__test_data__/camera/")


def get_sample_image(
    state: str = "normal", phase: str = "start", aspect: str = "center"
) -> Image:
    bridge = CvBridge()
    image = bridge.cv2_to_imgmsg(
        cv2.imread(TEST_DATA_DIR.joinpath(state, phase, aspect + ".png")),
        encoding="bgr8",
    )
    return image
