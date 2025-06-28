
from enum import Enum
import base64

class CameraDevice(Enum):
    FRONT = "/dev/video0"
    RIGHT = "/dev/video2"

def get_image_from_camera(device: CameraDevice):

    return image

def transform_image2base64(image) -> base64:
    
    return base64_image
    