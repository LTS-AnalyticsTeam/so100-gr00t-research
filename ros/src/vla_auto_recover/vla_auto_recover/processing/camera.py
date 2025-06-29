from importlib.resources import path
import cv2
import numpy as np


class Camera:

    CENTER_CAM = "/dev/video0"
    RIGHT_CAM = "/dev/video2"
    RESOLUTION = (640, 480)

    def get_images_from_camera(self) -> tuple[np.ndarray, np.ndarray]:
        return center_image, right_image

    def get_images_from_path(
        self, center_path: str, right_path: str
    ) -> tuple[np.ndarray, np.ndarray]:
        center_image = cv2.imread(center_path)
        right_image = cv2.imread(right_path)
        return center_image, right_image
