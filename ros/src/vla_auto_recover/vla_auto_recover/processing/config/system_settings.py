from enum import Enum

from dataclasses import dataclass
import numpy as np

# from cv_bridge import CvBridge
from vla_interfaces.msg._image_pair import ImagePair
from vla_auto_recover.processing.utils.image_convert import imgmsg_to_ndarray


class State(Enum):
    RUNNING = "RUNNING"
    RECOVERY = "RECOVERY"
    VERIFICATION = "VERIFICATION"
    END = "END"


class ADR(Enum):
    """Anomaly Detection Result - 異常検出の結果"""

    NORMAL = "NORMAL"
    ANOMALY = "ANOMALY"
    COMPLETION = "COMPLETION"


class RDR(Enum):
    """Recovery Detection Result - 回復検出の結果"""

    RECOVERED = "RECOVERED"
    UNRECOVERED = "UNRECOVERED"


class VDR(Enum):
    """Verification Detection Result - 検証検出の結果"""

    SOLVED = "SOLVED"
    UNSOLVED = "UNSOLVED"


def get_DR(detection_result_str: str) -> ADR | RDR | VDR:
    """Detection Resultを取得する"""
    if detection_result_str in ADR.__members__:
        return ADR[detection_result_str]
    elif detection_result_str in RDR.__members__:
        return RDR[detection_result_str]
    elif detection_result_str in VDR.__members__:
        return VDR[detection_result_str]
    else:
        raise ValueError(f"Invalid detection result: {detection_result_str}")


CB_PREFIX = "CB_"


@dataclass
class CB_InputIF:
    images: list[np.ndarray] = None
    action_id: int = None

    @classmethod
    def from_msg(cls, msg: ImagePair, action_id: int):
        # bridge = CvBridge()

        center_cam: np.ndarray = imgmsg_to_ndarray(msg.center_cam)
        right_cam: np.ndarray = imgmsg_to_ndarray(msg.right_cam)

        return cls(
            images=[center_cam, right_cam],
            action_id=action_id,
        )


@dataclass
class CB_OutputIF:
    detection_result: ADR | RDR | VDR = None
    action_id: int = None
    reason: str = None
