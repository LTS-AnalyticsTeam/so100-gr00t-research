from enum import Enum

from dataclasses import dataclass
import numpy as np


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


CB_PREFIX = "CB_"


@dataclass
class CB_InputIF:
    images: list[np.ndarray] = None
    action_id: int = None


@dataclass
class CB_OutputIF:
    detection_result: ADR | RDR | VDR = None
    action_id: int = None
    reason: str = None


if __name__ == "__main__":
    # Example usage
    ss = SystemState()
    print(ss.state)  # 現在の状態名が表示される
    getattr(ss, ADR.ANOMALY.value)()
    print(ss.state)  # 現在の状態名が表示される

    # mermaid形式で状態遷移図を出力
    print(ss.write_mermaid())
