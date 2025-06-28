import vla_interfaces
from enum import Enum

class State(Enum):
    NORMAL = vla_interfaces.msg.State.NORMAL
    ANOMALY = vla_interfaces.msg.State.ANOMALY
    RECOVERING = vla_interfaces.msg.State.RECOVERING