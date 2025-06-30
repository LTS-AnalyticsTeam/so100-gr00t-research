from transitions import Machine
from enum import Enum


class State(Enum):
    NORMAL = "NORMAL"
    RECOVERY = "RECOVERY"
    VERIFICATION = "VERIFICATION"
    COMPLETION = "COMPLETION"


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


class SystemState:
    states = [State.NORMAL.value, State.RECOVERY.value, State.VERIFICATION.value]
    transitions = [
        {"trigger": ADR.NORMAL.value, "source": State.NORMAL.value, "dest": State.NORMAL.value},
        {"trigger": ADR.ANOMALY.value, "source": State.NORMAL.value, "dest": State.RECOVERY.value},
        {"trigger": RDR.UNRECOVERED.value, "source": State.RECOVERY.value, "dest": State.RECOVERY.value},
        {"trigger": RDR.RECOVERED.value, "source": State.RECOVERY.value, "dest": State.VERIFICATION.value},
        {"trigger": VDR.SOLVED.value, "source": State.VERIFICATION.value, "dest": State.NORMAL.value},
        {"trigger": VDR.UNSOLVED.value, "source": State.VERIFICATION.value, "dest": State.RECOVERY.value},
    ] # fmt: skip

    def __init__(self):
        self.machine = Machine(
            model=self,
            states=self.states,
            transitions=self.transitions,
            initial=State.NORMAL.value,
        )

    def current_cb(self) -> str:
        """Get the current callback name based on the current state."""
        return self.get_cb_name(self.state)

    def get_cb_name(self, state: str) -> str:
        """Get the callback name for a given state."""
        return f"CB_{state}"

    def write_mermaid(self):
        lines = [
            "```mermaid",
            "---",
            "config:",
            "  theme: redux",
            "  flowchart:",
            "    nodeSpacing: 300",
            "    rankSpacing: 60",
            "    curve: basis",
            "---",
        ]
        lines.append("flowchart TD")
        for state in self.states:
            lines.append(f"    {state}({state})")
            lines.append(f"    {state}({state}) --> {self.get_cb_name(state)}{{{self.get_cb_name(state)}}}") # fmt: skip
        for linkage in self.transitions:
            lines.append(
                f"    {self.get_cb_name(linkage['source'])}{{{self.get_cb_name(linkage['source'])}}} -- {linkage['trigger']} --> {linkage['dest']}"
            )
        lines.append("```")
        return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    ss = SystemState()
    print(ss.state)  # 現在の状態名が表示される
    getattr(ss, ADR.ANOMALY.value)()
    print(ss.state)  # 現在の状態名が表示される

    # mermaid形式で状態遷移図を出力
    print(ss.write_mermaid())
