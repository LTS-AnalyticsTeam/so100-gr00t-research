from .config.system_settings import ADR, RDR, VDR, State, CB_PREFIX
from transitions import Machine
from transitions.extensions import GraphMachine


class StateManager:

    STATES = [
        State.RUNNING.value,
        State.RECOVERY.value,
        State.VERIFICATION.value,
        State.END.value,
    ]

    TRANSITIONS = [
        {"trigger": ADR.NORMAL.value, "source": State.RUNNING.value, "dest": State.RUNNING.value},
        {"trigger": ADR.ANOMALY.value, "source": State.RUNNING.value, "dest": State.RECOVERY.value},
        {"trigger": ADR.COMPLETION.value, "source": State.RUNNING.value, "dest": State.END.value},
        {"trigger": RDR.UNRECOVERED.value, "source": State.RECOVERY.value, "dest": State.RECOVERY.value},
        {"trigger": RDR.RECOVERED.value, "source": State.RECOVERY.value, "dest": State.VERIFICATION.value},
        {"trigger": VDR.SOLVED.value, "source": State.VERIFICATION.value, "dest": State.RUNNING.value},
        {"trigger": VDR.UNSOLVED.value, "source": State.VERIFICATION.value, "dest": State.RECOVERY.value},
    ] # fmt: skip

    def __init__(self, debug=False):

        self.machine = Machine(
            model=self,
            states=self.STATES,
            transitions=self.TRANSITIONS,
            initial=State.RUNNING.value,
        )

    def transition(self, detection_result: ADR | RDR | VDR) -> None:
        """TRANSITIONSのシナリオに基づいて状態遷移を実行"""
        # 状態遷移の実行
        prev_state = self.state
        trigger_name = detection_result.value
        transition_func = getattr(self, trigger_name)
        transition_func()

        if self.state == prev_state:
            # 状態が変わらなかった場合は、False
            return False
        else:
            # 状態が変わった場合は、True
            return True

    @classmethod
    def print_mermaid(cls):
        self = cls()
        machine = GraphMachine(
            model=self,
            states=cls.STATES,
            transitions=cls.TRANSITIONS,
            initial="IDLE",
            graph_engine="mermaid",  # ← ここがポイント
            show_conditions=True,  # 任意: ガードをエッジに表示
        )
        mermaid_src = self.get_graph().source
        print(mermaid_src)
        return None

    @classmethod
    def pprint_mermaid(cls):
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
        for state in cls.STATES:
            lines.append(f"    {state}({state})")
            lines.append(f"    {state}({state}) --> {CB_PREFIX}{state}{{{CB_PREFIX}{state}}}") # fmt: skip
        for linkage in cls.TRANSITIONS:
            lines.append(
                f"    {CB_PREFIX}{linkage['source']}{{{CB_PREFIX}{linkage['source']}}} -- {linkage['trigger']} --> {linkage['dest']}({linkage['dest']})"
            )
        lines.append("```")
        return "\n".join(lines)
