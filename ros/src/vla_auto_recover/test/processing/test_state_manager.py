from vla_auto_recover.processing.state_manager import StateManager
from vla_auto_recover.processing.config.system_settings import ADR, RDR, VDR, State
from vla_auto_recover.processing.config.system_settings import CB_InputIF


def test_show_state_transition_diagram():
    # mermaid形式で状態遷移図を出力
    print(StateManager.pprint_mermaid())


def test_StateManager_state_transition():
    """Test state transitions in StateManager
    次のシナリオを検証する。
    1. 初期状態はNORMALであることを確認
    2. 10回ADR.NORMALを呼び出しても状態は変わらない
    3. 最終的にADR.ANOMALYの結果を得たと仮定し、RECOVERY状態に遷移する。
    4. 10回RDR.UNRECOVEREDを呼び出しても状態は変わらない
    5. 最終的にRDR.RECOVEREDの結果を得たと仮定し、VERIFICATION状態に遷移する。
    6. UNSOLVEDとなり、未解決な問題があるため、再度RECOVERY状態に遷移する。
    7. 10回RDR.UNRECOVEREDを呼び出しても状態は変わらない
    8. 最終的にRDR.RECOVEREDの結果を得たと仮定し、VERIFICATION状態に遷移する。
    9. SOLVEDとなり、問題が解決されたため、NORMAL状態に遷移する。
    """
    state_manager = StateManager()
    # 初期状態はRUNNINGであることを確認
    assert state_manager.state == State.RUNNING.value, "Initial state should be RUNNING"
    step = 0
    for i in range(10):
        # 10回ADR.NORMALを呼び出しても状態は変わらない
        state_manager.transition(ADR.NORMAL)
        print(f"{step}. state: {state_manager.state}, CB: CB_{state_manager.state}")
        step += 1
        assert state_manager.state == State.RUNNING.value
    else:
        # 最終的にADR.ANOMALYの結果を得たと仮定し、RECOVERY状態に遷移する。
        state_manager.transition(ADR.ANOMALY)
        print(f"{step}. state: {state_manager.state}, CB: CB_{state_manager.state}")
        step += 1
        assert state_manager.state == State.RECOVERY.value

    for i in range(10):
        # 10回RDR.UNRECOVEREDを呼び出しても状態は変わらない
        state_manager.transition(RDR.UNRECOVERED)
        print(f"{step}. state: {state_manager.state}, CB: CB_{state_manager.state}")
        step += 1
        assert state_manager.state == State.RECOVERY.value
    else:
        # 最終的にRDR.RECOVEREDの結果を得たと仮定し、VERIFICATION状態に遷移する。
        state_manager.transition(RDR.RECOVERED)
        print(f"{step}. state: {state_manager.state}, CB: CB_{state_manager.state}")
        step += 1
        assert state_manager.state == State.VERIFICATION.value

    # UNSOLVEDとなり、未解決な問題があるため、再度RECOVERY状態に遷移する。

    state_manager.transition(VDR.UNSOLVED)
    print(f"{step}. state: {state_manager.state}, CB: CB_{state_manager.state}")
    step += 1
    assert state_manager.state == State.RECOVERY.value

    for i in range(10):
        # 10回RDR.UNRECOVEREDを呼び出しても状態は変わらない
        state_manager.transition(RDR.UNRECOVERED)
        print(f"{step}. state: {state_manager.state}, CB: CB_{state_manager.state}")
        step += 1
        assert state_manager.state == State.RECOVERY.value
    else:
        # 最終的にRDR.RECOVEREDの結果を得たと仮定し、VERIFICATION状態に遷移する。
        state_manager.transition(RDR.RECOVERED)
        print(f"{step}. state: {state_manager.state}, CB: CB_{state_manager.state}")
        step += 1
        assert state_manager.state == State.VERIFICATION.value

    # SOLVEDとなり、問題が解決されたため、RUNNING状態に遷移する。
    state_manager.transition(VDR.SOLVED)
    print(f"{step}. state: {state_manager.state}, CB: CB_{state_manager.state}")
    step += 1
    assert state_manager.state == State.RUNNING.value

    # 最終的にEND状態に遷移する。
    state_manager.transition(ADR.COMPLETION)
    print(f"{step}. state: {state_manager.state}, CB: CB_{state_manager.state}")
    step += 1
    assert state_manager.state == State.END.value
