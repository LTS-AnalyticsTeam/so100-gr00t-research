import pytest
import cv2
import numpy as np
from pathlib import Path
from vla_auto_recover.processing.vlm_monitor import VLMDetector, CB_InputIF, VLMMonitor
from vla_auto_recover.processing.config.system_state import ADR, RDR, VDR, State

TEST_DATA_DIR = Path("/workspace/ros/src/vla_auto_recover/test/__test_data__/camera")


def test_init_vlm_monitor():
    """Test initialization of VLMMonitor"""
    vlm = VLMDetector()
    assert vlm.client is not None, "VLM client should be initialized"
    assert hasattr(vlm, "use_azure"), "VLMMonitor should have use_azure attribute"


@pytest.mark.parametrize("pattern, phase, result_answer", [
    ("normal", "start", ADR.NORMAL.value),
    ("normal", "end", ADR.COMPLETION.value),
    ("anomaly-mispalace", "start", ADR.ANOMALY.value),
    ("anomaly-stacked_dish", "start", ADR.ANOMALY.value)
]) # fmt: skip
def test_CB_RUNNING(pattern, phase, result_answer):
    """Test anomaly detection with VLM"""
    vlm = VLMDetector()
    images = [
        cv2.imread(str(TEST_DATA_DIR.joinpath(pattern, phase, "center_cam.png"))),
        cv2.imread(str(TEST_DATA_DIR.joinpath(pattern, phase, "right_cam.png"))),
    ]
    input_data = CB_InputIF(images=images, action_id=None)
    output = vlm.CB_RUNNING(input_data=input_data)
    print(f"Detected state: {output.detection_result}, Action ID: {output.action_id}, Reason: {output.reason}") # fmt: skip
    assert output.detection_result.value == result_answer, f"Expected state {pattern}, got {output.detection_result}" # fmt: skip


@pytest.mark.parametrize("pattern, phase, result_answer, action_id", [
    ("anomaly-mispalace", "start", RDR.UNRECOVERED.value, 1),
    ("anomaly-mispalace", "end", RDR.RECOVERED.value, 1),
    ("anomaly-stacked_dish", "start", RDR.UNRECOVERED.value, 2),
    ("anomaly-stacked_dish", "end", RDR.RECOVERED.value, 2),
]) # fmt: skip
def test_CB_RECOVERY(pattern, phase, result_answer, action_id):
    """Test recovery state check"""
    vlm = VLMDetector()
    images = [
        cv2.imread(str(TEST_DATA_DIR.joinpath(pattern, phase, "center_cam.png"))),
        cv2.imread(str(TEST_DATA_DIR.joinpath(pattern, phase, "right_cam.png"))),
    ]
    input_data = CB_InputIF(images=images, action_id=action_id)
    output = vlm.CB_RECOVERY(input_data=input_data)
    print(f"Recovery detection: {output.detection_result}, Reason: {output.reason}")
    assert (
        output.detection_result.value == result_answer
    ), f"Expected recovery result {result_answer}, got {output.detection_result}"


@pytest.mark.parametrize("pattern, phase, result_answer, action_id", [
    ("normal", "start", VDR.SOLVED.value, 1),
    ("normal", "end", VDR.SOLVED.value, 1),
    ("normal", "start", VDR.SOLVED.value, 2),
    ("normal", "end", VDR.SOLVED.value, 2),
    ("anomaly-mispalace", "start", VDR.UNSOLVED.value, 1),
    ("anomaly-mispalace", "end", VDR.SOLVED.value, 1),
    ("anomaly-stacked_dish", "start", VDR.UNSOLVED.value, 2),
    ("anomaly-stacked_dish", "end", VDR.SOLVED.value, 2),
]) # fmt: skip
def test_CB_VERIFICATION(pattern, phase, result_answer, action_id):
    """Test verification state check"""
    vlm = VLMDetector()
    images = [
        cv2.imread(str(TEST_DATA_DIR.joinpath(pattern, phase, "center_cam.png"))),
        cv2.imread(str(TEST_DATA_DIR.joinpath(pattern, phase, "right_cam.png"))),
    ]
    input_data = CB_InputIF(images=images, action_id=action_id)
    output = vlm.CB_VERIFICATION(input_data=input_data)
    print(
        f"Verification detection: {output.detection_result}, Action ID: {output.action_id}, Reason: {output.reason}"
    )
    assert (
        output.detection_result.value == result_answer
    ), f"Expected verification result {result_answer}, got {output.detection_result}"


def test_show_state_transition_diagram():
    # mermaid形式で状態遷移図を出力
    print(VLMMonitor.pprint_mermaid())


def test_VLMMonitor_state_transition():
    """Test state transitions in VLMMonitor
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
    vlm_monitor = VLMMonitor()
    # 初期状態はRUNNINGであることを確認
    assert (
        vlm_monitor.mdl.state == State.RUNNING.value
    ), "Initial state should be RUNNING"
    step = 0
    for i in range(10):
        # 10回ADR.NORMALを呼び出しても状態は変わらない
        vlm_monitor.transition(ADR.NORMAL)
        print(f"{step}. state: {vlm_monitor.mdl.state}, CB: CB_{vlm_monitor.mdl.state}")
        step += 1
        assert vlm_monitor.mdl.state == State.RUNNING.value
    else:
        # 最終的にADR.ANOMALYの結果を得たと仮定し、RECOVERY状態に遷移する。
        vlm_monitor.transition(ADR.ANOMALY)
        print(f"{step}. state: {vlm_monitor.mdl.state}, CB: CB_{vlm_monitor.mdl.state}")
        step += 1
        assert vlm_monitor.mdl.state == State.RECOVERY.value

    for i in range(10):
        # 10回RDR.UNRECOVEREDを呼び出しても状態は変わらない
        vlm_monitor.transition(RDR.UNRECOVERED)
        print(f"{step}. state: {vlm_monitor.mdl.state}, CB: CB_{vlm_monitor.mdl.state}")
        step += 1
        assert vlm_monitor.mdl.state == State.RECOVERY.value
    else:
        # 最終的にRDR.RECOVEREDの結果を得たと仮定し、VERIFICATION状態に遷移する。
        vlm_monitor.transition(RDR.RECOVERED)
        print(f"{step}. state: {vlm_monitor.mdl.state}, CB: CB_{vlm_monitor.mdl.state}")
        step += 1
        assert vlm_monitor.mdl.state == State.VERIFICATION.value

    # UNSOLVEDとなり、未解決な問題があるため、再度RECOVERY状態に遷移する。

    vlm_monitor.transition(VDR.UNSOLVED)
    print(f"{step}. state: {vlm_monitor.mdl.state}, CB: CB_{vlm_monitor.mdl.state}")
    step += 1
    assert vlm_monitor.mdl.state == State.RECOVERY.value

    for i in range(10):
        # 10回RDR.UNRECOVEREDを呼び出しても状態は変わらない
        vlm_monitor.transition(RDR.UNRECOVERED)
        print(f"{step}. state: {vlm_monitor.mdl.state}, CB: CB_{vlm_monitor.mdl.state}")
        step += 1
        assert vlm_monitor.mdl.state == State.RECOVERY.value
    else:
        # 最終的にRDR.RECOVEREDの結果を得たと仮定し、VERIFICATION状態に遷移する。
        vlm_monitor.transition(RDR.RECOVERED)
        print(f"{step}. state: {vlm_monitor.mdl.state}, CB: CB_{vlm_monitor.mdl.state}")
        step += 1
        assert vlm_monitor.mdl.state == State.VERIFICATION.value

    # SOLVEDとなり、問題が解決されたため、RUNNING状態に遷移する。
    vlm_monitor.transition(VDR.SOLVED)
    print(f"{step}. state: {vlm_monitor.mdl.state}, CB: CB_{vlm_monitor.mdl.state}")
    step += 1
    assert vlm_monitor.mdl.state == State.RUNNING.value

    # 最終的にEND状態に遷移する。
    vlm_monitor.transition(ADR.COMPLETION)
    print(f"{step}. state: {vlm_monitor.mdl.state}, CB: CB_{vlm_monitor.mdl.state}")
    step += 1
    assert vlm_monitor.mdl.state == State.END.value


def test_VLMMonitor_call_CB():
    """各状態でのコールバック関数の呼び出しをテスト"""
    vlm_monitor = VLMMonitor(debug=True)
    input_data = CB_InputIF(images=[], action_id=None)
    print("=" * 20)

    vlm_monitor.mdl.state = State.RUNNING.value
    print(f"state: {vlm_monitor.mdl.state}")
    vlm_monitor.call_CB(input_data)
    print("=" * 20)

    vlm_monitor.mdl.state = State.RECOVERY.value
    print(f"state: {vlm_monitor.mdl.state}")
    vlm_monitor.call_CB(input_data)
    print("=" * 20)

    vlm_monitor.mdl.state = State.VERIFICATION.value
    print(f"state: {vlm_monitor.mdl.state}")
    vlm_monitor.call_CB(input_data)
    print("=" * 20)

    vlm_monitor.mdl.state = State.END.value
    print(f"state: {vlm_monitor.mdl.state}")
    vlm_monitor.call_CB(input_data)
    print("=" * 20)
