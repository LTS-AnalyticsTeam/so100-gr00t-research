import pytest
import cv2
import numpy as np
from pathlib import Path
from ros.src.vla_auto_recover.vla_auto_recover.processing.vlm_detector import (
    VLMDetector,
    BaseDetector,
)
from vla_auto_recover.processing.config.system_settings import CB_InputIF
from ros.src.vla_auto_recover.vla_auto_recover.processing.config.system_settings import (
    ADR,
    RDR,
    VDR,
    State,
)


TEST_DATA_DIR = Path("/workspace/ros/src/vla_auto_recover/test/__test_data__/camera")


def test_init_vlm_detector():
    """Test initialization of VLMDetector"""
    vlm = VLMDetector()
    assert vlm.client is not None, "VLM client should be initialized"
    assert hasattr(vlm, "use_azure"), "VLMDetector should have use_azure attribute"


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


def test_StateManager_call_CB():
    """各状態でのコールバック関数の呼び出しをテスト"""
    detector = BaseDetector()
    input_data = CB_InputIF(images=[], action_id=None)
    print("=" * 20)

    state = State.RUNNING
    print(f"state: {state}")
    detector.call_CB(state, input_data)
    print("=" * 20)

    state = State.RECOVERY
    print(f"state: {state}")
    detector.call_CB(state, input_data)
    print("=" * 20)

    state = State.VERIFICATION
    print(f"state: {state}")
    detector.call_CB(state, input_data)
    print("=" * 20)

    state = State.END
    print(f"state: {state}")
    detector.call_CB(state, input_data)
    print("=" * 20)
