import pytest
import cv2
import numpy as np
from pathlib import Path
from vla_auto_recover.processing.vlm_monitor import VLMMonitor

TEST_DATA_DIR = Path("/workspace/ros/src/vla_auto_recover/test/__test_data__/")
TEST_DATA = {
    "normal": {
            "center_cam": TEST_DATA_DIR.joinpath("camera", "normal", "center_cam.png"),
            "right_cam": TEST_DATA_DIR.joinpath("camera", "normal", "right_cam.png")
    },
    "anomaly-mispalace": {
            "center_cam": TEST_DATA_DIR.joinpath("camera", "anomaly-mispalace", "center_cam.png"),
            "right_cam": TEST_DATA_DIR.joinpath("camera", "anomaly-mispalace", "right_cam.png")
    },
    "anomaly-stacked_dish": {
            "center_cam": TEST_DATA_DIR.joinpath("camera", "anomaly-stacked_dish", "center_cam.png"),
            "right_cam": TEST_DATA_DIR.joinpath("camera", "anomaly-stacked_dish", "right_cam.png")
    },
} # fmt: skip


def test_init_vlm_monitor():
    """Test initialization of VLMMonitor"""
    vlm = VLMMonitor()
    assert vlm.client is not None, "VLM client should be initialized"
    assert hasattr(vlm, "use_azure"), "VLMMonitor should have use_azure attribute"


@pytest.mark.parametrize("pattern, state_answer", [
    ("normal", "NORMAL"),
    ("anomaly-mispalace", "ANOMALY"),
    ("anomaly-stacked_dish", "ANOMALY")
]) # fmt: skip
def test_detect_anomaly(pattern, state_answer):
    """Test anomaly detection with VLM"""
    vlm = VLMMonitor()
    images = [
        cv2.imread(str(TEST_DATA[pattern]["center_cam"])),
        cv2.imread(str(TEST_DATA[pattern]["right_cam"])),
    ]
    state, action_id, reason = vlm.detect_anomaly(
        images=images,
        language_instruction="move blocks from tray to matching dishes.",
    )
    print(f"Detected state: {state}, Action ID: {action_id}, Reason: {reason}")
    assert state.value == state_answer, f"Expected state {pattern}, got {state}"


@pytest.mark.parametrize("pattern, state_answer", [
    ("anomaly-mispalace", "ANOMALY", "move blocks from tray to matching dishes."),
    ("anomaly-stacked_dish", "ANOMALY", "move blocks from tray to matching dishes.")
]) # fmt: skip
def test_check_recovery_state(pattern, state_answer, language_instruction):
    """Test recovery state check"""
    vlm = VLMMonitor()
    images = [
        cv2.imread(str(TEST_DATA[pattern]["center_cam"])),
        cv2.imread(str(TEST_DATA[pattern]["right_cam"])),
    ]
    state, action_id, reason = vlm.detect_anomaly(
        images=images,
        language_instruction="move blocks from tray to matching dishes.",
    )
    assert state.value == "NORMAL", "Expected NORMAL state after recovery check"
    assert action_id == 0, "Expected action ID 0 for normal state"
    assert reason == "No anomalies detected", "Expected no anomalies reason"
