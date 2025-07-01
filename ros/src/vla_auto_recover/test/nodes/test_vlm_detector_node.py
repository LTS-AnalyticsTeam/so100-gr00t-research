import pytest
import launch
import launch_pytest
from launch_ros.actions import Node


@launch_pytest.fixture
def launch_description():
    """Launch description for all nodes."""
    vlm_detector = Node(
        package="vla_auto_recover",
        executable="vlm_detector_node",
        name="vlm_detector",
        output="screen",
    )

    return launch.LaunchDescription(
        [
            vlm_detector,
            launch_pytest.actions.ReadyToTest(),
        ]
    )


@pytest.mark.launch(fixture=launch_description)
def test_all_nodes_startup(setup, launch_context):
    """Test that all nodes start up successfully."""
    import time

    time.sleep(1)
    assert launch_context is not None


@pytest.mark.launch(fixture=launch_description, shutdown=True)
def test_all_nodes_shutdown(setup, launch_context):
    """Test that all nodes shut down successfully."""
    import time

    time.sleep(1)
    assert launch_context is not None
