# Test Documentation for VLA & VLM Auto Recovery System

## Overview

This document describes the comprehensive test suite for the VLA & VLM Auto Recovery system. The test suite covers unit tests, integration tests, and system-level tests for all major components.

## Test Structure

```
tests/
├── __init__.py                    # Main test package
├── test_utils.py                  # Test utilities and fixtures
└── integration/
    ├── __init__.py
    └── test_system_integration.py # System integration tests

src/
├── vlm_node/test/
│   ├── __init__.py
│   └── test_vlm_node.py          # VLM node unit tests
├── state_manager/test/
│   ├── __init__.py
│   └── test_state_manager.py     # State manager unit tests
└── gr00t_controller/test/
    ├── __init__.py
    └── test_gr00t_controller.py  # GR00T controller unit tests
```

## Running Tests

### Prerequisites

Install test dependencies:
```bash
pip install pytest pytest-mock opencv-python numpy
```

### Run All Tests
```bash
python run_tests.py
```

### Run Specific Test Types
```bash
# Unit tests only
python run_tests.py --type unit

# Integration tests only  
python run_tests.py --type integration

# With verbose output
python run_tests.py --verbose

# With coverage reporting
python run_tests.py --coverage
```

### Run Individual Test Files
```bash
# VLM Node tests
python -m pytest src/vlm_node/test/test_vlm_node.py -v

# State Manager tests
python -m pytest src/state_manager/test/test_state_manager.py -v

# GR00T Controller tests
python -m pytest src/gr00t_controller/test/test_gr00t_controller.py -v

# Integration tests
python -m pytest tests/integration/test_system_integration.py -v
```

## Test Coverage

The test suite covers the following components and functionality:

### VLMNode Tests (`src/vlm_node/test/test_vlm_node.py`)
- ✅ VLMNode initialization and configuration
- ✅ Prompt file loading and validation
- ✅ Action list loading and validation
- ✅ Simulated VLM analysis functionality
- ✅ Recovery action and target selection
- ✅ Azure OpenAI integration (mocked)
- ✅ Image processing and analysis workflows

### StateManager Tests (`src/state_manager/test/test_state_manager.py`)
- ✅ StateManager initialization and configuration
- ✅ System state enumeration and transitions
- ✅ State transition from Normal to Recovering
- ✅ Anomaly detection callback handling
- ✅ Recovery status callback processing
- ✅ Verification result callback (success/failure)
- ✅ Verification timeout handling
- ✅ Full recovery workflow simulation

### GR00TController Tests (`src/gr00t_controller/test/test_gr00t_controller.py`)
- ✅ GR00TController initialization and configuration
- ✅ VLA pause/resume callback handling
- ✅ Recovery action callback processing
- ✅ VLA execution when paused (should skip)
- ✅ Recovery action execution workflow
- ✅ Robot interface mocking and error handling
- ✅ Status reporting and logging
- ✅ Full recovery workflow simulation

### Integration Tests (`tests/integration/test_system_integration.py`)
- ✅ End-to-end recovery workflow simulation
- ✅ ROS2 message flow between components
- ✅ Error handling scenarios
- ✅ Configuration file validation
- ✅ Environment variable configuration
- ✅ System behavior under error conditions

## Test Methodology

### Mocking Strategy
The tests use extensive mocking to isolate components and avoid dependencies:

- **ROS2 Dependencies**: All ROS2 components (rclpy, Node, messages) are mocked
- **External APIs**: Azure OpenAI API calls are mocked with realistic responses
- **Robot Hardware**: SO100Robot and related hardware interfaces are mocked
- **File System**: Temporary directories and files are used for testing

### Test Data
Tests use realistic test data including:
- Sample action lists with robot movement commands
- Test prompt templates for VLM analysis
- Mock image data for vision processing
- Simulated sensor data and robot states

### Error Simulation
Tests include scenarios for:
- Network connectivity issues
- API rate limiting and failures
- Robot hardware connection problems
- Configuration file errors
- Invalid input data

## Test Markers

The test suite uses pytest markers for categorization:

- `@pytest.mark.unit`: Unit tests for individual components
- `@pytest.mark.integration`: Integration tests across components
- `@pytest.mark.slow`: Tests that take longer to run
- `@pytest.mark.requires_hardware`: Tests requiring actual robot hardware
- `@pytest.mark.requires_api`: Tests requiring external API access

## Configuration

Test configuration is managed through:
- `pytest.ini`: Pytest configuration and test discovery
- `tests/test_utils.py`: Common test utilities and fixtures
- Environment variables for external service configuration

## Continuous Integration

The test suite is designed to run in CI/CD environments:
- No external dependencies required for core tests
- Comprehensive mocking prevents flaky tests
- Configurable test execution based on available resources
- Coverage reporting integration

## Adding New Tests

When adding new functionality:

1. **Unit Tests**: Add tests to the appropriate component test file
2. **Integration Tests**: Add system-level tests to `tests/integration/`
3. **Test Utilities**: Add reusable fixtures to `tests/test_utils.py` 
4. **Documentation**: Update this file with test coverage information

### Test Template
```python
import unittest
from unittest.mock import Mock, patch
from tests.test_utils import TestFixtures, ROSMockHelper

class TestNewComponent(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_data = TestFixtures.setup_test_files(self.temp_dir)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_new_functionality(self):
        # Test implementation
        pass
```

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all test dependencies are installed
2. **Mock Failures**: Check that mocks are applied before imports
3. **File Path Issues**: Use absolute paths in tests
4. **ROS2 Dependencies**: Ensure ROS2 components are properly mocked

### Debug Mode
Run tests with additional debugging:
```bash
python -m pytest --pdb --pdbcls=IPython.terminal.debugger:Pdb -v
```