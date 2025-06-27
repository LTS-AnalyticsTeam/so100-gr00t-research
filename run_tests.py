#!/usr/bin/env python3

"""Test runner script for the VLA & VLM Auto Recovery system"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(test_type="all", verbose=False, coverage=False):
    """Run tests based on specified type"""
    
    # Get the repository root directory
    repo_root = Path(__file__).parent
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    # Add coverage if requested
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
    
    # Determine test paths based on type
    if test_type == "unit":
        test_paths = [
            "src/vlm_node/test",
            "src/state_manager/test", 
            "src/gr00t_controller/test"
        ]
        cmd.extend(["-m", "unit"])
    elif test_type == "integration":
        test_paths = ["tests/integration"]
        cmd.extend(["-m", "integration"])
    elif test_type == "all":
        test_paths = [
            "src/vlm_node/test",
            "src/state_manager/test",
            "src/gr00t_controller/test",
            "tests/integration"
        ]
    else:
        print(f"Unknown test type: {test_type}")
        return 1
    
    # Add test paths to command
    for path in test_paths:
        full_path = repo_root / path
        if full_path.exists():
            cmd.append(str(full_path))
        else:
            print(f"Warning: Test path {full_path} does not exist")
    
    # Set working directory to repo root
    os.chdir(repo_root)
    
    # Add repo root to Python path
    env = os.environ.copy()
    python_paths = [
        str(repo_root),
        str(repo_root / 'src'),
        str(repo_root / 'src' / 'vlm_node'),
        str(repo_root / 'src' / 'state_manager'), 
        str(repo_root / 'src' / 'gr00t_controller')
    ]
    existing_path = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = ':'.join(python_paths) + (':' + existing_path if existing_path else '')
    
    print(f"Running tests with command: {' '.join(cmd)}")
    print(f"Working directory: {repo_root}")
    
    # Run the tests
    try:
        result = subprocess.run(cmd, env=env, check=False)
        return result.returncode
    except FileNotFoundError:
        print("Error: pytest not found. Please install pytest:")
        print("pip install pytest pytest-mock")
        return 1


def check_dependencies():
    """Check if required test dependencies are installed"""
    required_packages = ['pytest', 'pytest-mock']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required test dependencies:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall missing dependencies with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Run tests for VLA & VLM Auto Recovery system")
    parser.add_argument(
        "--type", "-t", 
        choices=["unit", "integration", "all"], 
        default="all",
        help="Type of tests to run (default: all)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Run tests with verbose output"
    )
    parser.add_argument(
        "--coverage", "-c",
        action="store_true", 
        help="Run tests with coverage reporting"
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check if test dependencies are installed"
    )
    
    args = parser.parse_args()
    
    if args.check_deps:
        if check_dependencies():
            print("All test dependencies are installed.")
            return 0
        else:
            return 1
    
    # Check dependencies before running tests
    if not check_dependencies():
        return 1
    
    # Run tests
    return run_tests(
        test_type=args.type,
        verbose=args.verbose,
        coverage=args.coverage
    )


if __name__ == "__main__":
    sys.exit(main())