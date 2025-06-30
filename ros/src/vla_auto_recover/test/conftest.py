import os
import subprocess
from pathlib import Path
import pytest


@pytest.fixture(scope="session", autouse=True)
def setup():
    """ROS環境を自動的にセットアップ"""
    workspace_root = Path("/workspace/ros")

    print(f"Setting up ROS environment from {workspace_root}")

    # ワークスペースディレクトリに移動
    os.chdir(workspace_root)

    # パッケージをビルド（必要な場合）
    print("Building vla_auto_recover package...")
    build_result = subprocess.run(
        ["colcon", "build"],
        cwd=workspace_root,
        capture_output=True,
        text=True,
    )

    if build_result.returncode != 0:
        print(f"Build warning/error: {build_result.stderr}")
        # エラーでも続行（既存のビルドがある可能性）

    # setup.bashから環境変数を取得して設定
    setup_bash = workspace_root / "install" / "setup.bash"
    if setup_bash.exists():
        print("Sourcing ROS environment...")
        env_result = subprocess.run(
            ["bash", "-c", f"source {setup_bash} && env"],
            capture_output=True,
            text=True,
            cwd=workspace_root,
        )

        if env_result.returncode == 0:
            # 環境変数を現在のプロセスに設定
            for line in env_result.stdout.split("\n"):
                if "=" in line and any(
                    var in line
                    for var in [
                        "ROS_",
                        "AMENT_",
                        "PYTHONPATH",
                        "LD_LIBRARY_PATH",
                        "CMAKE_PREFIX_PATH",
                    ]
                ):
                    try:
                        key, value = line.split("=", 1)
                        os.environ[key] = value
                        print(f"Set {key}={value[:50]}...")  # 最初の50文字のみ表示
                    except ValueError:
                        continue
            print("ROS environment variables set successfully")
        else:
            print(f"Failed to source environment: {env_result.stderr}")
    else:
        print("Warning: setup.bash not found")

    yield
