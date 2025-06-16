from setuptools import setup
import os
from glob import glob

package_name = 'vlm_node'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # ROS2が期待する場所に実行ファイルを配置
        ('lib/' + package_name, ['scripts/vlm_node']),
    ],
    install_requires=[
        'setuptools',
        'python-dotenv',
    ],
    zip_safe=True,
    maintainer='aoi kadoya',
    maintainer_email='aoi.kadoya@lt-s.jp',
    description='VLM node using lerobot for anomaly detection',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vlm_node = vlm_node.vlm_node:main',
        ],
    },
)
