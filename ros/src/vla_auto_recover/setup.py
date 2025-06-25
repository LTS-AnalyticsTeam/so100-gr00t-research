from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'vla_auto_recover'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='shota.inoue@lt-s.jp',
    description='VLAが異常を自律的に回避するように設計されたアーキテクチャ',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vlm_watcher_node = vla_auto_recover.vlm_watcher_node:main',
            'gr00t_controller_node = vla_auto_recover.gr00t_controller_node:main',
            'state_manager_node = vla_auto_recover.state_manager_node:main',
        ],
    },
)
