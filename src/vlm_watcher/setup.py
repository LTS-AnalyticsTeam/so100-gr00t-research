from setuptools import setup

package_name = 'vlm_watcher'

setup(
    name=package_name,
    version='0.1.0',
    packages=['vlm_watcher'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('lib/' + package_name, ['scripts/vlm_watcher_node']), 
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aoi.kadoya',
    maintainer_email='aoi.kadoya@lt-s.jp',
    description='VLM watcher package',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'vlm_watcher_node = vlm_watcher.vlm_watcher_node:main',
        ],
    },
)
