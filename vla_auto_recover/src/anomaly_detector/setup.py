from setuptools import setup

package_name = 'anomaly_detector'

setup(
    name=package_name,
    version='0.1.0',
    packages=['anomaly_detector'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('lib/' + package_name, ['scripts/anomaly_detector_node']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aoi.kadoya',
    maintainer_email='aoi.kadoya@lt-s.jp',
    description='Anomaly detector package',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'anomaly_detector_node = anomaly_detector.anomaly_detector_node:main',
        ],
    },
)
