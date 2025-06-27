from setuptools import setup

package_name = 'state_manager'

setup(
    name=package_name,
    version='0.1.0',
    packages=['state_manager'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('lib/' + package_name, ['scripts/state_manager_node']), 
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aoi.kadoya',
    maintainer_email='aoi.kadoya@lt-s.jp',
    description='Simple two-state manager for hybrid control',
    license='Apache-2.0',
    tests_require=['pytest', 'pytest-mock'],
    extras_require={
        'test': ['pytest', 'pytest-mock']
    },
    entry_points={
        'console_scripts': [
            'state_manager_node = state_manager.state_manager_node:main',
        ],
    },
)
