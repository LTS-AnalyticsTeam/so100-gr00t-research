from setuptools import setup

package_name = 'gr00t_controller'

setup(
    name=package_name,
    version='0.1.0',
    packages=['gr00t_controller'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('lib/' + package_name, ['scripts/gr00t_controller_node']), 
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aoi.kadoya',
    maintainer_email='aoi.kadoya@lt-s.jp',
    description='Gr00t controller package',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'gr00t_controller_node = gr00t_controller.gr00t_controller_node:main',
        ],
    },
)
