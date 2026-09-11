from setuptools import setup
from glob import glob

package_name = 'robomaster_ep_moveit_config'

setup(
    name=package_name,
    version='0.0.1',
    packages=[],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml') + glob('config/*.srdf')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='LpTriX',
    maintainer_email='LpTriX@users.noreply.github.com',
    description='MoveIt 2 configuration for the RoboMaster EP pick-and-place simulation',
    license='MIT',
)
