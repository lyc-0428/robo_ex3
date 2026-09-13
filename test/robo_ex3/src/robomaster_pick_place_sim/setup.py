from setuptools import setup, find_packages
from glob import glob

package_name = "robomaster_pick_place_sim"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/urdf", glob("urdf/*")),
        ("share/" + package_name + "/meshes", glob("meshes/*")),
        ("share/" + package_name + "/worlds", glob("worlds/*")),
        ("share/" + package_name + "/config", glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="nvidia",
    maintainer_email="nvidia@localhost",
    description="RoboMaster EP repaired static Gazebo snapshot",
    license="MIT",
    entry_points={
        "console_scripts": [
            "pick_place_trajectory = robomaster_pick_place_sim.pick_place_trajectory:main",
            "grasp_place_demo = robomaster_pick_place_sim.pick_place_trajectory:main",
            "continuous_grasp = robomaster_pick_place_sim.continuous_grasp:main",
        ],
    },
)
