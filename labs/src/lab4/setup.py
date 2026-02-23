from setuptools import find_packages, setup

package_name = 'lab4'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/turtlebot_bringup.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mohammadnsr1',
    maintainer_email='nasrmohammad661@gmail.com',
    description='Lab4 reactive navigation nodes: subscribe to laser scan, publish cmd_vel, toggle via service.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'reactive_navigator = lab4.reactive_navigator:main',
            'reactive_navigator_v2 = lab4.reactive_navigator_v2:main',
        ],
    },
)
