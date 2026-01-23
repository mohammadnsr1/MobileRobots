from setuptools import find_packages, setup

package_name = 'lab1'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mohammadnsr1',
    maintainer_email='nasrmohammad661@gmail.com',
    description='Lab1 package for practicing ROS2',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'lab1_publisher = lab1.lab1_publisher:main',
            'lab1_subscriber = lab1.lab1_subscriber:main',
        ],
    },
)
