from setuptools import find_packages, setup
import os
from glob import glob


package_name = 'ras598_assignment_2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name,'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'maps'), glob('maps/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mohammadnsr1',
    maintainer_email='nasrmohammad661@gmail.com',
    description='ras598 assignment 2',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'planner = ras598_assignment_2.planner:main',
            'controller = ras598_assignment_2.controller:main',
        ],
    },
)
