from setuptools import find_packages, setup

package_name = 'ras598_assignment_3'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mohammadnsr1',
    maintainer_email='nasrmohammad661@gmail.com',
    description='ras598_assignment_3',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'bayes_filter = ras598_assignment_3.bayes_boilerplate:main',
        ],
    },
)
