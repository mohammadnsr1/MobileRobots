# MobileRobots

ROS 2 coursework repository containing a workspace of mobile robotics labs

## Overview

This repository is organized as a ROS 2 Python workspace under `labs/`. The packages currently included cover:

- basic ROS 2 pub/sub with `rclpy`
- TF2 frame broadcasting and listening with `turtlesim`
- simple scripted robot motion
- reactive obstacle avoidance using `LaserScan`
- point cloud processing for cylinder detection and RViz visualization

## Repository Structure

```text
MobileRobots/
├── README.md
├── LICENSE
├── gz_world.sdf
└── labs/
    ├── src/
    │   ├── lab1/
    │   ├── lab2/
    │   ├── lab3/
    │   ├── lab4/
    │   └── assignment1/
    ├── build/
    ├── install/
    └── log/
```

`build/`, `install/`, and `log/` are generated workspace artifacts from `colcon`.
