
# EE3033-Balancing-Robot

## Project Overview

This project implements a **ROS 2-based autonomous exploration and target localization system** for a two-wheeled balancing robot.

The system integrates **LiDAR-based SLAM, AMCL localization, Nav2 navigation, frontier-based exploration, and YOLO-based object detection**.

## Main Features

- **LiDAR-Based Mapping and Localization**
- **Autonomous Frontier Exploration**
- **YOLO-Based Bottle Detection**
- **LiDAR-Based Target Position Estimation**
- **Rotational Camera Scanning**
- **RViz2 Target Visualization**
- **Autonomous Return-to-Start Navigation**

## Main Files

### `lidar_coverage_exploration.py`

Implements autonomous frontier exploration, target localization, rotational camera scanning, navigation failure handling, and mission control.

### `knownwp.py`

Provides the ROS 2 launch configuration for robot bringup, LiDAR sensing, AMCL localization, Nav2 navigation, and coverage exploration.

## System Requirements

### Software

- ROS 2
- Nav2 Navigation Framework
- N10P LiDAR Driver
- YOLO ROS 2 Detection Interface
- Required robot-specific ROS 2 packages and configuration files

### Hardware

- Two-wheeled balancing robot
- Raspberry Pi onboard computer
- N10P LiDAR sensor
- USB camera
- External Ubuntu VM for object detection

## System Workflow

1. **Mapping:** Construct and save an occupancy grid map using LiDAR-based SLAM.
2. **Localization:** Initialize the robot's position using AMCL.
3. **Exploration:** Navigate autonomously between selected frontier viewpoints.
4. **Detection:** Search for the target bottle using YOLO.
5. **Target Localization:** Estimate the target coordinates using LiDAR measurements.
6. **Mission Completion:** Publish the estimated target position and return to the starting location.

## Authors

**EE3033 Group 03**
> **Note:** This repository contains the two main Python scripts developed for the project. The system also depends on additional ROS 2 packages, launch files, and configuration files provided by the robot platform and development environment.
