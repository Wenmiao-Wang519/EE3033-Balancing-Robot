# EE3033-Balancing-Robot
Project Overview

This project implements a ROS 2-based autonomous exploration and target localization system for a two-wheeled balancing robot.

The system integrates LiDAR-based SLAM, AMCL localization, Nav2 navigation, frontier-based exploration, and YOLO-based object detection.

Main Features

LiDAR-based mapping and localization

Autonomous frontier exploration

YOLO-based bottle detection

LiDAR-based target position estimation

Rotational camera scanning

RViz2 target visualization

Autonomous return-to-start navigation

Main Files

lidar_coverage_exploration.py — Autonomous exploration, target localization, and mission control.

knownwp.py — ROS 2 launch configuration for robot bringup, localization, navigation, and exploration.

System Requirements

ROS 2 with Nav2

N10P LiDAR driver

YOLO ROS 2 detection interface

Raspberry Pi onboard computer

Ubuntu VM for object detection

Required robot-specific ROS 2 packages and configuration files

Authors

EE3033 Group 03
