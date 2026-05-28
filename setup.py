#!/usr/bin/env python3
"""ROS catkin 安装脚本 —— 将 chess_robot_arm 的 Python 子模块安装到系统路径。"""
from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=[
        'chess_robot_arm',
        'chess_robot_arm.vision',
        'chess_robot_arm.chess_engine',
        'chess_robot_arm.robot_arm',
        'chess_robot_arm.utils',
    ],
    package_dir={
        'chess_robot_arm': '',
        'chess_robot_arm.vision': 'vision',
        'chess_robot_arm.chess_engine': 'chess_engine',
        'chess_robot_arm.robot_arm': 'robot_arm',
        'chess_robot_arm.utils': 'utils',
    },
)

setup(**d)
