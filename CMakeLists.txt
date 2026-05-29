cmake_minimum_required(VERSION 3.0.2)
project(chess_robot_arm)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  cv_bridge
  message_generation
)

# =====================================================
# 自定义消息（从 EE368_Project 并入，消除外部依赖）
# =====================================================
add_message_files(
  FILES
  ArucoMarker.msg
  ArucoMarkerArray.msg
  ChessboardCorners.msg
  ChessboardPixelCorners.msg
  PickAndPlaceGoalInCamera.msg
  RegionMatrix.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    geometry_msgs
    sensor_msgs
    cv_bridge
    message_runtime
)

# =====================================================
# 安装可执行 Python 脚本到 ROS bin 目录
# （rosrun chess_robot_arm xxx.py 使用）
# =====================================================
catkin_install_python(PROGRAMS
  vision/board_detector.py
  vision/piece_classifier.py
  main.py
  robot_arm/motion_planner.py
  utils/calibration.py
  utils/pick_place_calibrate.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# =====================================================
# 安装 Python 模块到 dist-packages
# （from chess_robot_arm.xxx import yyy 使用）
# =====================================================
install(FILES
  __init__.py
  config.py
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}/chess_robot_arm
)

install(DIRECTORY
  utils
  vision
  chess_engine
  robot_arm
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}/chess_robot_arm
  FILES_MATCHING PATTERN "*.py"
  PATTERN ".git" EXCLUDE
)

# =====================================================
# 安装 launch 文件
# =====================================================
install(DIRECTORY
  launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
