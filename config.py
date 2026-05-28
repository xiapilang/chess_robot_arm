"""
集中配置加载器。
从 ROS 参数服务器读取配置，若参数不存在则回退到 constants.py 中的默认值。
"""

import rospy
import numpy as np
from chess_robot_arm.utils.constants import (
    BOARD_ROWS, BOARD_COLS,
    ARUCO_TOP_LEFT_ID, ARUCO_TOP_RIGHT_ID,
    ARUCO_BOTTOM_LEFT_ID, ARUCO_BOTTOM_RIGHT_ID,
    ARUCO_MARKER_SIZE, ARUCO_DICT_NAME,
    PIECE_ID_TO_FEN, PIECE_CLASS_NAMES,
    DH_PARAMETERS, NUM_JOINTS,
    T_EE_CAMERA_TRANSLATION, T_EE_CAMERA_ROTATION,
    HOME_JOINT_ANGLES_DEG,
    DEFAULT_GRIPPER_ORIENTATION_DEG,
    PRE_ACTION_Z_LIFT, PICK_Z_OFFSET, PLACE_Z_OFFSET,
    GRIPPER_OPEN_VALUE, GRIPPER_CLOSE_VALUE, MOTION_DURATION,
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    STOCKFISH_PATH, STOCKFISH_SKILL_LEVEL, STOCKFISH_THINK_TIME,
)


def get_param(name, default):
    """从 ROS 参数服务器读取参数，不存在时返回默认值。"""
    return rospy.get_param(name, default)


class Config:
    """配置单例，在初始化时从 ROS param 服务器读取所有参数。"""

    def __init__(self):
        # --- 棋盘参数 ---
        self.board_rows = BOARD_ROWS
        self.board_cols = BOARD_COLS

        # --- ArUco 标签参数 ---
        self.aruco_top_left_id     = get_param("~aruco_top_left_id", ARUCO_TOP_LEFT_ID)
        self.aruco_top_right_id    = get_param("~aruco_top_right_id", ARUCO_TOP_RIGHT_ID)
        self.aruco_bottom_left_id  = get_param("~aruco_bottom_left_id", ARUCO_BOTTOM_LEFT_ID)
        self.aruco_bottom_right_id = get_param("~aruco_bottom_right_id", ARUCO_BOTTOM_RIGHT_ID)
        self.aruco_marker_size     = get_param("~aruco_marker_size", ARUCO_MARKER_SIZE)
        self.aruco_dict_name       = get_param("~aruco_dict_name", ARUCO_DICT_NAME)

        # --- 相机参数 ---
        self.camera_width  = get_param("~camera_width", CAMERA_WIDTH)
        self.camera_height = get_param("~camera_height", CAMERA_HEIGHT)
        self.camera_fps    = get_param("~camera_fps", CAMERA_FPS)

        # --- 运动学参数 ---
        self.dh_parameters = DH_PARAMETERS
        self.num_joints    = NUM_JOINTS
        self.T_ee_camera_translation = get_param(
            "~T_ee_camera_translation", T_EE_CAMERA_TRANSLATION)
        self.T_ee_camera_rotation = get_param(
            "~T_ee_camera_rotation", T_EE_CAMERA_ROTATION)
        self.home_joint_angles_deg = get_param(
            "~home_joint_angles_deg", HOME_JOINT_ANGLES_DEG)

        # --- 运动控制参数 ---
        self.gripper_orientation_deg = get_param(
            "~gripper_orientation_deg", DEFAULT_GRIPPER_ORIENTATION_DEG)
        self.pre_action_z_lift  = get_param("~pre_action_z_lift", PRE_ACTION_Z_LIFT)
        self.pick_z_offset      = get_param("~pick_z_offset", PICK_Z_OFFSET)
        self.place_z_offset     = get_param("~place_z_offset", PLACE_Z_OFFSET)
        self.gripper_open_value  = get_param("~gripper_open_value", GRIPPER_OPEN_VALUE)
        self.gripper_close_value = get_param("~gripper_close_value", GRIPPER_CLOSE_VALUE)
        self.motion_duration     = get_param("~motion_duration", MOTION_DURATION)

        # --- AI 参数 ---
        self.stockfish_path       = get_param("~stockfish_path", STOCKFISH_PATH)
        self.stockfish_skill_level = get_param("~stockfish_skill_level", STOCKFISH_SKILL_LEVEL)
        self.stockfish_think_time  = get_param("~stockfish_think_time", STOCKFISH_THINK_TIME)

        # --- 构建手眼变换矩阵 T_ee_camera ---
        self.T_ee_camera = np.identity(4)
        self.T_ee_camera[:3, :3] = np.array(self.T_ee_camera_rotation)
        self.T_ee_camera[:3, 3]  = np.array(self.T_ee_camera_translation)

        # --- 棋子类别映射 ---
        self.piece_id_to_fen = PIECE_ID_TO_FEN
        self.piece_class_names = PIECE_CLASS_NAMES
        self.empty_square = '0'

        # --- 机械臂名称 ---
        self.robot_name = get_param("~robot_name", "my_gen3_lite")


# 全局配置单例
config = None

def get_config():
    """获取全局配置单例（首次调用时初始化）。"""
    global config
    if config is None:
        config = Config()
    return config
