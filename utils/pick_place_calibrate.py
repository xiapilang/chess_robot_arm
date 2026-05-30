#!/usr/bin/env python3
"""
抓取-放置手动校准脚本。

重复执行: 从棋盘上抓取一个棋子 → 放置到另一个格子。
每次运行后观察偏差，修改 constants.py 中的偏移参数，再运行验证。

用法:
    rosrun chess_robot_arm pick_place_calibrate.py

需要先启动 board_detector.py 提供 ArUco 角点。

修改下方 SRC_SQUARE / DST_SQUARE 可变更校准格子。
修改 constants.py 中的以下参数可微调精度:
    PICK_XY_OFFSET  = {"x": 0.0, "y": 0.0}   # 抓取点 X/Y 偏移
    PLACE_XY_OFFSET = {"x": 0.0, "y": 0.0}   # 放置点 X/Y 偏移
    PICK_Z_OFFSET   = -0.008                  # 抓取点 Z 偏移
    PLACE_Z_OFFSET  = -0.008                  # 放置点 Z 偏移
    MIN_APPROACH_Z  = 0.085                   # 最低逼近高度
    PICK_TRANSIT    = {...}                   # 抓取前安全中转点
"""

import os
import sys
import math
import time
import rospy
import chess
import cv2
import numpy as np

# 确保包路径可用
_script_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_root = os.path.dirname(_script_dir)
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

from chess_robot_arm.robot_arm.arm_controller import KinovaArmController
from chess_robot_arm.msg import ChessboardCorners
from chess_robot_arm.utils.constants import (
    BOARD_ROWS, BOARD_COLS, BOARD_CORNERS_BASE,
    PICK_XY_OFFSET, PLACE_XY_OFFSET,
    PICK_Z_OFFSET, PLACE_Z_OFFSET,
    MIN_APPROACH_Z, PICK_TRANSIT, POST_CALIB_HOME,
    PRE_ACTION_Z_LIFT, ROW_PICK_OFFSETS, ROW_PLACE_OFFSETS,
    COL_PICK_OFFSETS, COL_PLACE_OFFSETS, ROW_Z_OFFSET,
    GRIPPER_TILT_THRESHOLD, GRIPPER_MAX_TILT_DEG,
    GRIPPER_OPEN_VALUE, GRIPPER_CLOSE_VALUE,
    DEFAULT_GRIPPER_ORIENTATION_DEG,
)

# ============================================================
# 校准配置
# ============================================================
SRC_SQUARE = "f1"       # 从哪个格子抓取棋子
DST_SQUARE = "h8"       # 放置到哪个格子

# 夹爪姿态
GRIPPER_RX, GRIPPER_RY, GRIPPER_RZ = DEFAULT_GRIPPER_ORIENTATION_DEG


# ============================================================
# 坐标计算
# ============================================================

def uci_to_row_col(uci_square):
    """将 UCI 坐标名转换为矩阵 (row, col)。row 0=第8横排, col 0=左边缘(h线)。"""
    file_idx = chess.FILE_NAMES.index(uci_square[0])
    rank = int(uci_square[1:])
    row = 8 - rank
    col = 7 - file_idx   # 相机黑方视角，h线在图像左侧
    return row, col


def calibrate_homography(cam_corners, base_corners):
    """用四对点标定单应性矩阵（相机 XY → 基座 XY），可处理透视畸变。"""
    src = cam_corners.reshape(4, 1, 2).astype(np.float32)
    dst = base_corners.reshape(4, 1, 2).astype(np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H


def matrix_to_base_point_homography(col, row, cam_corners, H, board_z):
    """ArUco 相机帧双线性插值 + 单应性变换 → 基座标系 3D 点。"""
    u = col / float(BOARD_COLS - 1)
    v = row / float(BOARD_ROWS - 1)

    c_tl, c_tr = cam_corners[0], cam_corners[1]
    c_bl, c_br = cam_corners[2], cam_corners[3]
    top = c_tl + u * (c_tr - c_tl)
    bot = c_bl + u * (c_br - c_bl)
    cam_xy = top + v * (bot - top)

    pts = np.array([[[cam_xy[0], cam_xy[1]]]], dtype=np.float32)
    base_xy = cv2.perspectiveTransform(pts, H)[0][0]
    return float(base_xy[0]), float(base_xy[1]), float(board_z)


# ============================================================
# 抓取-放置逻辑
# ============================================================

def compute_gripper_orient(x, y, default_orient):
    """远点自动倾斜夹爪，yaw指向目标方向，倾斜角≤GRIPPER_MAX_TILT_DEG。"""
    dist = math.hypot(x, y)
    if dist <= GRIPPER_TILT_THRESHOLD:
        return default_orient
    tilt_deg = min((dist - GRIPPER_TILT_THRESHOLD) / 0.25 * GRIPPER_MAX_TILT_DEG,
                   GRIPPER_MAX_TILT_DEG)
    target_yaw = math.degrees(math.atan2(y, x))
    return np.array([0.0, 180.0 - tilt_deg, target_yaw])


def pick_and_place(arm, src_uci, dst_uci, to_base_func):
    """
    执行一次抓取-放置：从 src_uci 抓取 → 放置到 dst_uci。
    to_base_func(col, row) → (x, y, z)，应用 constants.py 中的偏移量。
    """
    drx, dry, drz = GRIPPER_RX, GRIPPER_RY, GRIPPER_RZ  # 默认垂直姿态

    # --- 计算目标位置 ---
    s_row, s_col = uci_to_row_col(src_uci)
    d_row, d_col = uci_to_row_col(dst_uci)

    s_x, s_y, s_z = to_base_func(s_col, s_row)
    d_x, d_y, d_z = to_base_func(d_col, d_row)

    # 应用偏移（全局 + 逐排 + 逐列，行列叠加）
    pick_row_off = ROW_PICK_OFFSETS.get(s_row, {"x": 0.0, "y": 0.0})
    pick_col_off = COL_PICK_OFFSETS.get(s_col, {"x": 0.0, "y": 0.0})
    place_row_off = ROW_PLACE_OFFSETS.get(d_row, {"x": 0.0, "y": 0.0})
    place_col_off = COL_PLACE_OFFSETS.get(d_col, {"x": 0.0, "y": 0.0})

    pick_x = s_x + PICK_XY_OFFSET["x"] + pick_row_off["x"] + pick_col_off["x"]
    pick_y = s_y + PICK_XY_OFFSET["y"] + pick_row_off["y"] + pick_col_off["y"]
    pick_z = s_z + ROW_Z_OFFSET.get(s_row, PICK_Z_OFFSET)

    place_x = d_x + PLACE_XY_OFFSET["x"] + place_row_off["x"] + place_col_off["x"]
    place_y = d_y + PLACE_XY_OFFSET["y"] + place_row_off["y"] + place_col_off["y"]
    place_z = d_z + ROW_Z_OFFSET.get(d_row, PLACE_Z_OFFSET)

    rospy.loginfo(f"抓取 {src_uci}: matrix({s_row},{s_col}) → base({s_x:.3f},{s_y:.3f},{s_z:.3f})")
    if pick_row_off["x"] or pick_row_off["y"]:
        rospy.loginfo(f"  逐排偏移 row={s_row}: x={pick_row_off['x']:.3f} y={pick_row_off['y']:.3f}")
    if pick_col_off["x"] or pick_col_off["y"]:
        rospy.loginfo(f"  逐列偏移 col={s_col}: x={pick_col_off['x']:.3f} y={pick_col_off['y']:.3f}")
    rospy.loginfo(f"  +XY偏移 → ({pick_x:.3f},{pick_y:.3f},{pick_z:.3f})")
    rospy.loginfo(f"放置 {dst_uci}: matrix({d_row},{d_col}) → base({d_x:.3f},{d_y:.3f},{d_z:.3f})")
    if place_row_off["x"] or place_row_off["y"]:
        rospy.loginfo(f"  逐排偏移 row={d_row}: x={place_row_off['x']:.3f} y={place_row_off['y']:.3f}")
    if place_col_off["x"] or place_col_off["y"]:
        rospy.loginfo(f"  逐列偏移 col={d_col}: x={place_col_off['x']:.3f} y={place_col_off['y']:.3f}")
    rospy.loginfo(f"  +XY偏移 → ({place_x:.3f},{place_y:.3f},{place_z:.3f})")

    # --- 计算抓取/放置姿态（远点自动倾斜） ---
    pick_orient = compute_gripper_orient(pick_x, pick_y, [drx, dry, drz])
    place_orient = compute_gripper_orient(place_x, place_y, [drx, dry, drz])
    prx, pry_, prz = pick_orient
    plrx, plry, plrz = place_orient

    # --- 步骤 0: 移动到安全中转点（始终用默认垂直姿态） ---
    rospy.loginfo(f"步骤0: 移动到安全中转点 ({PICK_TRANSIT['x']:.3f}, {PICK_TRANSIT['y']:.3f}, {PICK_TRANSIT['z']:.3f})")
    if not arm.move_to_cartesian_pose(PICK_TRANSIT["x"], PICK_TRANSIT["y"], PICK_TRANSIT["z"], drx, dry, drz):
        rospy.logerr("移动到中转点失败。")
        return False

    # --- 步骤 1: 移动到抓取点上方 ---
    pick_above_z = max(pick_z + PRE_ACTION_Z_LIFT, MIN_APPROACH_Z)
    rospy.loginfo(f"步骤1: 移动到抓取点上方 ({pick_x:.3f}, {pick_y:.3f}, {pick_above_z:.3f})")
    if not arm.move_to_cartesian_pose(pick_x, pick_y, pick_above_z, prx, pry_, prz):
        rospy.logerr("移动到抓取点上方失败。")
        return False

    # --- 步骤 2: 张开夹爪 ---
    rospy.loginfo(f"步骤2: 张开夹爪 ({GRIPPER_OPEN_VALUE*100:.0f}%)")
    arm.move_gripper(GRIPPER_OPEN_VALUE)
    rospy.sleep(0.5)

    # --- 步骤 3: 下降到抓取位置 ---
    rospy.loginfo(f"步骤3: 下降到抓取位置 ({pick_x:.3f}, {pick_y:.3f}, {pick_z:.3f})")
    if not arm.move_to_cartesian_pose(pick_x, pick_y, pick_z, prx, pry_, prz):
        rospy.logerr("下降到抓取位置失败。")
        return False
    rospy.sleep(0.3)

    # --- 步骤 4: 闭合夹爪 ---
    rospy.loginfo(f"步骤4: 闭合夹爪 ({GRIPPER_CLOSE_VALUE*100:.0f}%)")
    arm.move_gripper(GRIPPER_CLOSE_VALUE)
    rospy.sleep(1.0)

    # --- 步骤 5: 抬升 ---
    rospy.loginfo(f"步骤5: 抬升到抓取点上方 ({pick_x:.3f}, {pick_y:.3f}, {pick_above_z:.3f})")
    if not arm.move_to_cartesian_pose(pick_x, pick_y, pick_above_z, prx, pry_, prz):
        rospy.logerr("抬升失败。")
        return False

    # --- 步骤 6: 移动到放置点上方 ---
    place_above_z = max(place_z + PRE_ACTION_Z_LIFT, MIN_APPROACH_Z)
    rospy.loginfo(f"步骤6: 移动到放置点上方 ({place_x:.3f}, {place_y:.3f}, {place_above_z:.3f})")
    if not arm.move_to_cartesian_pose(place_x, place_y, place_above_z, plrx, plry, plrz):
        rospy.logerr("移动到放置点上方失败。")
        return False

    # --- 步骤 7: 下降到放置位置 ---
    rospy.loginfo(f"步骤7: 下降到放置位置 ({place_x:.3f}, {place_y:.3f}, {place_z:.3f})")
    if not arm.move_to_cartesian_pose(place_x, place_y, place_z, plrx, plry, plrz):
        rospy.logerr("下降到放置位置失败。")
        return False
    rospy.sleep(0.3)

    # --- 步骤 8: 张开夹爪释放棋子 ---
    rospy.loginfo(f"步骤8: 释放棋子 ({GRIPPER_OPEN_VALUE*100:.0f}%)")
    arm.move_gripper(GRIPPER_OPEN_VALUE)
    rospy.sleep(1.0)

    # --- 步骤 9: 抬升 ---
    rospy.loginfo(f"步骤9: 抬升 ({place_x:.3f}, {place_y:.3f}, {place_above_z:.3f})")
    if not arm.move_to_cartesian_pose(place_x, place_y, place_above_z, plrx, plry, plrz):
        rospy.logerr("抬升失败。")
        return False

    # --- 步骤 10: 归位（始终用避让位姿自带姿态） ---
    rospy.loginfo(f"步骤10: 归位到避让位置 ({POST_CALIB_HOME['x']:.3f}, {POST_CALIB_HOME['y']:.3f}, {POST_CALIB_HOME['z']:.3f})")
    if not arm.move_to_cartesian_pose(POST_CALIB_HOME["x"], POST_CALIB_HOME["y"],
                                       POST_CALIB_HOME["z"],
                                       POST_CALIB_HOME["rx"], POST_CALIB_HOME["ry"],
                                       POST_CALIB_HOME["rz"]):
        rospy.logerr("归位失败。")

    rospy.loginfo("抓取-放置校准循环完成。")
    return True


# ============================================================
# ArUco 角点收集
# ============================================================

class ArucoCornerCollector:
    """订阅一帧 ArUco 角点并缓存。"""

    def __init__(self, timeout=15.0):
        self.corners = None
        self.received = False
        self.sub = rospy.Subscriber(
            "/chessboard_corners_3d", ChessboardCorners, self._cb)
        rospy.loginfo("等待 ArUco 角点 (/chessboard_corners_3d)...")
        start = time.time()
        while not rospy.is_shutdown() and not self.received:
            if time.time() - start > timeout:
                rospy.logwarn("ArUco 角点等待超时。")
                break
            rospy.sleep(0.1)
        if self.received:
            self.sub.unregister()

    def _cb(self, msg):
        if self.received:
            return
        self.corners = np.array([
            [msg.top_left.x,     msg.top_left.y],
            [msg.top_right.x,    msg.top_right.y],
            [msg.bottom_left.x,  msg.bottom_left.y],
            [msg.bottom_right.x, msg.bottom_right.y],
        ])
        self.received = True
        rospy.loginfo(f"收到 ArUco 相机帧角点:\n{np.round(self.corners, 3)}")


# ============================================================
# 主函数
# ============================================================

def main():
    rospy.init_node('pick_place_calibrate', anonymous=True)

    # 打印当前偏移参数
    rospy.loginfo("=" * 60)
    rospy.loginfo("当前偏移参数 (constants.py):")
    rospy.loginfo(f"  PICK_XY_OFFSET  = {PICK_XY_OFFSET}")
    rospy.loginfo(f"  PLACE_XY_OFFSET = {PLACE_XY_OFFSET}")
    rospy.loginfo(f"  PICK_Z_OFFSET   = {PICK_Z_OFFSET}")
    rospy.loginfo(f"  PLACE_Z_OFFSET  = {PLACE_Z_OFFSET}")
    rospy.loginfo(f"  MIN_APPROACH_Z  = {MIN_APPROACH_Z}")
    rospy.loginfo(f"  PICK_TRANSIT    = {PICK_TRANSIT}")
    rospy.loginfo("=" * 60)
    rospy.loginfo(f"测试走法: {SRC_SQUARE} → {DST_SQUARE}")

    # --- ArUco 角点采集 + 仿射标定 ---
    collector = ArucoCornerCollector(timeout=30.0)
    if not collector.received:
        rospy.logerr("未收到 ArUco 角点，请确认 board_detector.py 正在运行且 4 个标签均被检测到。")
        return

    bc = BOARD_CORNERS_BASE
    base_corners = np.array([
        [bc["top_left"]["x"],     bc["top_left"]["y"]],
        [bc["top_right"]["x"],    bc["top_right"]["y"]],
        [bc["bottom_left"]["x"],  bc["bottom_left"]["y"]],
        [bc["bottom_right"]["x"], bc["bottom_right"]["y"]],
    ])
    H = calibrate_homography(collector.corners, base_corners)
    board_z = bc["top_left"]["z"]
    rospy.loginfo(f"单应性矩阵:\n{np.round(H, 4)}")
    rospy.loginfo(f"基座四角:\n{np.round(base_corners, 3)}")

    def to_base_func(col, row):
        return matrix_to_base_point_homography(
            col, row, collector.corners, H, board_z)

    # --- 初始化机械臂 ---
    arm = KinovaArmController()
    if not arm.is_init_success:
        rospy.logerr("机械臂控制器初始化失败。")
        return

    if not arm.activate():
        rospy.logerr("机械臂激活失败。")
        return

    # --- 执行校准抓放 ---
    success = pick_and_place(arm, SRC_SQUARE, DST_SQUARE, to_base_func)

    if success:
        rospy.loginfo("校准完成。观察放置精度：")
        rospy.loginfo("  - 若 X 偏左/右: 调整 PICK_XY_OFFSET/PLACE_XY_OFFSET 的 'x'")
        rospy.loginfo("  - 若 Y 偏前/后: 调整 PICK_XY_OFFSET/PLACE_XY_OFFSET 的 'y'")
        rospy.loginfo("  - 若 Z 太高/太低: 调整 PICK_Z_OFFSET/PLACE_Z_OFFSET")
        rospy.loginfo("  - 若途中撞到棋子: 增大 MIN_APPROACH_Z 或调整 PICK_TRANSIT")


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
