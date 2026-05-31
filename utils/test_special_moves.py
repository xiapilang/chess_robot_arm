#!/usr/bin/env python3
"""
特殊走法机械臂实测脚本: 王车易位 / 过路兵 / 兵升变。

用法:
    rosrun chess_robot_arm test_special_moves.py

需要先启动 board_detector.py 提供 ArUco 角点。
终端输入数字选择:
    1 = 王车易位（黑方短易位: e8→g8, 车 h8→f8）
    2 = 过路兵（黑兵 a4 吃白兵 b4 过路至 b3）
    3 = 兵升变（黑兵 a2→a1 升变为后）
"""

import os
import sys
import math
import time
import rospy
import chess
import cv2
import numpy as np

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
    GRIPPER_TILT_THRESHOLD, GRIPPER_MAX_JOINT5_DEG, GRIPPER_FIXED_YAW_DEG,
    FAR_TRANSIT, FAR_PICK_Z_OFFSET, FAR_PLACE_Z_OFFSET,
    FAR_GRIPPER_CLOSE, FAR_ROW_Z_OFFSET, SPECIAL_Z_OVERRIDE, FAR_CELLS,
    GRIPPER_OPEN_VALUE, GRIPPER_CLOSE_VALUE,
    GARBAGE_POINT, GARBAGE_STEP,
    DEFAULT_GRIPPER_ORIENTATION_DEG,
)

GRIPPER_RX, GRIPPER_RY, GRIPPER_RZ = DEFAULT_GRIPPER_ORIENTATION_DEG


# ============================================================
# 坐标 + 姿态计算（与校准脚本一致）
# ============================================================

def uci_to_row_col(uci_square):
    file_idx = chess.FILE_NAMES.index(uci_square[0])
    rank = int(uci_square[1:])
    return (8 - rank, 7 - file_idx)


def calibrate_homography(cam_corners, base_corners):
    src = cam_corners.reshape(4, 1, 2).astype(np.float32)
    dst = base_corners.reshape(4, 1, 2).astype(np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H


def to_base_point(col, row, cam_corners, H, board_z):
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


def compute_z(row, col, is_far=False, is_pick=True):
    sp = SPECIAL_Z_OVERRIDE.get((row, col))
    if sp is not None:
        return sp
    base = ROW_Z_OFFSET.get(row, PICK_Z_OFFSET if is_pick else PLACE_Z_OFFSET)
    if is_far:
        base += FAR_PICK_Z_OFFSET if is_pick else FAR_PLACE_Z_OFFSET
        base += FAR_ROW_Z_OFFSET.get(row, 0.0)
    return base


def compute_xy(s_x, s_y, row, col, is_pick=True):
    roff = ROW_PICK_OFFSETS.get(row, {"x": 0.0, "y": 0.0}) if is_pick \
           else ROW_PLACE_OFFSETS.get(row, {"x": 0.0, "y": 0.0})
    coff = COL_PICK_OFFSETS.get(col, {"x": 0.0, "y": 0.0}) if is_pick \
           else COL_PLACE_OFFSETS.get(col, {"x": 0.0, "y": 0.0})
    gxy = PICK_XY_OFFSET if is_pick else PLACE_XY_OFFSET
    return s_x + gxy["x"] + roff["x"] + coff["x"], \
           s_y + gxy["y"] + roff["y"] + coff["y"]


def compute_gripper_orient(x, y):
    dist = math.hypot(x, y)
    is_far = dist > 0.40  # simplified check
    if dist <= GRIPPER_TILT_THRESHOLD:
        tilt = 0.0
    else:
        tilt = min((dist - GRIPPER_TILT_THRESHOLD) / 0.25 * GRIPPER_MAX_JOINT5_DEG,
                   GRIPPER_MAX_JOINT5_DEG)
    return np.array([0.0, 180.0 - tilt, GRIPPER_FIXED_YAW_DEG])


# ============================================================
# 抓取/放置公共动作
# ============================================================

def move_to(arm, x, y, z, orient):
    rx, ry, rz = orient
    return arm.move_to_cartesian_pose(x, y, z, rx, ry, rz)


def do_pick(arm, x, y, z, orient, gripper_close=GRIPPER_CLOSE_VALUE):
    """标准抓取序列: 中转→上方→张开→下降→闭合→抬升。"""
    above_z = max(z + PRE_ACTION_Z_LIFT, MIN_APPROACH_Z)
    # 中转
    rospy.loginfo(f"  移动到中转点 ({PICK_TRANSIT['x']:.3f},{PICK_TRANSIT['y']:.3f},{PICK_TRANSIT['z']:.3f})")
    if not arm.move_to_cartesian_pose(
        PICK_TRANSIT["x"], PICK_TRANSIT["y"], PICK_TRANSIT["z"], GRIPPER_RX, GRIPPER_RY, GRIPPER_RZ):
        return False
    # 上方
    if not move_to(arm, x, y, above_z, orient):
        return False
    # 张开
    arm.move_gripper(GRIPPER_OPEN_VALUE)
    rospy.sleep(0.5)
    # 下降
    if not move_to(arm, x, y, z, orient):
        return False
    rospy.sleep(0.3)
    # 闭合
    arm.move_gripper(gripper_close)
    rospy.sleep(1.0)
    # 抬升
    if not move_to(arm, x, y, above_z, orient):
        return False
    return True


def do_place(arm, x, y, z, orient):
    """标准放置序列: 上方→下降→张开→抬升。"""
    above_z = max(z + PRE_ACTION_Z_LIFT, MIN_APPROACH_Z)
    if not move_to(arm, x, y, above_z, orient):
        return False
    if not move_to(arm, x, y, z, orient):
        return False
    rospy.sleep(0.3)
    arm.move_gripper(GRIPPER_OPEN_VALUE)
    rospy.sleep(1.0)
    if not move_to(arm, x, y, above_z, orient):
        return False
    return True


# ============================================================
# 三类特殊走法
# ============================================================

def run_castling(arm, cam_corners, H, board_z):
    """黑方短易位: e8→g8, 车 h8→f8。"""
    king_from = uci_to_row_col("e8")  # (0, 3)
    king_to   = uci_to_row_col("g8")  # (0, 1)
    rook_from = uci_to_row_col("h8")  # (0, 0)
    rook_to   = uci_to_row_col("f8")  # (0, 2)

    rospy.loginfo("=== 王车易位: 先移王 e8→g8, 再移车 h8→f8 ===")

    # 移王
    s_row, s_col = king_from
    d_row, d_col = king_to
    s_x, s_y, _ = to_base_point(s_col, s_row, cam_corners, H, board_z)
    d_x, d_y, _ = to_base_point(d_col, d_row, cam_corners, H, board_z)
    pick_z = board_z + compute_z(s_row, s_col, True, True)
    place_z = board_z + compute_z(d_row, d_col, True, False)
    px, py = compute_xy(s_x, s_y, s_row, s_col, True)
    plx, ply = compute_xy(d_x, d_y, d_row, d_col, False)
    orient_pick = compute_gripper_orient(px, py)
    orient_place = compute_gripper_orient(plx, ply)

    rospy.loginfo(f"  王: ({s_row},{s_col})→({d_row},{d_col})")
    if not do_pick(arm, px, py, pick_z, orient_pick):
        rospy.logerr("抓取王失败")
        return
    if not do_place(arm, plx, ply, place_z, orient_place):
        rospy.logerr("放置王失败")
        return
    rospy.sleep(1.0)

    # 移车
    s_row, s_col = rook_from
    d_row, d_col = rook_to
    s_x, s_y, _ = to_base_point(s_col, s_row, cam_corners, H, board_z)
    d_x, d_y, _ = to_base_point(d_col, d_row, cam_corners, H, board_z)
    pick_z = board_z + compute_z(s_row, s_col, True, True)
    place_z = board_z + compute_z(d_row, d_col, True, False)
    px, py = compute_xy(s_x, s_y, s_row, s_col, True)
    plx, ply = compute_xy(d_x, d_y, d_row, d_col, False)
    orient_pick = compute_gripper_orient(px, py)
    orient_place = compute_gripper_orient(plx, ply)

    rospy.loginfo(f"  车: ({s_row},{s_col})→({d_row},{d_col})")
    if not do_pick(arm, px, py, pick_z, orient_pick):
        rospy.logerr("抓取车失败")
        return
    if not do_place(arm, plx, ply, place_z, orient_place):
        rospy.logerr("放置车失败")
        return

    rospy.loginfo("王车易位完成。")


def run_en_passant(arm, cam_corners, H, board_z):
    """过路兵: 黑兵 a4 吃白兵 b4 过路至 b3。被吃白兵在 b4，不移到弃子区直接移除。"""
    pawn_from = uci_to_row_col("a4")   # (4, 7)
    pawn_to   = uci_to_row_col("b3")   # (5, 6)
    captured  = uci_to_row_col("b4")   # (4, 6) — 被吃白兵位置

    rospy.loginfo("=== 过路兵: 黑兵 a4→b3, 被吃兵在 b4 ===")

    # 先移除被吃白兵到弃子区
    capt_row, capt_col = captured
    s_x, s_y, _ = to_base_point(capt_col, capt_row, cam_corners, H, board_z)
    pick_z = board_z + compute_z(capt_row, capt_col, True, True)
    px, py = compute_xy(s_x, s_y, capt_row, capt_col, True)
    orient = compute_gripper_orient(px, py)

    rospy.loginfo(f"  移除被吃兵: ({capt_row},{capt_col}) → 弃子区")
    if not do_pick(arm, px, py, pick_z, orient):
        rospy.logerr("抓取被吃兵失败")
        return

    garbage_z = GARBAGE_POINT["z"]
    if not do_place(arm, GARBAGE_POINT["x"], GARBAGE_POINT["y"], garbage_z, orient):
        rospy.logerr("放置被吃兵到弃子区失败")
        return
    rospy.sleep(1.0)

    # 移动黑兵
    s_row, s_col = pawn_from
    d_row, d_col = pawn_to
    s_x, s_y, _ = to_base_point(s_col, s_row, cam_corners, H, board_z)
    d_x, d_y, _ = to_base_point(d_col, d_row, cam_corners, H, board_z)
    pick_z = board_z + compute_z(s_row, s_col, True, True)
    place_z = board_z + compute_z(d_row, d_col, False, False)
    px, py = compute_xy(s_x, s_y, s_row, s_col, True)
    plx, ply = compute_xy(d_x, d_y, d_row, d_col, False)
    orient_pick = compute_gripper_orient(px, py)
    orient_place = compute_gripper_orient(plx, ply)

    rospy.loginfo(f"  移动兵: ({s_row},{s_col})→({d_row},{d_col})")
    if not do_pick(arm, px, py, pick_z, orient_pick):
        rospy.logerr("抓取兵失败")
        return
    if not do_place(arm, plx, ply, place_z, orient_place):
        rospy.logerr("放置兵失败")
        return

    rospy.loginfo("过路兵完成。")


def run_promotion(arm, cam_corners, H, board_z):
    """兵升变: 黑兵 a2→a1 升变为后。提醒人工替换。"""
    pawn_from = uci_to_row_col("a2")   # (6, 7)
    pawn_to   = uci_to_row_col("a1")   # (7, 7)

    rospy.loginfo("=== 兵升变: 黑兵 a2→a1 ===")

    s_row, s_col = pawn_from
    d_row, d_col = pawn_to
    s_x, s_y, _ = to_base_point(s_col, s_row, cam_corners, H, board_z)
    d_x, d_y, _ = to_base_point(d_col, d_row, cam_corners, H, board_z)
    pick_z = board_z + compute_z(s_row, s_col, True, True)
    place_z = board_z + compute_z(d_row, d_col, True, False)
    px, py = compute_xy(s_x, s_y, s_row, s_col, True)
    plx, ply = compute_xy(d_x, d_y, d_row, d_col, False)
    orient_pick = compute_gripper_orient(px, py)
    orient_place = compute_gripper_orient(plx, ply)

    rospy.loginfo(f"  兵: ({s_row},{s_col})→({d_row},{d_col})")
    if not do_pick(arm, px, py, pick_z, orient_pick,
                   gripper_close=GRIPPER_CLOSE_VALUE):
        rospy.logerr("抓取兵失败")
        return
    if not do_place(arm, plx, ply, place_z, orient_place):
        rospy.logerr("放置兵失败")
        return

    rospy.logwarn(">>> 兵已放到 a1！请手动将兵替换为「后」。<<<")
    rospy.loginfo("兵升变完成。")


# ============================================================
# ArUco 采集
# ============================================================

class ArucoCollector:
    def __init__(self, timeout=15.0):
        self.corners = None
        self.received = False
        self.sub = rospy.Subscriber("/chessboard_corners_3d", ChessboardCorners, self._cb)
        rospy.loginfo("等待 ArUco 角点...")
        start = time.time()
        while not rospy.is_shutdown() and not self.received:
            if time.time() - start > timeout:
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
        rospy.loginfo(f"收到 ArUco 角点:\n{np.round(self.corners, 3)}")


# ============================================================
# 主函数
# ============================================================

def main():
    rospy.init_node('test_special_moves', anonymous=True)

    print("\n特殊走法机械臂实测")
    print("  1 = 王车易位 (黑方短易位)")
    print("  2 = 过路兵 (黑兵 a4→b3)")
    print("  3 = 兵升变 (黑兵 a2→a1 升后)")

    try:
        choice = input("\n选择 (1-3): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n退出。")
        return

    if choice not in ('1', '2', '3'):
        print("无效选择。")
        return

    # ArUco
    collector = ArucoCollector(timeout=15.0)
    if not collector.received:
        rospy.logerr("未收到 ArUco 角点，请先启动 board_detector.py。")
        return

    bc = BOARD_CORNERS_BASE
    base_corners = np.array([
        [bc["top_left"]["x"], bc["top_left"]["y"]],
        [bc["top_right"]["x"], bc["top_right"]["y"]],
        [bc["bottom_left"]["x"], bc["bottom_left"]["y"]],
        [bc["bottom_right"]["x"], bc["bottom_right"]["y"]],
    ])
    H = calibrate_homography(collector.corners, base_corners)
    board_z = bc["top_left"]["z"]
    rospy.loginfo(f"单应性矩阵:\n{np.round(H, 4)}")

    # 机械臂
    arm = KinovaArmController()
    if not arm.is_init_success:
        rospy.logerr("机械臂初始化失败。")
        return
    if not arm.activate():
        rospy.logerr("机械臂激活失败。")
        return

    try:
        if choice == '1':
            run_castling(arm, collector.corners, H, board_z)
        elif choice == '2':
            run_en_passant(arm, collector.corners, H, board_z)
        else:
            run_promotion(arm, collector.corners, H, board_z)
    except Exception as e:
        rospy.logerr(f"执行出错: {e}")
        import traceback
        traceback.print_exc()

    # 归位
    rospy.loginfo("归位...")
    arm.move_to_cartesian_pose(
        POST_CALIB_HOME["x"], POST_CALIB_HOME["y"], POST_CALIB_HOME["z"],
        POST_CALIB_HOME["rx"], POST_CALIB_HOME["ry"], POST_CALIB_HOME["rz"])
    rospy.loginfo("测试完成。")


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
