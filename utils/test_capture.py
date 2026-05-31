#!/usr/bin/env python3
"""
吃子逻辑独立测试脚本。

用法:
    rosrun chess_robot_arm test_capture.py

需要先启动 board_detector.py 提供 ArUco 角点。

启动后输入两个 UCI 格子:
    被吃子的格子 (captured): 对方棋子所在的格子
    吃子棋子的起点 (from):    我方棋子当前所在格子
    吃子棋子的终点 (to):      我方棋子吃子后落定的格子 (等于 captured 的格子)

例如: 白后 e5 吃黑兵 a1
    captured = a1
    from_sq   = e5
    to_sq     = a1

动作序列:
  1. 被吃棋子: 抓取(captured) → 放置到弃子区 (普通逻辑)
  2. 吃子棋子: 抓取(from_sq)  → 放置到 to_sq   (若任一为远点则用远点中转站)
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
    COL_PICK_OFFSETS, COL_PLACE_OFFSETS, ROW_Z_OFFSET, ROW_PLACE_Z_OFFSET,
    GRIPPER_TILT_THRESHOLD, GRIPPER_MAX_JOINT5_DEG, GRIPPER_FIXED_YAW_DEG,
    FAR_TRANSIT, FAR_PICK_Z_OFFSET, FAR_PLACE_Z_OFFSET,
    FAR_GRIPPER_CLOSE, FAR_ROW_Z_OFFSET, SPECIAL_Z_OVERRIDE, SPECIAL_XY_PICK_OVERRIDE, SPECIAL_XY_PLACE_OVERRIDE, FAR_CELLS,
    GRIPPER_OPEN_VALUE, GRIPPER_CLOSE_VALUE,
    PAWN_KNIGHT_GRIPPER_CLOSE,
    GARBAGE_POINT, GARBAGE_STEP,
    DEFAULT_GRIPPER_ORIENTATION_DEG,
)

GRIPPER_RX, GRIPPER_RY, GRIPPER_RZ = DEFAULT_GRIPPER_ORIENTATION_DEG


# ============================================================
# 坐标计算
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


# ============================================================
# 姿态计算
# ============================================================

def compute_gripper_orient(x, y, row=None, col=None):
    """判断远点: 仅判断当前 (row,col) 是否在 FAR_CELLS 中。"""
    dist = math.hypot(x, y)
    is_far = (row is not None and col is not None
              and (row, col) in FAR_CELLS)
    if is_far:
        rospy.loginfo(f"  远点 ({x:.3f},{y:.3f}) row={row} col={col}, 远点处理")
        orient = np.array([FAR_TRANSIT["rx"], FAR_TRANSIT["ry"], FAR_TRANSIT["rz"]])
        return orient, True

    if dist <= GRIPPER_TILT_THRESHOLD:
        tilt = 0.0
    else:
        tilt = min((dist - GRIPPER_TILT_THRESHOLD) / 0.25 * GRIPPER_MAX_JOINT5_DEG,
                   GRIPPER_MAX_JOINT5_DEG)
    return np.array([0.0, 180.0 - tilt, GRIPPER_FIXED_YAW_DEG]), False


# ============================================================
# 抓取/放置动作序列
# ============================================================

def _move(arm, x, y, z, orient):
    rx, ry, rz = orient
    return arm.move_to_cartesian_pose(x, y, z, rx, ry, rz)


def do_pick_and_place(arm, src_uci, dst_uci, cam_corners, H, board_z,
                       is_garbage_place=False,
                       place_row=None, place_col=None,
                       piece_type=""):
    """
    执行一次完整的抓取-放置。

    参数:
        src_uci:          抓取源格子 (UCI)
        dst_uci:          放置目标格子 (UCI)，is_garbage_place=True 时忽略
        is_garbage_place: 放置目标是弃子区
        place_row/place_col: 实际放置格子的矩阵坐标（用于远点判断）。
                             当 is_garbage_place=True 时，用于远点中转站判断。
    """
    drx, dry, drz = GRIPPER_RX, GRIPPER_RY, GRIPPER_RZ

    # --- 抓取点计算 ---
    s_row, s_col = uci_to_row_col(src_uci)
    s_x, s_y, s_z = to_base_point(s_col, s_row, cam_corners, H, board_z)

    pick_is_far = (s_row, s_col) in FAR_CELLS
    pick_row_off = ROW_PICK_OFFSETS.get(s_row, {"x": 0.0, "y": 0.0})
    pick_col_off = COL_PICK_OFFSETS.get(s_col, {"x": 0.0, "y": 0.0})
    sp_xy_pick = SPECIAL_XY_PICK_OVERRIDE.get((s_row, s_col), {"x": 0.0, "y": 0.0})

    pick_x = s_x + PICK_XY_OFFSET["x"] + pick_row_off["x"] + pick_col_off["x"] + sp_xy_pick["x"]
    pick_y = s_y + PICK_XY_OFFSET["y"] + pick_row_off["y"] + pick_col_off["y"] + sp_xy_pick["y"]

    sp_z_pick = SPECIAL_Z_OVERRIDE.get((s_row, s_col))
    if sp_z_pick is not None:
        pick_z = s_z + sp_z_pick
    else:
        pick_z = s_z + ROW_Z_OFFSET.get(s_row, PICK_Z_OFFSET) \
                 + (FAR_PICK_Z_OFFSET if pick_is_far else 0.0) \
                 + (FAR_ROW_Z_OFFSET.get(s_row, 0.0) if pick_is_far else 0.0)

    # --- 放置点计算 ---
    if is_garbage_place:
        d_x, d_y = GARBAGE_POINT["x"], GARBAGE_POINT["y"]
        d_z = GARBAGE_POINT["z"]
        place_is_far = False
        place_x, place_y = d_x, d_y
        place_z = d_z
    else:
        d_row, d_col = uci_to_row_col(dst_uci)
        place_row, place_col = d_row, d_col  # 用实际放置格子覆盖
        d_x, d_y, d_z = to_base_point(d_col, d_row, cam_corners, H, board_z)
        place_is_far = (d_row, d_col) in FAR_CELLS
        place_row_off = ROW_PLACE_OFFSETS.get(d_row, {"x": 0.0, "y": 0.0})
        place_col_off = COL_PLACE_OFFSETS.get(d_col, {"x": 0.0, "y": 0.0})
        sp_xy_place = SPECIAL_XY_PLACE_OVERRIDE.get((d_row, d_col), {"x": 0.0, "y": 0.0})

        place_x = d_x + PLACE_XY_OFFSET["x"] + place_row_off["x"] + place_col_off["x"] + sp_xy_place["x"]
        place_y = d_y + PLACE_XY_OFFSET["y"] + place_row_off["y"] + place_col_off["y"] + sp_xy_place["y"]

        sp_z_place = SPECIAL_Z_OVERRIDE.get((d_row, d_col))
        if sp_z_place is not None:
            place_z = d_z + sp_z_place
        else:
            place_z = d_z + ROW_Z_OFFSET.get(d_row, PLACE_Z_OFFSET) \
                      + ROW_PLACE_Z_OFFSET.get(d_row, 0.0) \
                      + (FAR_PLACE_Z_OFFSET if place_is_far else 0.0) \
                      + (FAR_ROW_Z_OFFSET.get(d_row, 0.0) if place_is_far else 0.0)

    # --- 姿态（只看各自的操作点） ---
    pick_orient, pick_free = compute_gripper_orient(pick_x, pick_y, s_row, s_col)
    place_orient, place_free = compute_gripper_orient(place_x, place_y, place_row, place_col)

    # --- 日志 ---
    rospy.loginfo(f"抓取 {src_uci}: matrix({s_row},{s_col}) → base({s_x:.3f},{s_y:.3f},{s_z:.3f})")
    rospy.loginfo(f"  +XY偏移 → ({pick_x:.3f},{pick_y:.3f},{pick_z:.3f})  远点={pick_is_far}")
    if is_garbage_place:
        rospy.loginfo(f"放置 弃子区: ({place_x:.3f},{place_y:.3f},{place_z:.3f})")
    else:
        rospy.loginfo(f"放置 {dst_uci}: matrix({place_row},{place_col}) → base({d_x:.3f},{d_y:.3f},{d_z:.3f})")
        rospy.loginfo(f"  +XY偏移 → ({place_x:.3f},{place_y:.3f},{place_z:.3f})  远点={place_is_far}")

    # ================================================================
    # 步骤 0: 移动到中转点（只看抓取点是否为远点，与单次抓放逻辑一致）
    # ================================================================
    if pick_is_far:
        transit = FAR_TRANSIT
        rospy.loginfo(f"步骤0: 移动到远点中转站 "
                      f"({transit['x']:.3f}, {transit['y']:.3f}, {transit['z']:.3f})")
        if not arm.move_to_cartesian_pose(transit["x"], transit["y"], transit["z"],
                                           transit["rx"], transit["ry"], transit["rz"]):
            rospy.logerr("移动到远点中转站失败。")
            return False
    else:
        rospy.loginfo(f"步骤0: 移动到安全中转点 "
                      f"({PICK_TRANSIT['x']:.3f}, {PICK_TRANSIT['y']:.3f}, {PICK_TRANSIT['z']:.3f})")
        if not arm.move_to_cartesian_pose(PICK_TRANSIT["x"], PICK_TRANSIT["y"],
                                           PICK_TRANSIT["z"], drx, dry, drz):
            rospy.logerr("移动到中转点失败。")
            return False

    # --- 步骤 1: 移动到抓取点上方 ---
    pick_above_z = max(pick_z + PRE_ACTION_Z_LIFT, MIN_APPROACH_Z)
    rospy.loginfo(f"步骤1: 移动到抓取点上方 ({pick_x:.3f}, {pick_y:.3f}, {pick_above_z:.3f})")
    if not _move(arm, pick_x, pick_y, pick_above_z, pick_orient):
        rospy.logerr("移动到抓取点上方失败。")
        return False

    # --- 步骤 2: 张开夹爪 ---
    rospy.loginfo(f"步骤2: 张开夹爪 ({GRIPPER_OPEN_VALUE*100:.0f}%)")
    arm.move_gripper(GRIPPER_OPEN_VALUE)
    rospy.sleep(0.5)

    # --- 步骤 3: 下降到抓取位置 ---
    rospy.loginfo(f"步骤3: 下降到抓取位置 ({pick_x:.3f}, {pick_y:.3f}, {pick_z:.3f})")
    if not _move(arm, pick_x, pick_y, pick_z, pick_orient):
        rospy.logerr("下降到抓取位置失败。")
        return False
    rospy.sleep(0.3)

    # --- 步骤 4: 闭合夹爪 ---
    # 优先级: 兵/马专用 > 远点专用 > 默认
    if piece_type and piece_type.lower() in ('p', 'n'):
        close_val = PAWN_KNIGHT_GRIPPER_CLOSE
    elif pick_free:
        close_val = FAR_GRIPPER_CLOSE
    else:
        close_val = GRIPPER_CLOSE_VALUE
    rospy.loginfo(f"步骤4: 闭合夹爪 ({close_val*100:.0f}%)")
    arm.move_gripper(close_val)
    rospy.sleep(1.0)

    # --- 步骤 5: 抬升 ---
    rospy.loginfo(f"步骤5: 抬升到抓取点上方 ({pick_x:.3f}, {pick_y:.3f}, {pick_above_z:.3f})")
    if not _move(arm, pick_x, pick_y, pick_above_z, pick_orient):
        rospy.logerr("抬升失败。")
        return False

    # --- 步骤 6: 移动到放置点上方 ---
    place_above_z = max(place_z + PRE_ACTION_Z_LIFT, MIN_APPROACH_Z)
    rospy.loginfo(f"步骤6: 移动到放置点上方 ({place_x:.3f}, {place_y:.3f}, {place_above_z:.3f})")
    if not _move(arm, place_x, place_y, place_above_z, place_orient):
        rospy.logerr("移动到放置点上方失败。")
        return False

    # --- 步骤 7: 下降到放置位置 ---
    rospy.loginfo(f"步骤7: 下降到放置位置 ({place_x:.3f}, {place_y:.3f}, {place_z:.3f})")
    if not _move(arm, place_x, place_y, place_z, place_orient):
        rospy.logerr("下降到放置位置失败。")
        return False
    rospy.sleep(0.3)

    # --- 步骤 8: 张开夹爪释放棋子 ---
    rospy.loginfo(f"步骤8: 释放棋子 ({GRIPPER_OPEN_VALUE*100:.0f}%)")
    arm.move_gripper(GRIPPER_OPEN_VALUE)
    rospy.sleep(1.0)

    # --- 步骤 9: 抬升 ---
    rospy.loginfo(f"步骤9: 抬升 ({place_x:.3f}, {place_y:.3f}, {place_above_z:.3f})")
    if not _move(arm, place_x, place_y, place_above_z, place_orient):
        rospy.logerr("抬升失败。")
        return False

    rospy.loginfo("抓取-放置循环完成。")
    return True


# ============================================================
# 吃子测试主逻辑
# ============================================================

def run_capture_test(arm, cam_corners, H, board_z,
                     captured_uci, captured_piece,
                     from_uci, from_piece, to_uci):
    """
    执行吃子测试:
      1. 被吃子 captured_uci → 弃子区 (普通逻辑，带棋子种类)
      2. 吃子棋子 from_uci → to_uci (远点中转站考虑放置格 + 兵/马专用闭合度)
    """
    captured_row, captured_col = uci_to_row_col(captured_uci)
    from_row, from_col = uci_to_row_col(from_uci)
    to_row, to_col = uci_to_row_col(to_uci)

    rospy.loginfo("=" * 60)
    rospy.loginfo(f"吃子测试: 被吃子 {captured_uci}({captured_row},{captured_col}) "
                  f"种类={captured_piece} → 弃子区")
    rospy.loginfo(f"         吃子棋子 {from_uci}({from_row},{from_col}) "
                  f"种类={from_piece} → {to_uci}({to_row},{to_col})")
    rospy.loginfo("=" * 60)

    # ---- 第一步: 被吃子 → 弃子区 (普通逻辑) ----
    rospy.loginfo(">>> 第一步: 将被吃子移到弃子区 <<<")
    if not do_pick_and_place(arm, captured_uci, "", cam_corners, H, board_z,
                              is_garbage_place=True,
                              piece_type=captured_piece):
        rospy.logerr("第一步（被吃子→弃子区）失败。")
        return False

    rospy.sleep(1.0)

    # ---- 第二步: 吃子棋子 → 目标格子 ----
    rospy.loginfo(">>> 第二步: 吃子棋子移动到目标格子 <<<")
    if not do_pick_and_place(arm, from_uci, to_uci, cam_corners, H, board_z,
                              is_garbage_place=False, piece_type=from_piece):
        rospy.logerr("第二步（吃子棋子→目标格）失败。")
        return False

    rospy.loginfo("吃子测试完成。")
    return True


# ============================================================
# ArUco 角点采集
# ============================================================

class ArucoCollector:
    def __init__(self, timeout=15.0):
        self.corners = None
        self.received = False
        self.sub = rospy.Subscriber("/chessboard_corners_3d", ChessboardCorners, self._cb)
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
    rospy.init_node('test_capture', anonymous=True)

    rospy.loginfo("=" * 60)
    rospy.loginfo("吃子测试脚本")
    rospy.loginfo("  用法: 输入 UCI 格子和棋子种类")
    rospy.loginfo("    captured : 被吃子的格子 (如 a1)")
    rospy.loginfo("    captured_piece: 被吃子种类 (p/n/b/r/q/k/P/N/B/R/Q/K, 如 p)")
    rospy.loginfo("    from_sq  : 吃子棋子的起点 (如 e5)")
    rospy.loginfo("    from_piece: 吃子棋子种类 (如 Q)")
    rospy.loginfo("    to_sq    : 吃子棋子的终点 (如 a1，通常=captured)")
    rospy.loginfo("  (兵/马 p/n/P/N 将使用专用闭合度 0.95)")
    rospy.loginfo("  输入 'quit' 退出")
    rospy.loginfo("=" * 60)

    # --- ArUco 角点采集 ---
    collector = ArucoCollector(timeout=30.0)
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

    # --- 初始化机械臂 ---
    arm = KinovaArmController()
    if not arm.is_init_success:
        rospy.logerr("机械臂控制器初始化失败。")
        return
    if not arm.activate():
        rospy.logerr("机械臂激活失败。")
        return

    # --- 交互循环 ---
    while not rospy.is_shutdown():
        print("\n" + "-" * 40)
        try:
            captured = input("被吃子的格子 (captured, 如 a1): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break
        if captured == 'quit':
            break
        if not captured or len(captured) < 2:
            print("  格式错误，请输入类似 a1 的 UCI 格子名。")
            continue

        try:
            captured_piece = input("被吃子的种类 (captured_piece, p/n/b/r/q/k/P/N/B/R/Q/K): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break
        if captured_piece == 'quit':
            break
        if captured_piece not in 'pnbrqkPNBRQK' or len(captured_piece) != 1:
            print("  格式错误，请输入单个 FEN 棋子符号 (如 p, Q, N)。")
            continue

        try:
            from_sq = input("吃子棋子的起点 (from_sq, 如 e5): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break
        if from_sq == 'quit':
            break
        if not from_sq or len(from_sq) < 2:
            print("  格式错误。")
            continue

        try:
            from_piece = input("吃子棋子的种类 (from_piece, p/n/b/r/q/k/P/N/B/R/Q/K): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break
        if from_piece == 'quit':
            break
        if from_piece not in 'pnbrqkPNBRQK' or len(from_piece) != 1:
            print("  格式错误，请输入单个 FEN 棋子符号。")
            continue

        try:
            to_sq = input("吃子棋子的终点 (to_sq, 如 a1): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break
        if to_sq == 'quit':
            break
        if not to_sq or len(to_sq) < 2:
            print("  格式错误。")
            continue

        # 快速校验
        for label, sq in [("captured", captured), ("from_sq", from_sq), ("to_sq", to_sq)]:
            if sq[0] not in 'abcdefgh' or not sq[1:].isdigit():
                print(f"  {label}='{sq}' 不是合法的 UCI 格子名。")
                break
        else:
            try:
                run_capture_test(arm, collector.corners, H, board_z,
                                 captured, captured_piece,
                                 from_sq, from_piece, to_sq)
            except Exception as e:
                rospy.logerr(f"执行出错: {e}")
                import traceback
                traceback.print_exc()

            # 归位
            rospy.loginfo("归位中...")
            arm.move_to_cartesian_pose(
                POST_CALIB_HOME["x"], POST_CALIB_HOME["y"], POST_CALIB_HOME["z"],
                POST_CALIB_HOME["rx"], POST_CALIB_HOME["ry"], POST_CALIB_HOME["rz"])

    rospy.loginfo("吃子测试脚本退出。")


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
