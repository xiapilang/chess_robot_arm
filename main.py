#!/usr/bin/env python3
"""
国际象棋对弈主协调节点（终端输入版 —— 基座标系直接定位）。

坐标计算直接使用手动测量的棋盘四角基座标系坐标（BOARD_CORNERS_BASE），
不再依赖 T_ee_camera 手眼标定变换。

流程:
  1. 等待 ArUco 棋盘角点就绪（仅作棋盘在位确认）
  2. 在终端打印初始棋盘摆法
  3. 用户在终端输入当前棋盘状态（UCI/SAN 走法 或 FEN 字符串）
  4. AI (Stockfish) 计算最佳应招
  5. 向机械臂发布基座标系抓取-放置指令
  6. 打印更新后的棋盘摆法，等待用户再次输入，循环往复

机械臂执黑（后手），人类执白（先手）。
"""

import rospy
import sys
import chess
import cv2
import numpy as np
from geometry_msgs.msg import Point
from std_msgs.msg import String
from chess_robot_arm.msg import PickAndPlaceGoalInCamera, ChessboardCorners

from chess_robot_arm.chess_engine.ai_player import (
    ai_move_from_board, uci_to_matrix_coords, close_session_engine,
)
from chess_robot_arm.chess_engine.game_state import GameState
from chess_robot_arm.utils.constants import (
    BOARD_ROWS, BOARD_COLS, EMPTY_SQUARE, BOARD_CORNERS_BASE,
    ROW_PICK_OFFSETS, ROW_PLACE_OFFSETS,
    COL_PICK_OFFSETS, COL_PLACE_OFFSETS, ROW_Z_OFFSET,
    GARBAGE_POINT, GARBAGE_STEP,
)


class TerminalChessOrchestrator:
    """终端输入驱动的国际象棋对弈协调器（基座标系直接定位）。"""

    def __init__(self):
        rospy.init_node('chess_ai_orchestrator', anonymous=False)

        # --- 对弈状态 ---
        self.game = GameState(robot_plays_as=chess.BLACK)
        self.board_matrix = self.game.board_to_matrix()

        # --- 棋盘四角（相机系，由 ArUco 检测） ---
        self.cam_corners = None  # [tl, tr, bl, br] each (x, y)

        # --- 棋盘四角（基座标系，手动测量） ---
        bc = BOARD_CORNERS_BASE
        self.base_corners = np.array([
            [bc["top_left"]["x"],     bc["top_left"]["y"]],
            [bc["top_right"]["x"],    bc["top_right"]["y"]],
            [bc["bottom_left"]["x"],  bc["bottom_left"]["y"]],
            [bc["bottom_right"]["x"], bc["bottom_right"]["y"]],
        ])
        self.board_z = bc["top_left"]["z"]  # 棋盘表面高度

        # --- 单应性变换: 相机 XY → 基座 XY（ArUco 检测后标定） ---
        self.H = None  # 3x3 单应性矩阵

        # --- 弃子区 ---
        self.garbage_point = None
        self.eat_count = 0
        self.garbage_offset = None
        self._init_garbage_zone()

        # --- 机械臂状态 ---
        self.arm_status = "0"

        # --- ArUco 在位确认 ---
        self.corners_received = False

        # --- 参数 ---
        self.stockfish_path = rospy.get_param("~stockfish_path", "/usr/games/stockfish")
        self.think_time = rospy.get_param("~think_time", 3.0)

        # --- 订阅者 ---
        self.corners_sub = rospy.Subscriber(
            "/chessboard_corners_3d", ChessboardCorners, self._corners_callback)
        self.arm_status_sub = rospy.Subscriber(
            "/my_gen3_lite/arm_status", String, self._arm_status_callback)

        # --- 发布者 ---
        self.goal_pub = rospy.Publisher(
            "/kinova_pick_place/goal_in_camera",
            PickAndPlaceGoalInCamera, queue_size=10)
        self.eat_status_pub = rospy.Publisher(
            "/ai_eat_status", String, queue_size=1, latch=True)

        self.rate = rospy.Rate(10)
        rospy.loginfo("终端输入版国际象棋 AI 协调器已初始化（基座标系直接定位）。")

    # ------------------------------------------------------------------
    # 弃子区
    # ------------------------------------------------------------------

    def _init_garbage_zone(self):
        """弃子区参考位置（从 constants 读取）。"""
        self.garbage_point = Point(
            x=GARBAGE_POINT["x"], y=GARBAGE_POINT["y"], z=GARBAGE_POINT["z"])
        self.garbage_offset = Point(
            x=GARBAGE_STEP["x"], y=GARBAGE_STEP["y"], z=GARBAGE_STEP["z"])

    # ------------------------------------------------------------------
    # ROS 回调
    # ------------------------------------------------------------------

    def _arm_status_callback(self, msg):
        self.arm_status = msg.data

    def _corners_callback(self, msg):
        """ArUco 角点回调：存储相机系坐标并标定仿射变换。"""
        if self.corners_received:
            return

        # 存储相机系四角 XY（ArUco 定位精确，Z 不可靠）
        self.cam_corners = np.array([
            [msg.top_left.x,     msg.top_left.y],
            [msg.top_right.x,    msg.top_right.y],
            [msg.bottom_left.x,  msg.bottom_left.y],
            [msg.bottom_right.x, msg.bottom_right.y],
        ])

        # 标定仿射变换：相机 XY → 基座 XY
        self._calibrate_homography()

        self.corners_received = True
        rospy.loginfo("ArUco 棋盘标定完成。")
        rospy.loginfo(f"  相机四角: {np.round(self.cam_corners, 3).tolist()}")
        rospy.loginfo(f"  基座四角: {np.round(self.base_corners, 3).tolist()}")
        self.corners_sub.unregister()

    def _calibrate_homography(self):
        """用四对点标定单应性变换（相机 XY → 基座 XY），可处理透视畸变。"""
        src = self.cam_corners.reshape(4, 1, 2).astype(np.float32)
        dst = self.base_corners.reshape(4, 1, 2).astype(np.float32)
        self.H, _ = cv2.findHomography(src, dst)
        rospy.loginfo(f"  单应性矩阵:\n{np.round(self.H, 4)}")

    # ------------------------------------------------------------------
    # 坐标变换（基座标系双线性插值）
    # ------------------------------------------------------------------

    def _matrix_to_base_point(self, col, row):
        """ArUco 相机帧双线性插值 + 单应性变换 → 基座标系 3D 点。"""
        u = col / float(BOARD_COLS - 1)
        v = row / float(BOARD_ROWS - 1)

        c_tl, c_tr = self.cam_corners[0], self.cam_corners[1]
        c_bl, c_br = self.cam_corners[2], self.cam_corners[3]
        top = c_tl + u * (c_tr - c_tl)
        bot = c_bl + u * (c_br - c_bl)
        cam_xy = top + v * (bot - top)

        pts = np.array([[[cam_xy[0], cam_xy[1]]]], dtype=np.float32)
        base_xy = cv2.perspectiveTransform(pts, self.H)[0][0]
        return Point(x=base_xy[0], y=base_xy[1], z=self.board_z)

    def _pick_point(self, col, row):
        """抓取点：叠加逐排 + 逐列偏移（含 Z）。"""
        pt = self._matrix_to_base_point(col, row)
        row_off = ROW_PICK_OFFSETS.get(row, {"x": 0.0, "y": 0.0})
        col_off = COL_PICK_OFFSETS.get(col, {"x": 0.0, "y": 0.0})
        z = pt.z + ROW_Z_OFFSET.get(row, 0.0)
        return Point(x=pt.x + row_off["x"] + col_off["x"],
                     y=pt.y + row_off["y"] + col_off["y"], z=z)

    def _place_point(self, col, row):
        """放置点：叠加逐排 + 逐列偏移（含 Z）。"""
        pt = self._matrix_to_base_point(col, row)
        row_off = ROW_PLACE_OFFSETS.get(row, {"x": 0.0, "y": 0.0})
        col_off = COL_PLACE_OFFSETS.get(col, {"x": 0.0, "y": 0.0})
        z = pt.z + ROW_Z_OFFSET.get(row, 0.0)
        return Point(x=pt.x + row_off["x"] + col_off["x"],
                     y=pt.y + row_off["y"] + col_off["y"], z=z)

    def _get_garbage_place(self):
        """计算弃子区中下一个棋子的放置位置。"""
        gp = Point()
        gp.x = self.garbage_point.x + self.garbage_offset.x * self.eat_count
        gp.y = self.garbage_point.y + self.garbage_offset.y * self.eat_count
        gp.z = self.garbage_point.z + self.garbage_offset.z * self.eat_count
        return gp

    # ------------------------------------------------------------------
    # 机械臂指令
    # ------------------------------------------------------------------

    def _wait_for_arm_idle(self, timeout=3000):
        t = 0
        while self.arm_status != '0' and t < timeout:
            self.rate.sleep()
            t += 1
        return t < timeout

    def _execute_robot_move(self, from_row, from_col, to_row, to_col,
                            is_capture, is_en_passant, move_uci):
        """向运动规划器发布基座标系抓取-放置指令。"""

        # --- 吃子：先将对方棋子移到弃子区 ---
        if is_capture:
            rospy.loginfo("检测到吃子，先移除对方棋子。")
            if is_en_passant:
                captured_col, captured_row = to_col, from_row
            else:
                captured_col, captured_row = to_col, to_row

            self.eat_count += 1
            msg_eat = PickAndPlaceGoalInCamera()
            msg_eat.object_id_at_pick = f"{captured_row},{captured_col}"
            msg_eat.pick_position_in_camera = self._pick_point(
                captured_col, captured_row)
            msg_eat.place_position_in_camera = self._get_garbage_place()
            self.goal_pub.publish(msg_eat)
            self.eat_status_pub.publish(String("BUSY"))
            rospy.sleep(0.5)

        # --- 主动走法：从起点抓取，放到目标格 ---
        if not self._wait_for_arm_idle():
            rospy.logwarn("等待机械臂 IDLE 超时。")
            return False

        msg_move = PickAndPlaceGoalInCamera()
        msg_move.object_id_at_pick = f"{from_row},{from_col}"
        msg_move.pick_position_in_camera = self._pick_point(
            from_col, from_row)
        msg_move.target_location_id_at_place = f"{to_row},{to_col}"
        msg_move.place_position_in_camera = self._place_point(
            to_col, to_row)
        self.goal_pub.publish(msg_move)
        self.eat_status_pub.publish(String("IDLE"))

        # --- 王车易位：额外移动车 ---
        piece = self.current_board[from_row][from_col]
        if piece and piece != EMPTY_SQUARE and piece.lower() == 'k' and abs(to_col - from_col) == 2:
            rospy.loginfo("王车易位！移动车。")
            rospy.sleep(2.0)
            if not self._wait_for_arm_idle():
                return False
            rook_row = from_row
            rook_from, rook_to = (0, 2) if to_col < from_col else (7, 5)  # 列翻转: col减小=短易位
            msg_rook = PickAndPlaceGoalInCamera()
            msg_rook.object_id_at_pick = f"{rook_row},{rook_from}"
            msg_rook.pick_position_in_camera = self._pick_point(
                rook_from, rook_row)
            msg_rook.target_location_id_at_place = f"{rook_row},{rook_to}"
            msg_rook.place_position_in_camera = self._place_point(
                rook_to, rook_row)
            self.goal_pub.publish(msg_rook)

        # --- 兵升变提醒 ---
        if len(move_uci) == 5:
            piece_name = {'q': '后', 'r': '车', 'b': '象', 'n': '马'}.get(
                move_uci[4], move_uci[4])
            rospy.logwarn(f"兵升变！请在目标格手动替换为{piece_name}。")
            self.eat_status_pub.publish(String("PROMOTION"))
            rospy.sleep(1.0)

        return True

    # ------------------------------------------------------------------
    # 终端交互
    # ------------------------------------------------------------------

    def _print_board(self):
        """在终端打印当前棋盘摆法。"""
        print("\n" + "=" * 40)
        print("  当前棋盘摆法:")
        print("  " + "-" * 17)
        for row in range(BOARD_ROWS):
            rank = 8 - row
            cells = []
            for col in range(BOARD_COLS):
                p = self.board_matrix[row][col]
                cells.append(f" {p}" if p not in (EMPTY_SQUARE, '0') else " .")
            print(f"{rank} |" + "".join(cells) + " |")
        print("  " + "-" * 17)
        print("    a b c d e f g h")
        print(f"  FEN: {self.game.get_fen()}")
        print("=" * 40 + "\n")

    def _get_user_input(self):
        """
        获取用户在终端的输入。

        支持格式:
          - UCI 走法:   e2e4, g1f3, e7e8q
          - SAN 走法:   e4, Nf3, O-O, exd5
          - FEN 字符串: fen:rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1
          - 特殊命令:   quit 退出
        """
        print("请输入你的走法 (UCI 如 e2e4, SAN 如 e4, 或 fen:<完整FEN>):")
        while not rospy.is_shutdown():
            try:
                user_input = sys.stdin.readline().strip()
            except (EOFError, KeyboardInterrupt):
                return None

            if not user_input:
                continue
            if user_input.lower() == 'quit':
                return None

            # --- FEN 整盘输入 ---
            if user_input.lower().startswith('fen:'):
                fen = user_input[4:].strip()
                try:
                    board = chess.Board(fen)
                    self.game = GameState(robot_plays_as=chess.BLACK)
                    self.game.board = board
                    self.board_matrix = self.game.board_to_matrix()
                    return True
                except ValueError as e:
                    print(f"  FEN 无效: {e}")
                    continue

            # --- 尝试 SAN 格式 ---
            try:
                move = self.game.board.parse_san(user_input)
                if move in self.game.board.legal_moves:
                    self.game.board.push(move)
                    self.board_matrix = self.game.board_to_matrix()
                    return True
                else:
                    print(f"  非法走法: {user_input}")
                    continue
            except ValueError:
                pass

            # --- 尝试 UCI 格式 ---
            try:
                move = chess.Move.from_uci(user_input)
                if move in self.game.board.legal_moves:
                    self.game.board.push(move)
                    self.board_matrix = self.game.board_to_matrix()
                    return True
                else:
                    print(f"  非法走法: {user_input}")
                    continue
            except ValueError:
                print(f"  无法识别输入: '{user_input}'。请输入 UCI/SAN 走法或 fen:<FEN>")
                continue

        return None

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def run(self):
        rospy.loginfo("等待棋盘角点 (ArUco)...")
        while not rospy.is_shutdown() and not self.corners_received:
            self.rate.sleep()

        rospy.loginfo("棋盘角点已就绪，开始对弈。")

        # 打印初始棋盘
        self._print_board()
        print("机械臂执黑（后手），你执白（先手）。")
        print("在实体棋盘上走完一步后，在此终端输入你的走法。\n")

        try:
            while not rospy.is_shutdown():
                # --- 1. 获取用户走法 ---
                result = self._get_user_input()
                if result is None:
                    rospy.loginfo("退出对弈。")
                    break

                self._print_board()

                # --- 2. 检查游戏是否结束 ---
                if self.game.is_game_over():
                    print(f"对局结束！结果: {self.game.result()}")
                    break

                # --- 3. AI 计算走法 ---
                print("AI 思考中...")
                fen = self.game.get_fen()
                ai_result = ai_move_from_board(
                    fen, ai_color=chess.BLACK,
                    think_time=self.think_time,
                    stockfish_path=self.stockfish_path)

                move_uci, from_sq, to_sq, is_capture, is_en_passant, is_castling = ai_result

                if move_uci is None:
                    print("AI 无法找到走法，对局可能已结束。")
                    break

                from_row, from_col = uci_to_matrix_coords(from_sq)
                to_row, to_col = uci_to_matrix_coords(to_sq)

                print(f"AI 走法: {from_sq} -> {to_sq}"
                      f"  吃子={is_capture}  过路兵={is_en_passant}  易位={is_castling}")

                # --- 4. 机械臂执行 ---
                if not self._wait_for_arm_idle():
                    rospy.logwarn("等待机械臂就绪超时，继续等待...")
                    self._wait_for_arm_idle(300)

                self._execute_robot_move(from_row, from_col, to_row, to_col,
                                         is_capture, is_en_passant, move_uci)

                # --- 5. 更新内部状态 ---
                self.game.push_uci(move_uci)
                self.board_matrix = self.game.board_to_matrix()

                # --- 6. 打印新棋盘 ---
                self._print_board()

                if self.game.is_game_over():
                    print(f"对局结束！结果: {self.game.result()}")
                    break

                print("轮到你了！在实体棋盘走完后输入走法。\n")
        finally:
            close_session_engine()


if __name__ == '__main__':
    try:
        node = TerminalChessOrchestrator()
        node.run()
    except rospy.ROSInterruptException:
        pass
