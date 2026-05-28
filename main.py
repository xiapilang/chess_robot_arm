#!/usr/bin/env python3
"""
国际象棋对弈主协调节点。

这是整个项目的中央 ROS 节点，负责：
  1. 订阅 /chess_board_matrix（来自视觉模块的 FEN 棋盘状态）
  2. 订阅 /chessboard_corners_3d（棋盘四个角点的 3D 位置）
  3. 订阅 /arm_status（机械臂的当前状态）
  4. 调用 python-chess + Stockfish 引擎计算 AI 走法
  5. 向机械臂控制节点发布 PickAndPlaceGoalInCamera 移动指令

改编自 EE368_Project 中的 chess_ai_node.py，适配 8×8 国际象棋。
机械臂执黑（后手），人类执白（先手）。
"""

import rospy
import numpy as np
import chess
from geometry_msgs.msg import Point
from std_msgs.msg import String
from chess_robot_arm.msg import PickAndPlaceGoalInCamera, ChessboardCorners

from chess_robot_arm.chess_engine.ai_player import ai_move_from_board, uci_to_matrix_coords
from chess_robot_arm.chess_engine.game_state import GameState
from chess_robot_arm.utils.constants import BOARD_ROWS, BOARD_COLS, EMPTY_SQUARE


class ChessAIOrchestrator:
    """
    对弈流水线主协调器。

    数据流：视觉 → 棋盘状态 → AI 走法 → 机械臂指令
    """

    def __init__(self):
        rospy.init_node('chess_ai_orchestrator', anonymous=False)

        # --- 对弈状态 ---
        self.game = GameState(robot_plays_as=chess.BLACK)
        self.initial_board = self.game.board_to_matrix()
        self.current_board = None
        self.current_fen = None        # 来自视觉模块的原始 FEN（含过路兵信息）
        self.last_board = None
        self.is_ai_turn = False     # AI 执黑，为后手

        # --- 棋盘几何信息 ---
        self.top_left = None        # 棋盘左上角（相机坐标系）
        self.top_right = None       # 棋盘右上角
        self.bottom_left = None     # 棋盘左下角
        self.bottom_right = None    # 棋盘右下角
        self.corners_received = False

        # --- 被吃子的弃子区 ---
        self.garbage_point = None   # 弃子区参考原点
        self.eat_count = 0          # AI 已吃的棋子数量
        self.garbage_offset = None  # 弃子区中每个棋子的偏移量

        # --- 机械臂状态 ---
        self.arm_status = "0"       # 默认为 IDLE

        # --- Stockfish 引擎配置 ---
        self.stockfish_path = rospy.get_param("~stockfish_path", "/usr/games/stockfish")
        self.think_time = rospy.get_param("~think_time", 3.0)

        # --- 棋盘 Z 坐标（固定高度，避免机械臂撞向桌面） ---
        self.board_z = rospy.get_param("~board_z", 0.38)

        # --- 订阅者 ---
        # 来自视觉模块的棋盘状态（FEN 字符串）
        self.board_sub = rospy.Subscriber(
            "/chess_board_matrix", String, self._board_callback)

        # 棋盘角点的相机坐标系 3D 位置
        self.corners_sub = rospy.Subscriber(
            "/chessboard_corners_3d", ChessboardCorners, self._corners_callback)

        # 机械臂状态
        self.arm_status_sub = rospy.Subscriber(
            "/my_gen3_lite/arm_status", String, self._arm_status_callback)

        # --- 发布者 ---
        # 向机械臂发布抓取/放置目标指令
        self.goal_pub = rospy.Publisher(
            "/kinova_pick_place/goal_in_camera",
            PickAndPlaceGoalInCamera, queue_size=10)

        # 发布吃子状态
        self.eat_status_pub = rospy.Publisher(
            "/ai_eat_status", String, queue_size=1, latch=True)

        # --- 循环频率 ---
        self.rate = rospy.Rate(10)

        rospy.loginfo("国际象棋 AI 协调器已初始化（8×8 国际象棋）。")
        rospy.loginfo(f"Stockfish 引擎: {self.stockfish_path}, 思考时间: {self.think_time}s")

    def _arm_status_callback(self, msg):
        """机械臂状态回调：记录当前状态。"""
        self.arm_status = msg.data

    def _board_callback(self, msg):
        """
        棋盘状态回调：接收来自视觉模块的 FEN 字符串，
        转换为内部矩阵并与上一次状态对比，检测是否有变化。
        """
        try:
            fen = msg.data
            board = chess.Board(fen)
            matrix = board_to_matrix(board)

            if self.current_board is None or not np.array_equal(matrix, self.current_board):
                self.current_board = matrix
                self.current_fen = fen   # 保存原始 FEN，保留过路兵等特殊状态
                rospy.loginfo(f"收到更新的棋盘状态:\n{board}")
                rospy.loginfo("是否等于初始棋盘？%s",
                              np.array_equal(self.current_board, self.initial_board))
        except Exception as e:
            rospy.logerr(f"解析棋盘 FEN 出错: {e}")

    def _corners_callback(self, msg):
        """
        棋盘角点回调：接收四个棋盘角点在相机坐标系下的 3D 坐标，
        同时计算弃子区的参考位置。
        收到第一次消息后取消订阅（角点位置固定不变）。
        """
        self.top_left     = msg.top_left
        self.top_right    = msg.top_right
        self.bottom_left  = msg.bottom_left
        self.bottom_right = msg.bottom_right

        # 计算弃子区参考点（棋盘右侧，通过顶部边缘外推得到）
        self.garbage_point = Point()
        self.garbage_point.x = 2 * self.top_right.x - self.top_left.x
        self.garbage_point.y = 2 * self.top_right.y - self.top_left.y
        self.garbage_point.z = 2 * self.top_right.z - self.top_left.z

        # 弃子区中每个棋子的偏移量（沿棋盘右侧边线堆叠）
        self.garbage_offset = Point()
        self.garbage_offset.x = (self.top_right.x - self.bottom_right.x) / 8
        self.garbage_offset.y = (self.top_right.y - self.bottom_right.y) / 8
        self.garbage_offset.z = (self.top_right.z - self.bottom_right.z) / 8

        if not self.corners_received:
            self.corners_received = True
            rospy.loginfo("已收到棋盘角点坐标（相机坐标系）:")
            rospy.loginfo(f"  左上: ({self.top_left.x:.2f}, {self.top_left.y:.2f}, {self.top_left.z:.2f})")
            rospy.loginfo(f"  右上: ({self.top_right.x:.2f}, {self.top_right.y:.2f}, {self.top_right.z:.2f})")
            rospy.loginfo(f"  左下: ({self.bottom_left.x:.2f}, {self.bottom_left.y:.2f}, {self.bottom_left.z:.2f})")
            rospy.loginfo(f"  右下: ({self.bottom_right.x:.2f}, {self.bottom_right.y:.2f}, {self.bottom_right.z:.2f})")
            self.corners_sub.unregister()
            rospy.loginfo("已取消订阅 /chessboard_corners_3d。")

    def _matrix_to_camera_point(self, col, row):
        """
        将 8×8 棋盘矩阵中的 (col, row) 坐标转换为相机坐标系下的 3D 点。

        使用四个角点的双线性插值来计算棋盘上任意格子的位置。

        矩阵索引约定：
          - row 0 = 棋盘顶部（黑方底线，第 8 横排）
          - row 7 = 棋盘底部（白方底线，第 1 横排）
          - col 0 = 最左侧（a 线），col 7 = 最右侧（h 线）
        """
        u = col / float(BOARD_COLS - 1)    # 水平插值因子 0..1
        v = row / float(BOARD_ROWS - 1)    # 垂直插值因子 0..1

        # 双线性插值：先沿顶部/底部边缘插值，再在垂直方向插值
        top_x = self.top_left.x + u * (self.top_right.x - self.top_left.x)
        top_y = self.top_left.y + u * (self.top_right.y - self.top_left.y)
        top_z = self.top_left.z + u * (self.top_right.z - self.top_left.z)

        bot_x = self.bottom_left.x + u * (self.bottom_right.x - self.bottom_left.x)
        bot_y = self.bottom_left.y + u * (self.bottom_right.y - self.bottom_left.y)
        bot_z = self.bottom_left.z + u * (self.bottom_right.z - self.bottom_left.z)

        x = top_x + v * (bot_x - top_x)
        y = top_y + v * (bot_y - top_y)
        # Z 使用固定安全高度，避免机械臂撞向桌面

        return Point(x=x + 0.015, y=y, z=self.board_z)

    def _get_garbage_place(self):
        """计算当前被吃子应该放置的弃子区位置。"""
        gp = Point()
        gp.x = self.garbage_point.x - self.garbage_offset.x * self.eat_count
        gp.y = self.garbage_point.y - self.garbage_offset.y * self.eat_count
        gp.z = self.garbage_point.z - self.garbage_offset.z * self.eat_count
        return gp

    def run(self):
        """主运行循环：等待棋盘和角点信息就绪后，执行 AI 对弈逻辑。"""
        rospy.loginfo("等待棋盘角点和棋盘状态...")

        while not rospy.is_shutdown():
            # 等待角点数据就绪
            if not self.corners_received:
                rospy.logwarn_throttle(5, "等待棋盘角点信息...")
                self.rate.sleep()
                continue

            # 等待第一帧棋盘状态
            if self.current_board is None:
                rospy.logwarn_throttle(5, "等待第一帧棋盘状态...")
                self.rate.sleep()
                continue

            # --- 轮到 AI 走棋 ---
            if self.is_ai_turn:
                rospy.loginfo("=" * 40)
                rospy.loginfo("AI 正在思考...")

                # 使用视觉模块的原始 FEN（保留过路兵等信息）
                fen = self.current_fen or matrix_to_fen(self.current_board, chess.BLACK)
                rospy.loginfo(f"棋盘 FEN: {fen}")

                # 调用 AI 引擎获取最佳走法
                move_uci, from_sq, to_sq, is_capture, is_en_passant = ai_move_from_board(
                    fen, ai_color=chess.BLACK,
                    think_time=self.think_time,
                    stockfish_path=self.stockfish_path)

                if move_uci is None:
                    rospy.logwarn("AI 未返回走法，对局可能已结束。")
                    self.is_ai_turn = False
                    self.rate.sleep()
                    continue

                from_row, from_col = uci_to_matrix_coords(from_sq)
                to_row, to_col     = uci_to_matrix_coords(to_sq)

                rospy.loginfo(f"AI 走法: {from_sq}({from_col},{from_row}) -> "
                              f"{to_sq}({to_col},{to_row})  吃子={is_capture}")

                # --- 如果是吃子走法：先将对方棋子移到弃子区 ---
                if is_capture:
                    rospy.loginfo("检测到吃子！先将对方棋子移出棋盘。")
                    rospy.sleep(0.5)

                    if self.arm_status != '0':
                        rospy.logwarn("机械臂未就绪，等待...")
                        self.rate.sleep()
                        continue

                    # 过路兵：被吃的兵不在目标格，在 (from_row, to_col)
                    if is_en_passant:
                        captured_col = to_col
                        captured_row = from_row
                        rospy.loginfo("吃过路兵 — 被吃兵位于起点行、终点列")
                    else:
                        captured_col = to_col
                        captured_row = to_row

                    self.eat_count += 1
                    msg_eat = PickAndPlaceGoalInCamera()
                    msg_eat.object_id_at_pick = ""
                    msg_eat.pick_position_in_camera = self._matrix_to_camera_point(captured_col, captured_row)
                    msg_eat.target_location_id_at_place = ""
                    msg_eat.place_position_in_camera = self._get_garbage_place()
                    self.goal_pub.publish(msg_eat)
                    self.eat_status_pub.publish(String("BUSY"))
                    rospy.loginfo(f"已发布吃子指令: 吃掉 ({captured_col},{captured_row}) 处的棋子")

                # --- 主走法：从起点抓取棋子，放置到目标位置 ---
                rospy.sleep(0.5)

                # 等待机械臂变为 IDLE
                timeout = 0
                while self.arm_status != '0' and timeout < 100:
                    self.rate.sleep()
                    timeout += 1

                if timeout >= 100:
                    rospy.logwarn("等待机械臂 IDLE 超时。")
                    continue

                msg_move = PickAndPlaceGoalInCamera()
                msg_move.object_id_at_pick = ""
                msg_move.pick_position_in_camera = self._matrix_to_camera_point(from_col, from_row)
                msg_move.target_location_id_at_place = ""
                msg_move.place_position_in_camera = self._matrix_to_camera_point(to_col, to_row)
                self.goal_pub.publish(msg_move)
                self.eat_status_pub.publish(String("IDLE"))

                rospy.loginfo(f"已发布走法指令: ({from_col},{from_row}) -> ({to_col},{to_row})")

                # --- 处理特殊走法：兵升变 ---
                is_promotion = len(move_uci) == 5
                if is_promotion:
                    promotion_piece = {'q': '后', 'r': '车', 'b': '象', 'n': '马'}.get(move_uci[4], move_uci[4])
                    rospy.logwarn(f"兵升变！兵在 {to_sq} 升变为{promotion_piece}，请手动替换棋子。")
                    self.eat_status_pub.publish(String("PROMOTION"))
                    rospy.sleep(1.0)

                # --- 处理特殊走法：王车易位（额外移动车） ---
                piece_moved = self.current_board[from_row][from_col]
                if piece_moved and piece_moved.lower() == 'k' and abs(to_col - from_col) == 2:
                    rospy.loginfo("检测到王车易位！执行车的移动。")
                    rook_row = from_row  # 车和王在同一横排
                    if to_col > from_col:
                        # 短易位 (O-O): 车从 h 线→f 线
                        rook_from_col, rook_to_col = 7, 5
                        rospy.loginfo(f"  短易位: 车 h→f")
                    else:
                        # 长易位 (O-O-O): 车从 a 线→d 线
                        rook_from_col, rook_to_col = 0, 3
                        rospy.loginfo(f"  长易位: 车 a→d")

                    # 等待上一指令完成
                    rospy.sleep(2.0)
                    timeout = 0
                    while self.arm_status != '0' and timeout < 100:
                        self.rate.sleep()
                        timeout += 1

                    msg_rook = PickAndPlaceGoalInCamera()
                    msg_rook.object_id_at_pick = ""
                    msg_rook.pick_position_in_camera = self._matrix_to_camera_point(rook_from_col, rook_row)
                    msg_rook.target_location_id_at_place = ""
                    msg_rook.place_position_in_camera = self._matrix_to_camera_point(rook_to_col, rook_row)
                    self.goal_pub.publish(msg_rook)
                    rospy.loginfo(f"已发布王车易位指令: 车 ({rook_from_col},{rook_row}) -> ({rook_to_col},{rook_row})")

                # 更新内部棋盘状态
                self.game.push_uci(move_uci)
                self.current_board = self.game.board_to_matrix()
                self.current_fen = self.game.get_fen()  # 同步 FEN（含过路兵状态）
                self.last_board = self.current_board.copy()
                self.is_ai_turn = False

                self.rate.sleep()
                continue

            # --- 检测人类走棋 ---
            # 将当前棋盘与上一次记录对比，如有变化则判定人类已走棋
            if (self.last_board is None or
                not np.array_equal(self.current_board, self.last_board)) and \
               not np.array_equal(self.current_board, self.initial_board):
                rospy.sleep(0.5)  # 去抖动，避免误检测
                rospy.loginfo("检测到人类走棋，切换到 AI 回合。")
                self.is_ai_turn = True
                self.last_board = self.current_board.copy()

            self.rate.sleep()


def matrix_to_fen(matrix, side_to_move='b'):
    """将 8×8 棋盘矩阵转换为 FEN 字符串。"""
    fen_parts = []
    for row in range(BOARD_ROWS):
        empty = 0
        row_str = ""
        for col in range(BOARD_COLS):
            piece = str(matrix[row][col]) if matrix[row][col] is not None else EMPTY_SQUARE
            if piece == EMPTY_SQUARE or piece == '0' or piece == '.':
                empty += 1
            else:
                if empty > 0:
                    row_str += str(empty)
                    empty = 0
                row_str += piece
        if empty > 0:
            row_str += str(empty)
        fen_parts.append(row_str)
    return "/".join(fen_parts) + f" {side_to_move} KQkq - 0 1"


def board_to_matrix(board):
    """将 python-chess Board 对象转换为 8×8 numpy 矩阵。"""
    matrix = np.full((BOARD_ROWS, BOARD_COLS), EMPTY_SQUARE, dtype=object)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)
            matrix[row][col] = piece.symbol()
    return matrix


if __name__ == '__main__':
    try:
        node = ChessAIOrchestrator()
        node.run()
    except rospy.ROSInterruptException:
        pass
