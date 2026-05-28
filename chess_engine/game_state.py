#!/usr/bin/env python3
"""
国际象棋棋盘状态管理器（基于 python-chess）。

管理 8×8 国际象棋的棋盘状态、走法合法性校验、被吃子追踪、
以及 FEN 字符串与棋盘矩阵之间的相互转换。

棋盘方向约定（从机械臂 top_view 俯瞰视角）:
  - row 0（顶部）    = 第 8 横排（黑方底线）
  - row 7（底部）    = 第 1 横排（白方底线）
  - col 0（最左侧）  = a 线
  - col 7（最右侧）  = h 线

机械臂默认执黑（与原始 EE368 项目保持一致）。
"""

import chess
import numpy as np
from chess_robot_arm.utils.constants import BOARD_ROWS, BOARD_COLS, EMPTY_SQUARE


class GameState:
    """
    封装 python-chess.Board，为机械臂对弈流水线提供便捷接口。

    提供棋盘矩阵 ↔ FEN 字符串 ↔ 行列坐标 ↔ UCI 走法的转换，
    以及走法合法性校验和被吃子计数。
    """

    def __init__(self, robot_plays_as=chess.BLACK):
        self.board = chess.Board()          # python-chess 棋盘对象
        self.robot_color = robot_plays_as   # 机械臂执子颜色
        self.captured_count = 0             # 机械臂已吃棋子计数

    def reset(self):
        """重置为标准初始局面。"""
        self.board.reset()
        self.captured_count = 0
        return self.board.fen()

    def set_fen(self, fen_str):
        """从 FEN 字符串设置棋盘状态。"""
        try:
            self.board.set_fen(fen_str)
            return True
        except ValueError as e:
            return False

    def get_fen(self):
        """返回当前 FEN 字符串。"""
        return self.board.fen()

    def is_game_over(self):
        """判断对局是否结束。"""
        return self.board.is_game_over()

    def is_robot_turn(self):
        """判断当前是否轮到机械臂走棋。"""
        return self.board.turn == self.robot_color

    def result(self):
        """返回对局结果字符串，如 '1-0', '0-1', '1/2-1/2'。"""
        return self.board.result()

    def get_legal_moves(self):
        """获取当前所有合法走法列表。"""
        return list(self.board.legal_moves)

    def push_uci(self, uci_str):
        """
        执行一个 UCI 格式的走法（如 'e2e4'）。

        返回 (success, error_message)。
        """
        try:
            move = chess.Move.from_uci(uci_str)
            if move in self.board.legal_moves:
                self.board.push(move)
                return True, None
            return False, "非法走法"
        except Exception as e:
            return False, str(e)

    def push_san(self, san_str):
        """
        执行一个标准代数记谱法走法（如 'e4', 'Nf3'）。

        返回 (success, error_message)。
        """
        try:
            move = self.board.parse_san(san_str)
            self.board.push(move)
            return True, None
        except Exception as e:
            return False, str(e)

    def matrix_to_board(self, matrix):
        """
        将 8×8 棋盘矩阵转换为 python-chess Board 对象。

        矩阵索引：matrix[row][col]，row 0 = 第 8 横排，row 7 = 第 1 横排。
        返回 (success, fen_string)。
        """
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

        fen = "/".join(fen_parts) + " w KQkq - 0 1"
        # 根据机械臂执子颜色调整当前走子方
        fen = fen.replace(" w ", f" {'w' if self.board.turn == chess.WHITE else 'b'} ")

        try:
            self.board.set_fen(fen)
            return True, fen
        except ValueError as e:
            return False, str(e)

    def board_to_matrix(self):
        """
        将当前 python-chess 棋盘转换为 8×8 numpy 矩阵。
        row 0 = 第 8 横排，row 7 = 第 1 横排。
        """
        matrix = np.full((BOARD_ROWS, BOARD_COLS), EMPTY_SQUARE, dtype=object)
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is not None:
                row = 7 - chess.square_rank(square)   # rank 8 → row 0
                col = chess.square_file(square)        # file 'a' → col 0
                matrix[row][col] = piece.symbol()
        return matrix

    def square_to_uci(self, row, col):
        """
        将棋盘矩阵中的 (row, col) 位置转换为 UCI 坐标名。
        row 0 = 第 8 横排，col 0 = a 线。
        """
        rank = 8 - row
        file_idx = col
        return chess.FILE_NAMES[file_idx] + str(rank)

    def uci_to_matrix_coords(self, uci_square):
        """将 UCI 坐标名（如 'e2'）转换为矩阵中的 (row, col)。"""
        file_idx = chess.FILE_NAMES.index(uci_square[0])
        rank = int(uci_square[1])
        row = 8 - rank
        col = file_idx
        return (row, col)

    def find_move(self, from_row, from_col, to_row, to_col):
        """
        查找从 (from_row, from_col) 到 (to_row, to_col) 的合法走法。

        返回 (Move, is_capture) 或 (None, False)。
        """
        from_sq_name = self.square_to_uci(from_row, from_col)
        to_sq_name   = self.square_to_uci(to_row, to_col)

        try:
            from_sq = chess.parse_square(from_sq_name)
            to_sq   = chess.parse_square(to_sq_name)
        except ValueError:
            return None, False

        for move in self.board.legal_moves:
            if move.from_square == from_sq and move.to_square == to_sq:
                is_capture = self.board.is_capture(move)
                return move, is_capture

        return None, False

    def record_capture(self):
        """记录一次吃子（用于弃子区偏移计算）。"""
        self.captured_count += 1
        return self.captured_count


if __name__ == "__main__":
    # 快速功能验证
    gs = GameState()
    print("初始棋盘:")
    print(gs.board)
    print("\n棋盘矩阵:")
    print(gs.board_to_matrix())
    print("\nFEN:", gs.get_fen())
