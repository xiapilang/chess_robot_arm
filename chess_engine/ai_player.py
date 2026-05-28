#!/usr/bin/env python3
"""
国际象棋 AI 引擎（python-chess + Stockfish）。

提供与原项目 elephant_fish.py 相同接口的 ai_move_from_board() 函数，
支持 Stockfish 引擎和简易降级 AI 两种模式。

若 Stockfish 不可用，自动降级为基于 python-chess 的简易 AI。
"""

import chess
import chess.engine
import numpy as np
import os
from chess_robot_arm.utils.constants import BOARD_ROWS, BOARD_COLS, EMPTY_SQUARE


class ChessAI:
    """
    国际象棋 AI 封装类。

    优先尝试加载 Stockfish 引擎，若不可用则降级为简易 AI
    （优先吃子 > 优先将军 > 随机合法走法）。
    """

    def __init__(self, stockfish_path="/usr/games/stockfish",
                 skill_level=10, think_time=3.0):
        self.stockfish_path = stockfish_path
        self.skill_level = skill_level
        self.think_time = think_time
        self.engine = None
        self._init_engine()

    def _init_engine(self):
        """尝试初始化 Stockfish 引擎。"""
        if os.path.exists(self.stockfish_path):
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
                self.engine.configure({"Skill Level": self.skill_level})
                print(f"Stockfish 引擎已初始化（难度等级={self.skill_level}）。")
                return
            except Exception as e:
                print(f"Stockfish 初始化失败: {e}。将使用降级 AI。")
        else:
            print(f"未找到 Stockfish（路径: '{self.stockfish_path}'），将使用降级 AI。")

    def get_best_move(self, board):
        """
        获取当前局面的最佳走法。

        返回 (Move, None) 或 (None, 错误描述)。
        """
        if self.engine is not None:
            try:
                limit = chess.engine.Limit(time=self.think_time)
                result = self.engine.play(board, limit)
                return result.move, None
            except Exception as e:
                print(f"Stockfish 引擎出错: {e}，降级为简易 AI。")
                self.engine = None

        return self._fallback_move(board)

    def _fallback_move(self, board):
        """简易降级 AI：优先吃子，其次将军，否则随机走。"""
        import random
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        # 优先吃子
        captures = [m for m in legal_moves if board.is_capture(m)]
        if captures:
            return random.choice(captures)

        # 其次将军
        checks = [m for m in legal_moves if board.gives_check(m)]
        if checks:
            return random.choice(checks)

        # 否则随机
        return random.choice(legal_moves)

    def close(self):
        """安全关闭引擎进程。"""
        if self.engine is not None:
            self.engine.quit()

    def __del__(self):
        self.close()


def ai_move_from_board(board_fen, ai_color=chess.BLACK, think_time=3.0,
                       stockfish_path="/usr/games/stockfish"):
    """
    AI 引擎主接口函数，与原项目 elephant_fish 接口一致。

    参数:
        board_fen: 当前棋盘状态的 FEN 字符串
        ai_color: AI 执子颜色（默认 chess.BLACK，机械臂执黑）
        think_time: AI 思考时间（秒）
        stockfish_path: Stockfish 可执行文件路径

    返回:
        (uci_move_str, from_square, to_square, is_capture, is_en_passant, is_castling)
        失败时返回 (None, None, None, False, False, False)
    """
    board = chess.Board(board_fen)

    if board.is_game_over():
        return None, None, None, False, False, False

    ai = ChessAI(stockfish_path=stockfish_path, think_time=think_time)
    move, error = ai.get_best_move(board)
    ai.close()

    if move is None:
        print(f"AI 未返回走法。错误: {error}")
        return None, None, None, False, False, False

    is_capture = board.is_capture(move)
    is_en_passant = board.is_en_passant(move)
    is_castling = board.is_castling(move)
    from_sq = chess.square_name(move.from_square)
    to_sq   = chess.square_name(move.to_square)

    # 执行走法并打印结果
    board.push(move)
    print(f"AI 走法: {from_sq} -> {to_sq}（吃子={is_capture}, 过路兵={is_en_passant}, 易位={is_castling}）")
    print(board)

    return move.uci(), from_sq, to_sq, is_capture, is_en_passant, is_castling


def uci_to_matrix_coords(uci_square):
    """
    将 UCI 坐标名转换为 top_view 矩阵中的 (row, col)。

    row 0 = 第 8 横排，col 0 = a 线。
    示例: 'e2' → (6, 4)
    """
    file_idx = chess.FILE_NAMES.index(uci_square[0])
    rank = int(uci_square[1:])
    row = 8 - rank
    col = file_idx
    return (row, col)


def matrix_coords_to_uci(row, col):
    """
    将 top_view 矩阵的 (row, col) 转换为 UCI 坐标名。

    示例: (6, 4) → 'e2'
    """
    rank = 8 - row
    file_char = chess.FILE_NAMES[col]
    return f"{file_char}{rank}"


# ============================================================
# 适配器函数：与原项目 elephant_fish.py 的 ai_move_from_matrix() 接口兼容
# ============================================================
def ai_move_from_matrix(np_board):
    """
    适配器函数，接口与原项目 elephant_fish.py 完全一致，
    便于在原有 ROS 节点框架中直接替换。

    参数:
        np_board: 8×8 numpy 数组，每格为 FEN 字符
                  ('K','Q','R','B','N','P' = 白方,
                   'k','q','r','b','n','p' = 黑方,
                   '0' = 空格)

    返回:
        (new_board_8x8, (from_col, from_row), (to_col, to_row))
        失败时返回 (None, None, None)
    """
    from chess_robot_arm.chess_engine.game_state import GameState

    gs = GameState(robot_plays_as=chess.BLACK)

    # 从矩阵构建 FEN 字符串
    fen_parts = []
    for row in range(BOARD_ROWS):
        empty = 0
        row_str = ""
        for col in range(BOARD_COLS):
            piece = str(np_board[row][col]) if np_board[row][col] is not None else EMPTY_SQUARE
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
    fen = "/".join(fen_parts) + " b KQkq - 0 1"

    # 获取 AI 走法
    move_uci, from_sq, to_sq, is_capture, is_en_passant, is_castling = ai_move_from_board(fen)

    if move_uci is None:
        return None, None, None

    from_row, from_col = uci_to_matrix_coords(from_sq)
    to_row, to_col     = uci_to_matrix_coords(to_sq)

    # 构建走法后的新棋盘矩阵
    new_board = np.copy(np_board)
    new_board[to_row][to_col] = new_board[from_row][from_col]
    new_board[from_row][from_col] = EMPTY_SQUARE

    return new_board, (from_col, from_row), (to_col, to_row)


if __name__ == "__main__":
    # 命令行测试
    import sys
    fen = sys.argv[1] if len(sys.argv) > 1 else chess.STARTING_FEN
    print(f"输入 FEN: {fen}")
    move, frm, to, cap, ep, cas = ai_move_from_board(fen)
    print(f"结果: 走法={move}, 起点={frm}, 终点={to}, 吃子={cap}, 过路兵={ep}, 易位={cas}")
