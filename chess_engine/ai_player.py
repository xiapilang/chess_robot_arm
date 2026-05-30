#!/usr/bin/env python3
"""
国际象棋 AI 引擎（Stockfish + 简易降级 AI）。

提供与原项目 elephant_fish.py 相同接口的 ai_move_from_board() 函数。
Stockfish 通过 subprocess + UCI 协议直接控制，避开 python-chess 引擎管理
在 ROS 环境中的信号冲突问题。
"""

import chess
import os
import subprocess
import time
import numpy as np
from chess_robot_arm.utils.constants import BOARD_ROWS, BOARD_COLS, EMPTY_SQUARE


class ChessAI:
    """国际象棋 AI：Stockfish UCI 直连 + 简易降级 AI。"""

    def __init__(self, stockfish_path="/usr/games/stockfish",
                 skill_level=10, think_time=3.0):
        self.stockfish_path = stockfish_path
        self.skill_level = skill_level
        self.think_time = think_time
        self._proc = None
        self._init_engine()

    def _init_engine(self):
        """启动 Stockfish 子进程并完成 UCI 握手。"""
        if not os.path.exists(self.stockfish_path):
            print(f"未找到 Stockfish（路径: '{self.stockfish_path}'），将使用降级 AI。")
            return
        try:
            self._proc = subprocess.Popen(
                [self.stockfish_path],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True, bufsize=1,
                start_new_session=True,
            )
            self._send("uci")
            self._wait_for("uciok", timeout=5.0)
            self._send(f"setoption name Skill Level value {self.skill_level}")
            self._send("isready")
            self._wait_for("readyok", timeout=5.0)
            print(f"Stockfish 引擎已初始化（难度等级={self.skill_level}）。")
        except Exception as e:
            print(f"Stockfish 初始化失败: {e}。将使用降级 AI。")
            self._kill_proc()
            self._proc = None

    def _send(self, cmd):
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.write(cmd + "\n")
                self._proc.stdin.flush()
            except Exception:
                self._kill_proc()
                self._proc = None

    def _wait_for(self, keyword, timeout=10.0):
        if not self._proc:
            raise RuntimeError("Engine not running")
        start = time.time()
        while time.time() - start < timeout:
            if self._proc.poll() is not None:
                raise RuntimeError(f"Engine died (exit {self._proc.returncode})")
            line = self._proc.stdout.readline()
            if not line:
                time.sleep(0.01)
                continue
            if keyword in line:
                return line
        raise TimeoutError(f"Timeout waiting for '{keyword}'")

    def _read_until(self, keyword, timeout=10.0):
        """读取直到某行包含 keyword，返回所有读取的行。"""
        lines = []
        if not self._proc:
            raise RuntimeError("Engine not running")
        start = time.time()
        while time.time() - start < timeout:
            if self._proc.poll() is not None:
                raise RuntimeError(f"Engine died (exit {self._proc.returncode})")
            line = self._proc.stdout.readline()
            if not line:
                time.sleep(0.01)
                continue
            lines.append(line.strip())
            if keyword in line:
                return lines
        raise TimeoutError(f"Timeout waiting for '{keyword}'")

    def _kill_proc(self):
        try:
            if self._proc:
                self._proc.terminate()
                time.sleep(0.1)
                if self._proc.poll() is None:
                    self._proc.kill()
                self._proc = None
        except Exception:
            pass

    def get_best_move(self, board):
        """获取当前局面的最佳走法。返回 chess.Move 或 None。"""
        if self._proc is not None and self._proc.poll() is not None:
            # 引擎已死，清理
            self._proc = None

        if self._proc is None:
            self._init_engine()

        if self._proc is not None:
            try:
                fen = board.fen()
                self._send(f"position fen {fen}")
                self._send(f"go movetime {int(self.think_time * 1000)}")
                lines = self._read_until("bestmove", timeout=self.think_time + 5.0)
                for line in lines:
                    if line.startswith("bestmove"):
                        parts = line.split()
                        if len(parts) >= 2 and parts[1] != "(none)":
                            return chess.Move.from_uci(parts[1])
            except Exception as e:
                print(f"Stockfish 引擎出错: {e}，降级为简易 AI。")
                self._kill_proc()
                self._proc = None

        return self._fallback_move(board)

    def _fallback_move(self, board):
        """简易降级 AI：优先吃子，其次将军，否则随机走。"""
        import random
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        captures = [m for m in legal_moves if board.is_capture(m)]
        if captures:
            return random.choice(captures)
        checks = [m for m in legal_moves if board.gives_check(m)]
        if checks:
            return random.choice(checks)
        return random.choice(legal_moves)

    def close(self):
        self._kill_proc()

    def __del__(self):
        self.close()


_session_engine = None


def ai_move_from_board(board_fen, ai_color=chess.BLACK, think_time=3.0,
                       stockfish_path="/usr/games/stockfish"):
    """
    AI 引擎主接口函数。

    返回:
        (uci_move_str, from_square, to_square, is_capture, is_en_passant, is_castling)
    """
    global _session_engine

    board = chess.Board(board_fen)
    if board.is_game_over():
        return None, None, None, False, False, False

    if _session_engine is None:
        _session_engine = ChessAI(stockfish_path=stockfish_path,
                                  think_time=think_time)

    move = _session_engine.get_best_move(board)
    if move is None:
        print("AI 未返回走法。")
        return None, None, None, False, False, False

    is_capture = board.is_capture(move)
    is_en_passant = board.is_en_passant(move)
    is_castling = board.is_castling(move)
    from_sq = chess.square_name(move.from_square)
    to_sq = chess.square_name(move.to_square)

    board.push(move)
    print(f"AI 走法: {from_sq} -> {to_sq}（吃子={is_capture}, 过路兵={is_en_passant}, 易位={is_castling}）")
    print(board)

    return move.uci(), from_sq, to_sq, is_capture, is_en_passant, is_castling


def close_session_engine():
    global _session_engine
    if _session_engine is not None:
        _session_engine.close()
        _session_engine = None


def uci_to_matrix_coords(uci_square):
    """UCI 坐标 → (row, col)。"""
    file_idx = chess.FILE_NAMES.index(uci_square[0])
    rank = int(uci_square[1:])
    row = 8 - rank
    col = 7 - file_idx
    return (row, col)


def matrix_coords_to_uci(row, col):
    """(row, col) → UCI 坐标。"""
    rank = 8 - row
    file_char = chess.FILE_NAMES[7 - col]
    return f"{file_char}{rank}"


def ai_move_from_matrix(np_board):
    """适配器：与原项目 elephant_fish.py 接口兼容。"""
    from chess_robot_arm.chess_engine.game_state import GameState

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

    move_uci, from_sq, to_sq, is_capture, is_en_passant, is_castling = ai_move_from_board(fen)
    if move_uci is None:
        return None, None, None

    from_row, from_col = uci_to_matrix_coords(from_sq)
    to_row, to_col = uci_to_matrix_coords(to_sq)

    new_board = np.copy(np_board)
    new_board[to_row][to_col] = new_board[from_row][from_col]
    new_board[from_row][from_col] = EMPTY_SQUARE
    return new_board, (from_col, from_row), (to_col, to_row)


if __name__ == "__main__":
    import sys
    fen = sys.argv[1] if len(sys.argv) > 1 else chess.STARTING_FEN
    print(f"输入 FEN: {fen}")
    move, frm, to, cap, ep, cas = ai_move_from_board(fen)
    print(f"结果: 走法={move}, 起点={frm}, 终点={to}, 吃子={cap}, 过路兵={ep}, 易位={cas}")
