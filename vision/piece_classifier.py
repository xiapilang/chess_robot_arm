#!/usr/bin/env python3
"""
国际象棋棋子分类节点（基于 YOLOv8）。

在 8×8 国际象棋棋盘上检测 12 类棋子（6 种棋子 × 2 种颜色），
将检测结果映射到棋盘格位置，并发布 FEN 格式的棋盘状态字符串。

订阅话题:
  - /camera/color/image_raw     —— 原始彩色图像
  - /chessboard_corners_3d      —— 棋盘四个角点的 3D 坐标

发布话题:
  - /chess_board_matrix         —— FEN 字符串格式的棋盘状态
"""

import os
import rospy
import cv2
import numpy as np
import math
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String
from ultralytics import YOLO
from chess_robot_arm.msg import ChessboardCorners

from chess_robot_arm.utils.constants import (
    BOARD_ROWS, BOARD_COLS, PIECE_CLASS_NAMES, PIECE_ID_TO_FEN, EMPTY_SQUARE,
)


class PieceClassifier:
    """基于 YOLOv8 的国际象棋棋子分类器（12 类，适配 8×8 棋盘）。"""

    def __init__(self):
        rospy.init_node('piece_classifier', anonymous=True)

        self.bridge = CvBridge()

        # --- 加载 YOLO 模型 ---
        _pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _default_model = os.path.join(_pkg_root, "best.pt")
        model_path = rospy.get_param("~model_path", _default_model)
        self.model = YOLO(model_path)
        self.model.fuse()                # 模型融合以加速推理
        self.conf_threshold = rospy.get_param("~conf_threshold", 0.5)

        # --- ROS 订阅者 ---
        self.image_sub = rospy.Subscriber(
            "/camera/color/image_raw", Image, self._image_cb)
        self.corners_sub = rospy.Subscriber(
            "/chessboard_corners_3d", ChessboardCorners, self._corners_cb)
        self.arm_status_sub = rospy.Subscriber(
            "/my_gen3_lite/arm_status", String, self._arm_status_cb)

        # --- ROS 发布者 ---
        self.board_pub = rospy.Publisher("/chess_board_matrix", String, queue_size=10)
        self.home_pub = rospy.Publisher("/kinova_pick_place/home_cmd", String, queue_size=1)

        # --- 内部状态 ---
        self.latest_image   = None       # 最近一帧彩色图像
        self.latest_corners = None       # 棋盘四角点 3D 坐标 [(x,y), ...]
        self.grid_size      = None       # 棋盘格尺寸 (col_step, row_step) 单位：米
        self.corners_saved  = False      # 角点是否已保存
        self.arm_status     = ""         # 机械臂当前状态

        self.show_viz = rospy.get_param("~show_visualization", True)
        self.wait_for_arm = rospy.get_param("~wait_for_arm_home", True)

        rospy.loginfo("棋子分类器初始化完成（8×8 国际象棋，12 类棋子）。")

    def _image_cb(self, msg):
        """图像回调：将 ROS Image 消息转为 OpenCV 格式并缓存。"""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"图像转换失败: {e}")

    def _corners_cb(self, msg):
        """角点回调：缓存棋盘角点坐标，首次收到时计算格子尺寸并触发归位。"""
        self.latest_corners = [
            (msg.top_left.x,     msg.top_left.y),
            (msg.top_right.x,    msg.top_right.y),
            (msg.bottom_left.x,  msg.bottom_left.y),
            (msg.bottom_right.x, msg.bottom_right.y),
        ]
        if self.grid_size is None:
            self.grid_size = self._calc_grid_size()
            rospy.loginfo(f"棋盘格尺寸: 列间距={self.grid_size[0]:.4f}m, "
                          f"行间距={self.grid_size[1]:.4f}m")

        if not self.corners_saved:
            self.corners_saved = True
            rospy.loginfo("阶段1: 已保存棋盘角点，发送归位指令...")
            self.home_pub.publish(String(data="HOME"))
            self.corners_sub.unregister()
            rospy.loginfo("已取消订阅 /chessboard_corners_3d。")

    def _arm_status_cb(self, msg):
        """机械臂状态回调。"""
        self.arm_status = msg.data

    def _calc_grid_size(self):
        """根据四个角点坐标计算每个棋格的尺寸。"""
        if len(self.latest_corners) != 4:
            return None
        (tl_x, tl_y), (tr_x, tr_y), (bl_x, bl_y), (br_x, br_y) = self.latest_corners
        # 棋盘宽度：左上角到右上角的距离
        width  = math.hypot(tr_x - tl_x, tr_y - tl_y)
        # 棋盘高度：左上角到左下角的距离
        height = math.hypot(bl_y - tl_y, bl_x - tl_x)
        # 每个格子的尺寸
        col_step = width  / BOARD_COLS
        row_step = height / BOARD_ROWS
        return (col_step, row_step)

    def detect_pieces(self, img):
        """
        使用 YOLOv8 检测图像中的棋子。

        返回: 检测结果列表，每项包含 class_id, class_name, confidence, bbox, center
        """
        results = self.model(img, imgsz=640)

        detections = []
        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue

            cls_id = int(box.cls[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            cx = int((xyxy[0] + xyxy[2]) / 2)   # 边界框中心 x
            cy = int((xyxy[1] + xyxy[3]) / 2)   # 边界框中心 y

            detections.append({
                "class_id":   cls_id,
                "class_name": PIECE_CLASS_NAMES[cls_id],
                "confidence": conf,
                "bbox":       xyxy.tolist(),
                "center":     (cx, cy),
            })

        return detections

    def detections_to_board_matrix(self, detections):
        """
        将检测结果映射到 8×8 棋盘矩阵。

        使用像素坐标直接映射到棋盘的行列索引
        （假设相机平面与棋盘大致平行，即 top_view 姿态）。
        """
        if self.latest_corners is None or self.grid_size is None:
            return np.full((BOARD_ROWS, BOARD_COLS), EMPTY_SQUARE, dtype=object)

        board = np.full((BOARD_ROWS, BOARD_COLS), EMPTY_SQUARE, dtype=object)

        for det in detections:
            fen_char = PIECE_ID_TO_FEN.get(det["class_id"], EMPTY_SQUARE)
            cx, cy = det["center"]

            # 按图像比例映射到棋盘格
            col = int(round(BOARD_COLS * cx / self.latest_image.shape[1]))
            row = int(round(BOARD_ROWS * cy / self.latest_image.shape[0]))

            # 边界钳制
            col = max(0, min(col, BOARD_COLS - 1))
            row = max(0, min(row, BOARD_ROWS - 1))

            board[row][col] = fen_char

        return board

    def board_to_fen_string(self, board_matrix):
        """
        将 8×8 棋盘矩阵转换为 FEN 字符串。
        大写字母 = 白方，小写字母 = 黑方。
        """
        fen_parts = []
        for row in range(BOARD_ROWS):
            empty_count = 0
            row_str = ""
            for col in range(BOARD_COLS):
                piece = board_matrix[row][col]
                if piece == EMPTY_SQUARE or piece == '0':
                    empty_count += 1
                else:
                    if empty_count > 0:
                        row_str += str(empty_count)
                        empty_count = 0
                    row_str += str(piece)
            if empty_count > 0:
                row_str += str(empty_count)
            fen_parts.append(row_str)

        return "/".join(fen_parts)

    def visualize(self, img, board_matrix, detections):
        """在图像上绘制棋盘网格和棋子分类标签。"""
        h, w = img.shape[:2]
        cell_w = w // BOARD_COLS
        cell_h = h // BOARD_ROWS

        # 绘制 8×8 网格线
        for i in range(1, BOARD_ROWS):
            cv2.line(img, (0, i * cell_h), (w, i * cell_h), (0, 255, 0), 1)
        for j in range(1, BOARD_COLS):
            cv2.line(img, (j * cell_w, 0), (j * cell_w, h), (0, 255, 0), 1)

        # 绘制每个检测结果
        for det in detections:
            cx, cy = det["center"]
            label = det["class_name"]
            color = (0, 0, 255) if "white" in label else (255, 255, 255)
            bg_color = (0, 0, 0)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (cx - tw // 2, cy - th - 8),
                          (cx + tw // 2, cy + 5), bg_color, -1)
            cv2.putText(img, label, (cx - tw // 2, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow("Piece Detection", img)
        cv2.waitKey(1)

    def run(self):
        """主运行循环：分三阶段执行。

        阶段1 -- 等待 ArUco 角点数据就绪
        阶段2 -- 等待机械臂归位完成
        阶段3 -- 开始棋子检测与 FEN 发布
        """
        rate = rospy.Rate(2)

        # --- 阶段1: 等待角点 ---
        rospy.loginfo("阶段1: 等待棋盘角点...")
        while not rospy.is_shutdown() and not self.corners_saved:
            rate.sleep()
        rospy.loginfo("阶段1: 角点已保存。")

        # --- 阶段2: 等待机械臂归位 ---
        if self.wait_for_arm:
            rospy.loginfo("阶段2: 等待机械臂归位 (arm_status='0')...")
            while not rospy.is_shutdown() and self.arm_status != "0":
                rospy.logwarn_throttle(5, f"阶段2: 机械臂状态={self.arm_status}，等待归位...")
                rate.sleep()
            rospy.loginfo("阶段2: 机械臂已归位。")
        else:
            rospy.loginfo("阶段2: 跳过（wait_for_arm_home=false）。")

        # --- 阶段3: 棋子检测 ---
        rospy.loginfo("阶段3: 开始棋子识别。")
        # 给相机一点时间稳定画面
        rospy.sleep(1.0)

        detect_rate = rospy.Rate(1)  # 1 Hz
        while not rospy.is_shutdown():
            if self.latest_image is None:
                rospy.logwarn_throttle(5, "等待图像数据...")
                detect_rate.sleep()
                continue

            img = self.latest_image.copy()
            detections = self.detect_pieces(img)
            board = self.detections_to_board_matrix(detections)
            fen_str = self.board_to_fen_string(board)

            rospy.loginfo(f"棋盘 FEN: {fen_str}")
            self.board_pub.publish(String(data=fen_str))

            if self.show_viz:
                self.visualize(img, board, detections)

            detect_rate.sleep()

        if self.show_viz:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        node = PieceClassifier()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"棋子分类节点出错: {e}")
        import traceback
        traceback.print_exc()
