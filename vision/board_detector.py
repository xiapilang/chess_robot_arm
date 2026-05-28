#!/usr/bin/env python3
"""
国际象棋棋盘检测节点（ArUco 标签 + Intel RealSense 相机）。

改编自 EE368_Project 的 chessboard_detector_node.py，适配 8×8 国际象棋棋盘。

通过检测棋盘四个角上的 ArUco 标签来定位棋盘，并发布：
  - /chessboard_corners_3d（ChessboardCorners）—— 四个角点在相机坐标系中的 3D 坐标
  - /chessboard_corners_2d（ChessboardPixelCorners）—— 四个角点的 2D 像素坐标
  - /camera/color/image_raw（Image）—— 原始彩色图像供下游节点使用

ArUco 标签 ID 约定（默认）:
  ID 0 = 左上角, ID 1 = 右上角, ID 2 = 左下角, ID 3 = 右下角
"""

import rospy
import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from chess_robot_arm.msg import ChessboardCorners, ChessboardPixelCorners


class ChessboardArucoDetector:
    """基于 ArUco 标签和 RealSense 相机的 8×8 国际象棋棋盘检测器。"""

    def __init__(self):
        rospy.init_node('chessboard_aruco_detector', anonymous=False)

        # --- 参数加载 ---
        self.marker_size = rospy.get_param("~marker_size", 0.05)
        aruco_dict_name  = rospy.get_param("~aruco_dict_name", "DICT_6X6_250")

        self.corner_ids = {
            "top_left":     rospy.get_param("~top_left_id", 0),
            "top_right":    rospy.get_param("~top_right_id", 1),
            "bottom_left":  rospy.get_param("~bottom_left_id", 2),
            "bottom_right": rospy.get_param("~bottom_right_id", 3),
        }
        self.expected_ids = set(self.corner_ids.values())

        self.camera_frame_id = rospy.get_param("~camera_frame_id",
                                               "camera_color_optical_frame")
        self.show_cv_window  = rospy.get_param("~show_cv_window", True)

        # --- ArUco 字典与检测器参数 ---
        try:
            dict_id = getattr(aruco, aruco_dict_name)
            if dict_id is None:
                raise AttributeError
        except AttributeError:
            rospy.logerr(f"无效的 ArUco 字典名称: {aruco_dict_name}，回退使用 DICT_6X6_250。")
            dict_id = aruco.DICT_6X6_250

        self.dictionary = aruco.getPredefinedDictionary(dict_id)

        try:
            self.detector_params = aruco.DetectorParameters()
        except AttributeError:
            self.detector_params = aruco.DetectorParameters_create()

        # --- RealSense 相机管线初始化（仅使用彩色流） ---
        self.cam_w = rospy.get_param("~color_width", 1280)
        self.cam_h = rospy.get_param("~color_height", 720)
        self.fps   = rospy.get_param("~fps", 30)

        self.pipeline = rs.pipeline()
        self.config   = rs.config()
        self.config.enable_stream(rs.stream.color, self.cam_w, self.cam_h,
                                  rs.format.bgr8, self.fps)

        try:
            self.profile = self.pipeline.start(self.config)
        except RuntimeError as e:
            rospy.logerr(f"RealSense 管线启动失败: {e}")
            rospy.signal_shutdown("RealSense 初始化失败。")
            return

        # 获取相机内参
        color_profile = self.profile.get_stream(rs.stream.color)
        intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

        self.camera_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ], dtype=np.float32)

        # 畸变系数处理
        dist = np.array(intrinsics.coeffs, dtype=np.float32)
        if dist is None or len(dist) == 0 or intrinsics.model == rs.distortion.none:
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        else:
            if len(dist) < 5:
                dist = np.pad(dist, (0, 5 - len(dist)), 'constant')
            self.dist_coeffs = dist[:5].reshape(-1, 1)

        # --- 预计算：标签局部坐标系中的目标角点（左上物理边界） ---
        # 标签局部坐标系：原点在标签中心，X 轴向右，Y 轴向下，Z 轴朝外
        # 左上角对应 (-half, -half, 0)
        half = self.marker_size / 2.0
        self.target_corner_marker = np.array([[-half], [half], [0.0]], dtype=np.float32)

        rospy.loginfo(f"相机内参: fx={intrinsics.fx:.2f} fy={intrinsics.fy:.2f} "
                      f"cx={intrinsics.ppx:.2f} cy={intrinsics.ppy:.2f}")
        rospy.loginfo(f"畸变模型: {intrinsics.model}, 畸变系数: {self.dist_coeffs.flatten()}")

        # --- ROS 发布者 ---
        self.pub_3d  = rospy.Publisher("/chessboard_corners_3d", ChessboardCorners, queue_size=10)
        self.pub_2d  = rospy.Publisher("/chessboard_corners_2d", ChessboardPixelCorners, queue_size=10)
        self.pub_img = rospy.Publisher("/camera/color/image_raw", Image, queue_size=10)

        rospy.on_shutdown(self.shutdown)
        rospy.loginfo("ArUco 棋盘检测器初始化完成（8×8 国际象棋棋盘）。")

    def process_frame(self):
        """处理一帧图像：检测 ArUco 标签、计算角点、发布消息。"""
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
        except RuntimeError as e:
            rospy.logwarn_throttle(5, f"等待帧超时: {e}")
            return

        color_frame = frames.get_color_frame()
        if not color_frame:
            rospy.logwarn_throttle(5, "未接收到彩色帧")
            return

        color_img = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = aruco.detectMarkers(gray, self.dictionary,
                                              parameters=self.detector_params)

        now = rospy.Time.now()
        display = color_img.copy()

        detected_3d = {}    # 存储检测到的 3D 角点 {marker_id: Point}
        detected_2d = {}    # 存储检测到的 2D 像素角点 {marker_id: Point}

        if ids is not None and len(ids) > 0:
            aruco.drawDetectedMarkers(display, corners, ids)
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, self.camera_matrix, self.dist_coeffs)

            for i, marker_id_arr in enumerate(ids):
                marker_id = int(marker_id_arr[0])
                if marker_id not in self.expected_ids:
                    continue

                rvec = rvecs[i][0]
                tvec = tvecs[i][0]

                # 将标签左上角从标签局部坐标系变换到相机坐标系
                R_mat, _ = cv2.Rodrigues(rvec)
                tvec_col = tvec.reshape(3, 1)
                corner_cam = R_mat @ self.target_corner_marker + tvec_col
                detected_3d[marker_id] = Point(x=corner_cam[0, 0],
                                               y=corner_cam[1, 0],
                                               z=corner_cam[2, 0])

                # 获取标签左上角的 2D 像素坐标
                tl_pixel = corners[i][0][0]
                detected_2d[marker_id] = Point(x=float(tl_pixel[0]),
                                               y=float(tl_pixel[1]), z=0.0)

                # 可视化：绘制坐标轴和角点
                try:
                    cv2.drawFrameAxes(display, self.camera_matrix, self.dist_coeffs,
                                      rvec, tvec, self.marker_size / 2)
                    z_cam = corner_cam[2, 0]
                    if z_cam > 0:
                        u = int(intrinsics.fx * corner_cam[0, 0] / z_cam + intrinsics.ppx)
                        v = int(intrinsics.fy * corner_cam[1, 0] / z_cam + intrinsics.ppy)
                        cv2.circle(display, (u, v), 5, (0, 255, 255), -1)
                except Exception:
                    pass

        # 当全部四个角点都被检测到时发布消息
        if len(detected_3d) == 4 and len(detected_2d) == 4:
            msg_3d = ChessboardCorners()
            msg_3d.header.stamp = now
            msg_3d.header.frame_id = self.camera_frame_id
            msg_3d.top_left     = detected_3d[self.corner_ids["top_left"]]
            msg_3d.top_right    = detected_3d[self.corner_ids["top_right"]]
            msg_3d.bottom_left  = detected_3d[self.corner_ids["bottom_left"]]
            msg_3d.bottom_right = detected_3d[self.corner_ids["bottom_right"]]
            self.pub_3d.publish(msg_3d)

            msg_2d = ChessboardPixelCorners()
            msg_2d.header.stamp = now
            msg_2d.header.frame_id = self.camera_frame_id
            msg_2d.top_left_px     = detected_2d[self.corner_ids["top_left"]]
            msg_2d.top_right_px    = detected_2d[self.corner_ids["top_right"]]
            msg_2d.bottom_left_px  = detected_2d[self.corner_ids["bottom_left"]]
            msg_2d.bottom_right_px = detected_2d[self.corner_ids["bottom_right"]]
            self.pub_2d.publish(msg_2d)

            rospy.loginfo_throttle(1, "已发布 3D 和 2D 棋盘角点（8×8）。")

        # 发布原始彩色图像
        ros_img = Image()
        ros_img.header.stamp = now
        ros_img.header.frame_id = self.camera_frame_id
        ros_img.height = color_img.shape[0]
        ros_img.width  = color_img.shape[1]
        ros_img.encoding = "bgr8"
        ros_img.is_bigendian = 0
        ros_img.step = color_img.shape[1] * 3
        ros_img.data = color_img.tobytes()
        self.pub_img.publish(ros_img)

        # OpenCV 可视化窗口
        if self.show_cv_window:
            cv2.imshow("ArUco Board Detection", display)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                rospy.signal_shutdown("用户在 CV 窗口中按下了退出键")

    def run(self):
        """主循环：按设定帧率持续处理图像。"""
        # 预热相机，让自动曝光稳定
        rospy.loginfo("正在预热相机（等待自动曝光）...")
        for _ in range(60):
            try:
                self.pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                pass
        rospy.loginfo("相机预热完成。")

        # 诊断画面亮度
        try:
            test_frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            test_color = test_frames.get_color_frame()
            if test_color:
                test_img = np.asanyarray(test_color.get_data())
                rospy.loginfo(f"画面诊断: 尺寸={test_img.shape}, "
                              f"min={test_img.min()}, max={test_img.max()}, "
                              f"mean={test_img.mean():.1f}")
        except RuntimeError:
            pass

        # 预先创建可视化窗口
        if self.show_cv_window:
            cv2.namedWindow("ArUco Board Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("ArUco Board Detection", 960, 540)

        rate = rospy.Rate(self.fps if self.fps > 0 else 30)
        while not rospy.is_shutdown():
            self.process_frame()
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break
        if self.show_cv_window:
            cv2.destroyWindow("ArUco Board Detection")

    def shutdown(self):
        """节点关闭时的清理回调。"""
        rospy.loginfo("正在停止 ArUco 检测节点...")
        if hasattr(self, 'pipeline') and self.pipeline:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass
        if self.show_cv_window:
            cv2.destroyWindow("ArUco Board Detection")


if __name__ == '__main__':
    try:
        node = ChessboardArucoDetector()
        if hasattr(node, 'profile') and node.profile:
            node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("ArUco 检测节点被中断。")
    except Exception as e:
        rospy.logfatal(f"未处理的致命异常: {e}")
        import traceback
        traceback.print_exc()
