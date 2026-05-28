#!/usr/bin/env python3
"""
相机-机械臂手眼标定工具。

通过观测已知基座标系位置的 ArUco 标签，计算并验证
相机与机械臂之间的坐标变换关系（T_ee_camera 和 T_base_camera 链）。

使用方法:
    rosrun chess_robot_arm calibrate.py
    将 ArUco 标签（ID=0）放置在基座标系中已知位置
    按 's' 键记录样本，按 'q' 退出并输出标定结果
"""

import rospy
import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R


class Calibrator:
    """手眼标定工具：通过已知位置的 ArUco 标签建立相机-基座变换关系。"""

    def __init__(self):
        rospy.init_node('camera_arm_calibrator', anonymous=True)

        # --- 初始化 RealSense 相机 ---
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(self.config)

        # 获取相机内参
        color_profile = self.profile.get_stream(rs.stream.color)
        intr = color_profile.as_video_stream_profile().get_intrinsics()

        self.camera_matrix = np.array([
            [intr.fx, 0, intr.ppx],
            [0, intr.fy, intr.ppy],
            [0, 0, 1]
        ], dtype=np.float32)

        dist = np.array(intr.coeffs, dtype=np.float32)
        if len(dist) < 5:
            dist = np.pad(dist, (0, 5 - len(dist)), 'constant')
        self.dist_coeffs = dist[:5].reshape(-1, 1)

        # --- ArUco 检测器 ---
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.detector_params = aruco.DetectorParameters()

        rospy.loginfo("标定工具就绪。将 ArUco 标签 (ID=0) 放置在基座标系中的已知位置。")
        rospy.loginfo("按 's' 键记录样本，按 'q' 键退出并输出标定结果。")

    def detect_marker(self, frame):
        """检测图像中的 ArUco 标签，返回 (rvec, tvec)。"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(
            gray, self.dictionary, parameters=self.detector_params)
        if ids is not None and len(ids) > 0:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, 0.05, self.camera_matrix, self.dist_coeffs)
            return rvecs[0][0], tvecs[0][0]
        return None, None

    def run(self):
        """主运行循环：显示相机图像，接受用户输入记录标定样本。"""
        samples = []
        rospy.loginfo("将标签放在基座标系的已知位置，然后按 's'。")
        rospy.loginfo("输入格式: x y z（以米为单位，基座标系）")

        # 预热相机，让自动曝光稳定（丢弃前 60 帧）
        rospy.loginfo("正在预热相机（等待自动曝光）...")
        for i in range(60):
            self.pipeline.wait_for_frames()
        rospy.loginfo("相机预热完成。")

        # 确保窗口可见并置顶
        cv2.namedWindow("手眼标定", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("手眼标定", 960, 540)

        # 取一帧确认画面亮度
        test_frames = self.pipeline.wait_for_frames()
        test_color = test_frames.get_color_frame()
        if test_color:
            test_img = np.asanyarray(test_color.get_data())
            rospy.loginfo(f"画面诊断: 尺寸={test_img.shape}, "
                          f"min={test_img.min()}, max={test_img.max()}, "
                          f"mean={test_img.mean():.1f}")
            if test_img.max() < 20:
                rospy.logwarn("画面极暗！请检查镜头盖是否打开、环境是否有光照。")

        while not rospy.is_shutdown():
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                rospy.logwarn_throttle(3, "未收到彩色帧...")
                continue
            frame = np.asanyarray(color_frame.get_data())

            rvec, tvec = self.detect_marker(frame)
            display = frame.copy()

            color = (0, 255, 0) if rvec is not None else (0, 0, 255)
            status = f"tvec: {np.round(tvec, 3)}" if rvec is not None else "无标签 - 将标签放入画面"
            cv2.putText(display, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(display, "S=记录 Q=退出 (先点此窗口!)", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            cv2.imshow("手眼标定", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s') and rvec is not None:
                try:
                    pos_str = input("输入标签在基座标系中的位置（x y z 以米为单位）: ")
                    parts = [float(p) for p in pos_str.strip().split()]
                    if len(parts) != 3:
                        print("需要 3 个数值。")
                        continue
                    base_pos = np.array(parts)
                    R_mat, _ = cv2.Rodrigues(rvec)
                    T_marker_cam = np.eye(4)
                    T_marker_cam[:3, :3] = R_mat
                    T_marker_cam[:3, 3]  = tvec
                    samples.append((base_pos, T_marker_cam))
                    print(f"已记录样本 {len(samples)}。"
                          f"基座位置: {base_pos}, 相机 tvec: {np.round(tvec, 3)}")
                except ValueError:
                    print("输入格式无效。")

            elif key == ord('q'):
                break

            elif key == ord('s') and rvec is None:
                rospy.logwarn("未检测到 ArUco 标签，无法记录。请将标签放入画面。")

        cv2.destroyAllWindows()
        self.pipeline.stop()

        if len(samples) < 2:
            rospy.logwarn("至少需要 2 个样本进行标定。")
            return

        print("\n" + "=" * 50)
        print(f"已收集 {len(samples)} 个样本。")
        print("标定结果 (T_base_camera = T_base_marker * inv(T_camera_marker)):")

        for i, (base_pos, T_marker_cam) in enumerate(samples):
            T_cam_marker = np.linalg.inv(T_marker_cam)
            print(f"\n样本 {i}:")
            print(f"  标签基座位置: {base_pos}")
            print(f"  T_camera_marker:\n{np.round(T_cam_marker, 3)}")

        # 输出优化建议
        print("\n如需优化 T_ee_camera，将机械臂置于已知关节构型，"
              "运行 forward_kinematics 获取 T_base_ee。")
        print("然后通过公式: T_ee_camera = inv(T_base_ee) * T_base_camera 计算。")


if __name__ == '__main__':
    try:
        cal = Calibrator()
        cal.run()
    except rospy.ROSInterruptException:
        pass
