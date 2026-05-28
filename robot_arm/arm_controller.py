#!/usr/bin/env python3
"""
Kinova Gen3 Lite 机械臂底层控制器（基于 Kortex ROS 驱动）。

封装 kortex_driver 的 ROS 服务，提供：
  - 笛卡尔空间位姿移动
  - 夹爪开合控制
  - 故障清除与通知激活
  - 安全归位

改编自 EE368_Project 的 move_cartesian.py。
"""

import rospy
import time
import numpy as np

from kortex_driver.srv import *
from kortex_driver.msg import *


class KinovaArmController:
    """Kinova Gen3 Lite 机械臂底层控制器。"""

    def __init__(self, robot_name="my_gen3_lite"):
        self.robot_name = robot_name
        self.is_gripper_present = False
        self.is_init_success = False
        self.last_action_notif_type = None

        try:
            self.is_gripper_present = rospy.get_param(
                f"/{self.robot_name}/is_gripper_present", False)

            # 动作通知订阅者
            self.action_sub = rospy.Subscriber(
                f"/{self.robot_name}/action_topic",
                ActionNotification,
                self._action_cb)

            # ROS 服务代理
            clear_faults_srv = f'/{self.robot_name}/base/clear_faults'
            rospy.wait_for_service(clear_faults_srv)
            self.clear_faults = rospy.ServiceProxy(clear_faults_srv, Base_ClearFaults)

            execute_traj_srv = f'/{self.robot_name}/base/execute_waypoint_trajectory'
            rospy.wait_for_service(execute_traj_srv)
            self.execute_waypoint_trajectory = rospy.ServiceProxy(
                execute_traj_srv, ExecuteWaypointTrajectory)

            if self.is_gripper_present:
                gripper_srv = f'/{self.robot_name}/base/send_gripper_command'
                rospy.wait_for_service(gripper_srv)
                self.send_gripper_command = rospy.ServiceProxy(
                    gripper_srv, SendGripperCommand)

            activate_notif_srv = f'/{self.robot_name}/base/activate_publishing_of_action_topic'
            rospy.wait_for_service(activate_notif_srv)
            self.activate_notifications = rospy.ServiceProxy(
                activate_notif_srv, OnNotificationActionTopic)

            self.is_init_success = True
            rospy.loginfo(f"KinovaArmController: '{self.robot_name}' 初始化成功 "
                          f"（夹爪={'有' if self.is_gripper_present else '无'}）。")

        except Exception as e:
            rospy.logerr(f"KinovaArmController 初始化失败: {e}")
            self.is_init_success = False

    def _action_cb(self, notif):
        """动作通知回调：记录最近的机器人动作事件。"""
        self.last_action_notif_type = notif.action_event

    def _fill_cartesian_waypoint(self, x, y, z, theta_x, theta_y, theta_z,
                                  blending_radius=0.0):
        """构造笛卡尔空间路径点消息。"""
        waypoint = Waypoint()
        cart = CartesianWaypoint()
        cart.pose.x = x
        cart.pose.y = y
        cart.pose.z = z
        cart.pose.theta_x = theta_x
        cart.pose.theta_y = theta_y
        cart.pose.theta_z = theta_z
        cart.reference_frame = CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_BASE
        cart.blending_radius = blending_radius
        waypoint.oneof_type_of_waypoint.cartesian_waypoint.append(cart)
        return waypoint

    def _wait_for_action_end(self, timeout=10.0):
        """等待动作完成或超时。"""
        start = time.time()
        while not rospy.is_shutdown() and (time.time() - start) < timeout:
            if self.last_action_notif_type == ActionEvent.ACTION_END:
                return True
            elif self.last_action_notif_type == ActionEvent.ACTION_ABORT:
                rospy.logwarn("机械臂动作被中止。")
                return False
            time.sleep(0.01)
        rospy.logwarn("动作等待超时。")
        return False

    def activate(self):
        """清除故障并激活动作通知（带重试，应对 Kortex ActionServer 初始化延迟）。"""
        if not self.is_init_success:
            return False

        # 等待 Kortex 内部 ActionServer 完全初始化
        rospy.loginfo("等待 Kortex ActionServer 就绪...")
        rospy.sleep(3.0)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.clear_faults()
                rospy.sleep(1.0)
                self.activate_notifications(OnNotificationActionTopicRequest())
                rospy.sleep(1.0)
                rospy.loginfo("机械臂激活成功。")
                return True
            except rospy.ServiceException as e:
                rospy.logwarn(f"激活失败 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    rospy.sleep(2.0)

        rospy.logerr(f"机械臂激活失败，已重试 {max_retries} 次。")
        return False

    def move_to_cartesian_pose(self, x, y, z, theta_x, theta_y, theta_z):
        """
        将末端执行器移动到指定的绝对笛卡尔位姿。

        使用 execute_waypoint_trajectory 服务（绕过 execute_action 的
        ActionServer 故障问题）。

        参数:
            x, y, z: 目标位置（米，基座标系）
            theta_x, theta_y, theta_z: 目标姿态欧拉角（度，Tait-Bryan ZYX）
        """
        if not self.is_init_success:
            return False

        self.last_action_notif_type = None

        wp = self._fill_cartesian_waypoint(x, y, z, theta_x, theta_y, theta_z)
        req = ExecuteWaypointTrajectoryRequest()
        req.input.waypoints.append(wp)
        req.input.duration = 0
        req.input.use_optimal_blending = False

        rospy.loginfo(f"移动到: X={x:.3f} Y={y:.3f} Z={z:.3f} "
                      f"Rx={theta_x:.1f} Ry={theta_y:.1f} Rz={theta_z:.1f}")

        max_retries = 2
        for attempt in range(max_retries):
            try:
                self.execute_waypoint_trajectory(req)
                rospy.sleep(2.0)
                return True
            except rospy.ServiceException as e:
                rospy.logwarn(f"笛卡尔移动失败 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    rospy.sleep(2.0)

        rospy.logerr(f"笛卡尔移动失败，已重试 {max_retries} 次。")
        return False

    def move_gripper(self, value):
        """
        控制夹爪开合。

        参数:
            value: 0.0 = 完全张开, 1.0 = 完全闭合
        """
        if not self.is_gripper_present:
            rospy.logwarn("没有夹爪，跳过控制。")
            return True

        req = SendGripperCommandRequest()
        finger = Finger()
        finger.finger_identifier = 0
        finger.value = float(value)
        req.input.gripper.finger.append(finger)
        req.input.mode = GripperMode.GRIPPER_POSITION

        rospy.loginfo(f"夹爪 -> {value * 100:.0f}%")
        try:
            self.send_gripper_command(req)
            time.sleep(0.75)
            return True
        except rospy.ServiceException as e:
            rospy.logerr(f"夹爪控制失败: {e}")
            return False

    def go_home(self, home_x=0.30, home_y=0.10, home_z=0.25,
                home_rx=0.0, home_ry=180.0, home_rz=45.0):
        """移动机械臂到预定义的安全/home 笛卡尔位姿。"""
        rospy.loginfo("正在移动到 home 位置...")
        return self.move_to_cartesian_pose(home_x, home_y, home_z,
                                            home_rx, home_ry, home_rz)
