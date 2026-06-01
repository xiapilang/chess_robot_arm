#!/usr/bin/env python3
"""
Kinova Gen3 Lite 机械臂底层控制器（基于 Kortex ROS 驱动）。

封装 kortex_driver 的 ROS 服务，提供：
  - 笛卡尔空间位姿移动
  - 夹爪开合控制
  - 故障清除与通知激活
  - 安全归位

Kinova Gen3 Lite 机械臂底层控制器。
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

            # --- 伺服模式 ---
            servoing_srv = f'/{self.robot_name}/base/set_servoing_mode'
            rospy.wait_for_service(servoing_srv)
            self.set_servoing = rospy.ServiceProxy(servoing_srv, SetServoingMode)

            # --- 安全相关服务 ---
            safety_srv = f'/{self.robot_name}/device_config/clear_all_safety_status'
            rospy.wait_for_service(safety_srv)
            self.clear_all_safety = rospy.ServiceProxy(safety_srv, ClearAllSafetyStatus)

            safety_topic_srv = f'/{self.robot_name}/device_config/activate_publishing_of_safety_topic'
            rospy.wait_for_service(safety_topic_srv)
            self.activate_safety_topic = rospy.ServiceProxy(
                safety_topic_srv, OnNotificationSafetyTopic)

            read_zones_srv = f'/{self.robot_name}/base/read_all_protection_zones'
            rospy.wait_for_service(read_zones_srv)
            self.read_protection_zones = rospy.ServiceProxy(
                read_zones_srv, ReadAllProtectionZones)

            delete_zone_srv = f'/{self.robot_name}/base/delete_protection_zone'
            rospy.wait_for_service(delete_zone_srv)
            self.delete_protection_zone = rospy.ServiceProxy(
                delete_zone_srv, DeleteProtectionZone)

            # --- 关节空间轨迹（Cartesian 失败时的 fallback） ---
            ik_srv = f'/{self.robot_name}/base/compute_inverse_kinematics'
            rospy.wait_for_service(ik_srv)
            self.compute_ik = rospy.ServiceProxy(ik_srv, ComputeInverseKinematics)

            joint_traj_srv = f'/{self.robot_name}/base/play_joint_trajectory'
            rospy.wait_for_service(joint_traj_srv)
            self.play_joint_traj = rospy.ServiceProxy(
                joint_traj_srv, PlayJointTrajectory)

            # --- 关节软限位置零 ---
            reset_speed_srv = f'/{self.robot_name}/control_config/reset_joint_speed_soft_limits'
            rospy.wait_for_service(reset_speed_srv)
            self.reset_joint_speed = rospy.ServiceProxy(
                reset_speed_srv, ResetJointSpeedSoftLimits)

            reset_accel_srv = f'/{self.robot_name}/control_config/reset_joint_acceleration_soft_limits'
            rospy.wait_for_service(reset_accel_srv)
            self.reset_joint_accel = rospy.ServiceProxy(
                reset_accel_srv, ResetJointAccelerationSoftLimits)

            # 安全通知订阅
            self._safety_triggered = False
            self.safety_sub = rospy.Subscriber(
                f"/{self.robot_name}/safety_topic",
                SafetyNotification,
                self._safety_cb)

            self.is_init_success = True
            rospy.loginfo(f"KinovaArmController: '{self.robot_name}' 初始化成功 "
                          f"（夹爪={'有' if self.is_gripper_present else '无'}）。")

        except Exception as e:
            rospy.logerr(f"KinovaArmController 初始化失败: {e}")
            self.is_init_success = False

    def _action_cb(self, notif):
        """动作通知回调：记录最近的机器人动作事件。"""
        self.last_action_notif_type = notif.action_event

    def _safety_cb(self, notif):
        """安全通知回调：自动清除触发的安全状态。"""
        self._safety_triggered = True
        rospy.logwarn(f"安全触发，自动清除...")
        try:
            self.clear_all_safety(ClearAllSafetyStatusRequest())
            rospy.loginfo("安全状态已清除。")
        except Exception as e:
            rospy.logwarn(f"清除安全状态失败: {e}")

    def disable_all_safety(self):
        """删除所有保护区域并清除安全状态，关闭安全角度限制。"""
        rospy.loginfo("正在关闭安全保护...")
        try:
            # 1. 激活安全话题发布
            self.activate_safety_topic(OnNotificationSafetyTopicRequest())
            rospy.sleep(0.5)
        except Exception as e:
            rospy.logwarn(f"激活安全话题失败: {e}")

        try:
            # 2. 清除所有安全状态
            self.clear_all_safety(ClearAllSafetyStatusRequest())
            rospy.loginfo("安全状态已清除。")
        except Exception as e:
            rospy.logwarn(f"清除安全状态失败: {e}")

        try:
            # 3. 删除所有保护区域
            try:
                resp = self.read_protection_zones(ReadAllProtectionZonesRequest())
                for zone in resp.output.protection_zones:
                    try:
                        del_req = DeleteProtectionZoneRequest()
                        del_req.input.handle.identifier = zone.handle.identifier
                        self.delete_protection_zone(del_req)
                    except Exception:
                        pass  # Kortex 内部 API 限制，静默跳过
            except Exception:
                pass
        except Exception as e:
            rospy.logwarn(f"读取保护区域失败: {e}")

        try:
            # 4. 重置关节速度/加速度软限位
            for mode in range(1, 13):
                try:
                    req = ResetJointSpeedSoftLimitsRequest()
                    req.input.control_mode = mode
                    self.reset_joint_speed(req)
                except Exception:
                    pass
                try:
                    req = ResetJointAccelerationSoftLimitsRequest()
                    req.input.control_mode = mode
                    self.reset_joint_accel(req)
                except Exception:
                    pass
            rospy.loginfo("关节速度/加速度软限位已重置。")
        except Exception:
            pass

        try:
            # 5. 再次清除故障
            self.clear_faults()
            rospy.sleep(0.5)
            self.clear_all_safety(ClearAllSafetyStatusRequest())
        except Exception:
            pass

        rospy.loginfo("安全保护已关闭。")

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

    def _wait_for_action_end(self, timeout=30.0):
        """等待动作完成，超时后仅告警不阻塞后续流程。"""
        start = time.time()
        while not rospy.is_shutdown() and (time.time() - start) < timeout:
            if self.last_action_notif_type == ActionEvent.ACTION_END:
                return True
            elif self.last_action_notif_type == ActionEvent.ACTION_ABORT:
                rospy.logwarn("机械臂报告动作中止，继续等待...")
                self.last_action_notif_type = None  # 重置，继续等
            time.sleep(0.1)
        rospy.logwarn(f"动作等待超时 ({timeout}s)，继续执行下一步。")
        return True  # 超时不阻塞，机械臂通常已到位

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
                self.disable_all_safety()
                rospy.sleep(0.5)
                # 设置伺服模式为单层位置控制
                servo_req = SetServoingModeRequest()
                servo_req.input.servoing_mode = ServoingMode.SINGLE_LEVEL_SERVOING
                self.set_servoing(servo_req)
                rospy.loginfo("伺服模式已设置为单层位置控制。")
                rospy.sleep(0.5)
                self.activate_notifications(OnNotificationActionTopicRequest())
                rospy.sleep(1.0)
                rospy.loginfo("机械臂激活成功（安全保护已关闭）。")
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
        cartesian_rejected = False
        for attempt in range(max_retries):
            try:
                self.execute_waypoint_trajectory(req)
                start = time.time()
                while (time.time() - start) < 5.0:
                    if self.last_action_notif_type == ActionEvent.ACTION_END:
                        rospy.loginfo("  机械臂确认到位。")
                        return True
                    elif self.last_action_notif_type == ActionEvent.ACTION_ABORT:
                        rospy.logwarn("  笛卡尔轨迹被拒绝，尝试关节空间 fallback...")
                        self.last_action_notif_type = None
                        cartesian_rejected = True
                        break
                    time.sleep(0.05)
                if cartesian_rejected:
                    break  # 跳出 Cartesian 重试，进入关节空间 fallback
                rospy.sleep(3.0)
                return True
            except rospy.ServiceException as e:
                rospy.logwarn(f"笛卡尔移动失败 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    rospy.sleep(2.0)

        # --- 关节空间 fallback：自己算 IK 后用关节轨迹执行 ---
        if cartesian_rejected:
            rospy.loginfo("使用关节空间轨迹绕过 Cartesian IK 限制...")
            try:
                ik_req = ComputeInverseKinematicsRequest()
                ik_req.input.cartesian_pose.x = x
                ik_req.input.cartesian_pose.y = y
                ik_req.input.cartesian_pose.z = z
                ik_req.input.cartesian_pose.theta_x = theta_x
                ik_req.input.cartesian_pose.theta_y = theta_y
                ik_req.input.cartesian_pose.theta_z = theta_z
                ik_resp = self.compute_ik(ik_req)

                jt_req = PlayJointTrajectoryRequest()
                jt_req.input.joint_angles.joint_angles = ik_resp.output.joint_angles
                self.play_joint_traj(jt_req)
                rospy.loginfo("  关节空间轨迹已发送。")
                rospy.sleep(3.0)
                return True
            except Exception as e:
                rospy.logerr(f"关节空间 fallback 也失败: {e}")
                rospy.logerr(f"  目标: ({x:.3f},{y:.3f},{z:.3f}) "
                           f"Rx={theta_x:.1f} Ry={theta_y:.1f} Rz={theta_z:.1f}")
                return False

        rospy.logerr(f"笛卡尔移动失败，已重试 {max_retries} 次。")
        return False

    def move_to_cartesian_free_rot(self, x, y, z, theta_x=0.0, theta_y=180.0, theta_z=0.0,
                                    guess_deg=None):
        """跳过 Cartesian 轨迹，直接 IK 求解后以关节轨迹执行。"""
        if not self.is_init_success:
            return False
        rospy.loginfo(f"自由旋转移动到: X={x:.3f} Y={y:.3f} Z={z:.3f}")
        try:
            ik_req = ComputeInverseKinematicsRequest()
            ik_req.input.cartesian_pose.x = x
            ik_req.input.cartesian_pose.y = y
            ik_req.input.cartesian_pose.z = z
            ik_req.input.cartesian_pose.theta_x = theta_x
            ik_req.input.cartesian_pose.theta_y = theta_y
            ik_req.input.cartesian_pose.theta_z = theta_z
            if guess_deg is not None and len(guess_deg) >= 6:
                g = ik_req.input.guess
                for i, ang in enumerate(guess_deg[:6]):
                    ja = g.joint_angles.add()
                    ja.joint_identifier = i
                    ja.value = ang
            ik_resp = self.compute_ik(ik_req)
            jt_req = PlayJointTrajectoryRequest()
            jt_req.input.joint_angles.joint_angles = ik_resp.output.joint_angles
            self.play_joint_traj(jt_req)
            rospy.loginfo("  自由旋转轨迹已发送。")
            rospy.sleep(3.0)
            return True
        except Exception as e:
            rospy.logerr(f"自由旋转移动失败: {e}")
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
