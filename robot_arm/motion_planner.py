#!/usr/bin/env python3
"""
运动规划节点：8×8 国际象棋棋子的抓取-放置（Pick-and-Place）操作。

整合视觉坐标（相机坐标系）、正运动学（FK）和底层机械臂控制，
将来自 AI 协调器的移动指令转换为完整的码放棋子动作序列。

ROS 节点：订阅 /kinova_pick_place/goal_in_camera 话题，
         发布 /my_gen3_lite/arm_status 机械臂状态。

机械臂状态:
  0 = IDLE（空闲，等待指令）
  1 = BUSY（执行中，拒绝新指令）
  2 = ERROR（故障，需恢复）
"""

import rospy
import numpy as np
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
from kortex_driver.msg import BaseCyclic_Feedback
from std_msgs.msg import String
from chess_robot_arm.msg import PickAndPlaceGoalInCamera
from chess_robot_arm.utils.constants import POST_CALIB_HOME, PRE_CALIB_HOME


class MotionPlanner:
    """
    棋子抓取-放置运动规划器。

    执行流程:
      1. 收到目标消息后，获取当前 T_base_camera
      2. 将相机坐标系下的抓取/放置点转换为基座标系
      3. 依次执行抓取（PICK）和放置（PLACE）动作序列
      4. 每次动作后返回 home 安全位姿
    """

    STATE_IDLE  = 0      # 空闲
    STATE_BUSY  = 1      # 忙碌
    STATE_ERROR = 2      # 故障

    MSG_IDLE  = "0"
    MSG_BUSY  = "1"
    MSG_ERROR = "2"

    def __init__(self, robot_name="my_gen3_lite"):
        rospy.init_node('chess_motion_planner', anonymous=False)

        from chess_robot_arm.robot_arm.arm_controller import KinovaArmController
        from chess_robot_arm.robot_arm.kinematics import forward_kinematics

        self.robot_name = robot_name
        self.state = self.STATE_IDLE

        # --- 机械臂底层控制器 ---
        self.arm = KinovaArmController(robot_name=robot_name)

        # --- 状态发布者 ---
        self.state_pub = rospy.Publisher(
            f"/{self.robot_name}/arm_status", String, queue_size=1, latch=True)

        # --- 运动学 ---
        self.dh_params = rospy.get_param("~dh_params", None)
        self.dof = rospy.get_param("~dof", 6)

        # --- 手眼变换 T_ee_camera（末端执行器 → 相机光学坐标系） ---
        trans = rospy.get_param("~T_ee_camera_translation", [0.060, -0.040, -0.110])
        rot   = rospy.get_param("~T_ee_camera_rotation",
                                [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        self.T_ee_camera = np.eye(4)
        self.T_ee_camera[:3, :3] = np.array(rot)
        # 如果平移量数值较大（>10），说明单位是毫米，需转换为米
        self.T_ee_camera[:3, 3]  = (np.array(trans) / 1000.0
                                    if max(np.abs(trans)) > 10
                                    else np.array(trans))

        # --- 运动控制参数 ---
        self.gripper_orient_deg = np.array(rospy.get_param(
            "~gripper_orientation_deg", [0.0, 180.0, 45.0]))
        self.pre_action_lift = rospy.get_param("~pre_action_z_lift", 0.05)
        self.pick_z_offset   = rospy.get_param("~pick_z_offset", 0.005)
        self.place_z_offset  = rospy.get_param("~place_z_offset", 0.01)
        self.gripper_open_val  = rospy.get_param("~gripper_open", 0.66)
        self.gripper_close_val = rospy.get_param("~gripper_close", 0.96)
        self.use_base_frame_coords = rospy.get_param("~use_base_frame_coords", True)

        # --- 获取当前关节状态 ---
        self.current_joints_rad = None
        self.joint_names = []
        self._setup_joint_feedback()

        # --- 订阅 AI 协调器发布的抓取放置目标 ---
        self.goal_sub = rospy.Subscriber(
            "/kinova_pick_place/goal_in_camera",
            PickAndPlaceGoalInCamera,
            self._goal_callback,
            queue_size=1)

        # --- 订阅归位指令 ---
        self.home_sub = rospy.Subscriber(
            "/kinova_pick_place/home_cmd", String, self._home_cb, queue_size=1)

        # --- 激活机械臂 ---
        if not self.arm.is_init_success:
            rospy.logerr("机械臂控制器初始化失败。")
            self._set_state(self.STATE_ERROR)
            return

        if not self.arm.activate():
            rospy.logerr("机械臂激活失败。")
            self._set_state(self.STATE_ERROR)
            return

        # --- 初始归位（先到中间点避免相机碰撞，再到标定后避让位置） ---
        rospy.loginfo("正在执行初始归位...")
        self._set_state(self.STATE_BUSY)
        if not self._go_pre_calib_home():
            rospy.logwarn("移动至中间点失败。")
            self._set_state(self.STATE_ERROR)
        elif not self._go_post_calib_home():
            rospy.logwarn("归位失败。")
            self._set_state(self.STATE_ERROR)
        else:
            self._set_state(self.STATE_IDLE)
            rospy.loginfo("运动规划器就绪，等待目标指令: "
                          "/kinova_pick_place/goal_in_camera")

    def _setup_joint_feedback(self):
        """订阅关节状态反馈。优先使用 BaseCyclic_Feedback。"""
        self.feedback_sub = rospy.Subscriber(
            f"/{self.robot_name}/base_feedback",
            BaseCyclic_Feedback,
            self._base_feedback_cb, queue_size=1)

        rospy.loginfo("等待初始关节状态...")
        rate = rospy.Rate(10)
        for _ in range(50):
            if self.current_joints_rad is not None:
                break
            rate.sleep()

        if self.current_joints_rad is None:
            rospy.logwarn("BaseCyclic_Feedback 超时，尝试 JointState...")
            self.feedback_sub.unregister()
            self.joint_names = rospy.get_param(
                f"/{self.robot_name}/joint_names",
                [f"joint_{i+1}" for i in range(self.dof)])
            self.joint_sub = rospy.Subscriber(
                f"/{self.robot_name}/joint_states",
                JointState,
                self._joint_state_cb, queue_size=1)
            for _ in range(50):
                if self.current_joints_rad is not None:
                    break
                rate.sleep()

        if self.current_joints_rad is None:
            rospy.logerr("无法获取关节状态！")

    def _base_feedback_cb(self, msg):
        if len(msg.actuators) >= self.dof:
            self.current_joints_rad = [
                np.deg2rad(msg.actuators[i].position) for i in range(self.dof)]

    def _joint_state_cb(self, msg):
        if not self.joint_names:
            return
        angles = dict(zip(msg.name, msg.position))
        try:
            self.current_joints_rad = [angles[n] for n in self.joint_names]
        except KeyError:
            self.current_joints_rad = None

    def _set_state(self, new_state):
        """更新并发布机械臂状态。"""
        if new_state != self.state:
            self.state = new_state
            mapping = {self.STATE_IDLE: self.MSG_IDLE,
                       self.STATE_BUSY: self.MSG_BUSY,
                       self.STATE_ERROR: self.MSG_ERROR}
            msg_str = mapping.get(new_state, "UNKNOWN")
            rospy.loginfo(f"机械臂状态 -> {msg_str}")
            self.state_pub.publish(String(data=msg_str))

    def _get_T_base_camera(self):
        """使用正运动学计算当前的 T_base_camera。"""
        if self.current_joints_rad is None:
            rospy.logerr("无关节状态。")
            return None
        from chess_robot_arm.robot_arm.kinematics import forward_kinematics
        T_base_ee = forward_kinematics(self.current_joints_rad)
        return T_base_ee @ self.T_ee_camera

    def _camera_to_base(self, point_cam, T_base_cam):
        """将相机坐标系下的 Point 转换到基座标系。"""
        p_cam = np.array([point_cam.x, point_cam.y, point_cam.z, 1.0])
        p_base = T_base_cam @ p_cam
        return p_base[:3]

    def _pick_or_place(self, action_name, target_base, z_offset, is_pick):
        """
        执行单次抓取或放置动作序列:

        步骤:
          1. 移动到目标点上方（预动作位置）
          2. 如果是抓取：张开夹爪
          3. 下降到精确的目标 Z 位置
          4. 闭合（抓取）或张开（放置）夹爪
          5. 抬升离开目标点
        """
        rospy.loginfo(f"--- {action_name} 序列开始 ---")
        rx, ry, rz = self.gripper_orient_deg

        target_z = target_base[2] + z_offset
        above_x, above_y = target_base[0], target_base[1]
        above_z = target_z + self.pre_action_lift

        # 步骤 1: 移动到目标点上方
        rospy.loginfo(f"  步骤1: 移动到{action_name}点上方 "
                      f"({above_x:.3f}, {above_y:.3f}, {above_z:.3f})")
        if not self.arm.move_to_cartesian_pose(above_x, above_y, above_z, rx, ry, rz):
            rospy.logwarn(f"  移动到{action_name}点上方失败。")
            return False

        # 步骤 2: 抓取前张开夹爪
        if is_pick and self.arm.is_gripper_present:
            rospy.loginfo(f"  步骤2: 张开夹爪 ({self.gripper_open_val*100:.0f}%)")
            self.arm.move_gripper(self.gripper_open_val)
            rospy.sleep(0.5)

        # 步骤 3: 下降到精确 Z 位置
        rospy.loginfo(f"  步骤3: 下降到{action_name} Z "
                      f"({target_base[0]:.3f}, {target_base[1]:.3f}, {target_z:.3f})")
        if not self.arm.move_to_cartesian_pose(target_base[0], target_base[1],
                                                target_z, rx, ry, rz):
            rospy.logwarn(f"  下降到{action_name} Z 失败。")
            return False
        rospy.sleep(0.3)

        # 步骤 4: 驱动夹爪
        if self.arm.is_gripper_present:
            val = self.gripper_close_val if is_pick else self.gripper_open_val
            word = "闭合" if is_pick else "张开"
            rospy.loginfo(f"  步骤4: {word}夹爪 ({val*100:.0f}%)")
            self.arm.move_gripper(val)
            rospy.sleep(1.0)

        # 步骤 5: 抬升离开
        rospy.loginfo(f"  步骤5: 抬升 ({above_x:.3f}, {above_y:.3f}, {above_z:.3f})")
        if not self.arm.move_to_cartesian_pose(above_x, above_y, above_z, rx, ry, rz):
            rospy.logwarn(f"  从{action_name}点抬升失败。")
            return False

        rospy.loginfo(f"--- {action_name} 序列完成 ---")
        return True

    def _go_pre_calib_home(self):
        """初次启动时先移动到中间安全点，避免相机与机械臂碰撞。"""
        rospy.loginfo("先移动到中间安全点（避开相机）...")
        return self.arm.go_home(
            home_x=PRE_CALIB_HOME["x"],
            home_y=PRE_CALIB_HOME["y"],
            home_z=PRE_CALIB_HOME["z"],
            home_rx=PRE_CALIB_HOME["rx"],
            home_ry=PRE_CALIB_HOME["ry"],
            home_rz=PRE_CALIB_HOME["rz"],
        )

    def _go_post_calib_home(self):
        """使用 POST_CALIB_HOME 参数归位。"""
        return self.arm.go_home(
            home_x=POST_CALIB_HOME["x"],
            home_y=POST_CALIB_HOME["y"],
            home_z=POST_CALIB_HOME["z"],
            home_rx=POST_CALIB_HOME["rx"],
            home_ry=POST_CALIB_HOME["ry"],
            home_rz=POST_CALIB_HOME["rz"],
        )

    def _home_cb(self, msg):
        """归位指令回调：收到 HOME 指令时将机械臂移到标定后避让位置。"""
        if self.state != self.STATE_IDLE:
            rospy.logwarn(f"机械臂忙，忽略归位指令（状态={self.state}）")
            return
        rospy.loginfo("收到归位指令，正在移动到标定后避让位置...")
        self._set_state(self.STATE_BUSY)
        if self._go_post_calib_home():
            self._set_state(self.STATE_IDLE)
        else:
            self._set_state(self.STATE_ERROR)

    def _goal_callback(self, msg):
        """
        接收来自 AI 协调器的抓取放置目标并执行。

        当机械臂 BUSY 时忽略新目标；ERROR 状态下也忽略。
        """
        if self.state == self.STATE_BUSY:
            rospy.logwarn("机械臂忙碌中，忽略新目标。")
            return
        if self.state == self.STATE_ERROR:
            rospy.logwarn("机械臂故障状态，忽略新目标。")
            return

        self._set_state(self.STATE_BUSY)

        rospy.loginfo(f"收到目标: 抓取 '{msg.object_id_at_pick}' "
                      f"-> 放置 '{msg.target_location_id_at_place}'")

        if self.use_base_frame_coords:
            # 坐标已经是基座标系，直接使用
            pick_base = np.array([msg.pick_position_in_camera.x,
                                  msg.pick_position_in_camera.y,
                                  msg.pick_position_in_camera.z])
            place_base = np.array([msg.place_position_in_camera.x,
                                   msg.place_position_in_camera.y,
                                   msg.place_position_in_camera.z])
            rospy.loginfo(f"  抓取点 (基座标系直出): {np.round(pick_base, 3)}")
            rospy.loginfo(f"  放置点 (基座标系直出): {np.round(place_base, 3)}")
        else:
            # 相机系 → 基座标系变换
            T_cam = self._get_T_base_camera()
            if T_cam is None:
                rospy.logerr("无法获取 T_base_camera，中止操作。")
                self._go_post_calib_home()
                self._set_state(self.STATE_ERROR)
                return
            pick_base  = self._camera_to_base(msg.pick_position_in_camera, T_cam)
            place_base = self._camera_to_base(msg.place_position_in_camera, T_cam)
            rospy.loginfo(f"  抓取点 (基座标系): {np.round(pick_base, 3)}")
            rospy.loginfo(f"  放置点 (基座标系): {np.round(place_base, 3)}")

        # 执行抓取
        if not self._pick_or_place("抓取", pick_base, self.pick_z_offset, True):
            rospy.logerr("抓取失败。")
            self._go_post_calib_home()
            self._set_state(self.STATE_IDLE)
            return

        # 执行放置
        if not self._pick_or_place("放置", place_base, self.place_z_offset, False):
            rospy.logerr("放置失败。")
            self._go_post_calib_home()
            self._set_state(self.STATE_IDLE)
            return

        # 返回 home
        self._go_post_calib_home()
        self._set_state(self.STATE_IDLE)
        rospy.loginfo("抓取放置完成。机械臂空闲。")

    def run(self):
        """主运行循环（阻塞在 rospy.spin）。"""
        if self.state == self.STATE_ERROR:
            rospy.logerr("运动规划器处于故障状态，无法运行。")
            return
        rospy.loginfo("运动规划器运行中...")
        rospy.spin()


if __name__ == '__main__':
    try:
        planner = MotionPlanner()
        planner.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"运动规划器未处理异常: {e}")
        import traceback
        traceback.print_exc()
