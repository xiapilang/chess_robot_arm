#!/usr/bin/env python3
"""
Kinova Gen3 Lite 六自由度机械臂运动学（DH 参数法）。

正运动学 (FK): 关节角度 → 末端执行器位姿（4×4 齐次变换矩阵）
逆运动学 (IK): 末端执行器位姿 → 关节角度（基于 SciPy SLSQP 数值优化）

DH 参数采用标准 Craig 约定: [alpha_{i-1}, a_{i-1}, d_i, theta_offset_i]
角度单位：弧度，长度单位：米
"""

import numpy as np
from numpy import pi
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize

from chess_robot_arm.utils.constants import DH_PARAMETERS, NUM_JOINTS


def dh_transform(alpha_prev, a_prev, d, theta):
    """
    标准 DH 变换矩阵（Craig 约定）。

    计算从坐标系 {i-1} 到坐标系 {i} 的齐次变换矩阵。
    """
    return np.array([
        [ np.cos(theta), -np.sin(theta), 0, a_prev],
        [ np.sin(theta) * np.cos(alpha_prev),
          np.cos(theta) * np.cos(alpha_prev), -np.sin(alpha_prev),
         -np.sin(alpha_prev) * d],
        [ np.sin(theta) * np.sin(alpha_prev),
          np.cos(theta) * np.sin(alpha_prev),  np.cos(alpha_prev),
          np.cos(alpha_prev) * d],
        [0, 0, 0, 1]
    ])


def forward_kinematics(q, dh_params=None):
    """
    正运动学计算：关节角度 → 末端执行器位姿。

    参数:
        q: 关节角度列表（弧度）
        dh_params: DH 参数表，格式 [alpha_{i-1}, a_{i-1}, d_i, theta_offset_i]

    返回:
        T_end_effector: 4×4 齐次变换矩阵（末端执行器在基座标系中的位姿，单位：米）
    """
    if dh_params is None:
        dh_params = DH_PARAMETERS

    if len(q) != len(dh_params):
        raise ValueError(f"关节数量不匹配: {len(q)} vs {len(dh_params)} 行 DH 参数")

    T = np.eye(4)
    for i in range(len(q)):
        alpha, a, d, theta_offset = dh_params[i]
        theta = q[i] + theta_offset
        T = T @ dh_transform(alpha, a, d, theta)

    return T


def get_pose_error(T_target, T_current):
    """
    计算目标位姿与当前位姿之间的误差。

    返回: [位置误差(3,), 姿态误差_旋转向量(3,)]，形状 (6,1)
    """
    pos_err = T_target[:3, 3] - T_current[:3, 3]
    R_err = T_target[:3, :3] @ T_current[:3, :3].T
    ori_err = R.from_matrix(R_err).as_rotvec()
    return np.hstack((pos_err, ori_err)).reshape(6, 1)


def ik_objective(q, T_target, dh_params, pos_weight, ori_weight):
    """
    逆运动学优化的目标函数。

    最小化位置误差和姿态误差的加权平方和。
    """
    T_current = forward_kinematics(q, dh_params)
    err = get_pose_error(T_target, T_current)
    return pos_weight * np.linalg.norm(err[:3])**2 + ori_weight * np.linalg.norm(err[3:])**2


def inverse_kinematics(T_target, initial_q, joint_limits=None,
                       pos_weight=5.0, ori_weight=1.0,
                       pos_tol=1e-4, ori_tol=1e-4,
                       max_iter=300, ftol=1e-8):
    """
    逆运动学求解器（基于 SLSQP 数值优化）。

    参数:
        T_target: 4×4 目标位姿矩阵（单位：米）
        initial_q: 初始关节角度猜测（弧度）
        joint_limits: 关节限制 [[min, max], ...]（弧度），可选
        pos_weight: 位置误差权重
        ori_weight: 姿态误差权重
        pos_tol: 位置收敛容差
        ori_tol: 姿态收敛容差（旋转向量范数）
        max_iter: 最大优化迭代次数
        ftol: 优化器目标函数值收敛阈值

    返回:
        (solution_q_rad, converged, pos_err, ori_err)
    """
    dh = DH_PARAMETERS
    n_dof = len(initial_q) if initial_q is not None else NUM_JOINTS

    if initial_q is None:
        initial_q = np.zeros(n_dof)

    # 将初始角度归一化到 [-π, π]
    q0 = np.array([np.arctan2(np.sin(v), np.cos(v)) for v in initial_q])

    # 构建优化边界约束
    bounds = None
    if joint_limits:
        bounds = []
        for i in range(n_dof):
            lo, hi = joint_limits[i]
            if lo > hi:
                lo, hi = hi, lo
            bounds.append((lo, hi))

    # SLSQP 优化
    result = minimize(
        ik_objective,
        q0,
        args=(T_target, dh, pos_weight, ori_weight),
        method='SLSQP',
        bounds=bounds,
        options={'disp': False, 'maxiter': max_iter, 'ftol': ftol},
    )

    # 解归一化
    q_sol = np.array([np.arctan2(np.sin(v), np.cos(v)) for v in result.x])

    # 验证解的质量
    T_sol = forward_kinematics(q_sol, dh)
    err = get_pose_error(T_target, T_sol)
    pos_err = np.linalg.norm(err[:3])
    ori_err = np.linalg.norm(err[3:])

    converged = result.success and (pos_err < pos_tol) and (ori_err < ori_tol)

    return q_sol, converged, pos_err, ori_err


class KinematicsCalculator:
    """运动学计算便捷封装类。"""

    def __init__(self, dh_params=None):
        self.dh = dh_params if dh_params is not None else DH_PARAMETERS
        self.dof = len(self.dh)
        self.theta_offsets = [row[3] for row in self.dh]

    def fk(self, q_rad):
        """正运动学：关节角度 → 末端位姿（米）。"""
        return forward_kinematics(q_rad, self.dh)

    def ik(self, T_target, q_init_rad=None):
        """逆运动学：目标位姿 → 关节角度。"""
        if q_init_rad is None:
            q_init_rad = np.zeros(self.dof)
        return inverse_kinematics(T_target, q_init_rad)


if __name__ == "__main__":
    print(f"运动学测试（{NUM_JOINTS} 自由度 Kinova Gen3 Lite）")
    print("DH 参数表:")
    for i, row in enumerate(DH_PARAMETERS):
        print(f"  关节 {i+1}: alpha={row[0]:.2f}, a={row[1]:.3f}m, "
              f"d={row[2]:.3f}m, theta_offset={row[3]:.2f}")

    # 正运动学测试
    q_test = np.zeros(NUM_JOINTS)
    T = forward_kinematics(q_test)
    print(f"\n零位正运动学:\n{np.round(T, 3)}")

    # 逆运动学测试
    target_pos = np.array([0.3, 0.1, 0.25])
    target_ori = R.from_euler('xyz', [0, np.pi, np.pi/4]).as_matrix()
    T_target = np.eye(4)
    T_target[:3, :3] = target_ori
    T_target[:3, 3]  = target_pos

    print(f"\n逆运动学目标:\n{np.round(T_target, 3)}")
    q_sol, ok, p_err, o_err = inverse_kinematics(T_target, np.zeros(NUM_JOINTS))
    print(f"解（度）: {np.round(np.rad2deg(q_sol), 2)}")
    print(f"收敛: {ok}, 位置误差: {p_err:.2e}, 姿态误差: {o_err:.2e}")
