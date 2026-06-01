# 国际象棋机械臂对弈系统

基于 Kinova Gen3 Lite 六自由度机械臂 + Intel RealSense 深度相机 + ROS 1 的国际象棋人机对弈系统。AI 引擎使用 python-chess + Stockfish，棋盘状态通过终端手动输入。

---

## 运动学基础

### DH 参数法

机械臂运动学采用标准 **DH 参数（Craig 约定）** 建模。每个关节由四个参数描述：

| 参数 | 含义 |
|------|------|
| `alpha_{i-1}` | 绕 X_{i-1} 轴旋转，使 Z_{i-1} 与 Z_i 平行 |
| `a_{i-1}` | 沿 X_{i-1} 轴平移，使 Z_{i-1} 与 Z_i 共线 |
| `d_i` | 沿 Z_i 轴平移，使 X_{i-1} 与 X_i 共线 |
| `theta_offset_i` | 绕 Z_i 轴旋转（关节零位补偿） |

相邻坐标系间的齐次变换矩阵：

```
T_i = [
    [ cos(θ),            -sin(θ),            0,          a       ],
    [ sin(θ)cos(α),       cos(θ)cos(α),     -sin(α),    -sin(α)d ],
    [ sin(θ)sin(α),       cos(θ)sin(α),      cos(α),     cos(α)d ],
    [ 0,                   0,                 0,          1       ]
]
```

其中 `θ = q_i + theta_offset_i`（当前关节角 + DH 零位补偿）。

### 正运动学 (FK)

从基座到末端的变换通过逐级连乘 DH 矩阵得到：

```
T_base_ee = T_1 · T_2 · T_3 · T_4 · T_5 · T_6
```

给定 6 个关节角度 `[q1..q6]`（弧度），FK 输出末端执行器在基座标系中的 4×4 齐次位姿矩阵。实现见 [kinematics.py](chess_robot_arm/robot_arm/kinematics.py#L38)。

**手眼变换**：相机安装在末端执行器上（眼在手），相机系到基座系的变换为：

```
T_base_camera = T_base_ee · T_ee_camera
```

其中 `T_ee_camera` 为末端到相机的固定变换（手动标定，见 [constants.py](chess_robot_arm/utils/constants.py#L66)）。

### 逆运动学 (IK)

给定目标位姿 `T_target`，求解对应的 6 个关节角度。本系统采用 **SciPy SLSQP 数值优化** 方法：

```
minimize:  w_pos · ||pos_err||² + w_ori · ||ori_err||²
subject to:  q_min ≤ q ≤ q_max（关节限位）
```

- 位置误差：`T_target[:3,3] - T_current[:3,3]`
- 姿态误差：将旋转矩阵误差转换为轴角向量 `rotvec`
- 位置权重 `w_pos=1.0`，姿态权重 `w_ori=0.5`（位置精度优先）
- 优化变量初始值使用当前关节角或零位

实现见 [kinematics.py](chess_robot_arm/robot_arm/kinematics.py#L76)。

**实际使用中**，优先通过 Kortex 驱动自带的 Cartesian 轨迹服务执行移动。仅当 Cartesian 轨迹被拒绝（如目标超出可达工作空间、旋转角度过大）时，才降级为 IK + 关节空间轨迹执行（见 [arm_controller.py](chess_robot_arm/robot_arm/arm_controller.py#L262)）。

### 棋盘坐标定位

系统不使用手眼标定来转换坐标，而是通过 **ArUco + 手动锚点 + 单应性变换** 直接获取每个格子的基座标系坐标：

```
ArUco 检测 → 相机帧四角 XY
                  ↓
    4 对点 (相机 XY → 基座 XY) → 单应性矩阵 (3×3)
                  ↓
    双线性插值 + 透视变换 → 基座帧 3D 点
```

- ArUco 标签贴在棋盘四角（ID 0=左上 h8, 1=右上 a8, 2=左下 h1, 3=右下 a1）
- `BOARD_CORNERS_BASE` 为手动测量的四角基座坐标，提供绝对参考锚点
- 单应性变换可处理棋盘与机械臂之间的透视畸变

---

## 硬件依赖

| 设备 | 型号 |
|------|------|
| 机械臂 | Kinova Gen3 Lite (6-DOF) |
| 深度相机 | Intel RealSense D435/D435i (眼在手) |
| 夹爪 | Kinova 原装二指夹爪 |
| ArUco 标签 | 4 个 (ID 0-3)，边长 6cm |
| 棋盘 | 8×8 标准棋盘 (~40mm 格子) |

## 软件依赖

- ROS 1 (Noetic) + `kortex_driver`
- `pyrealsense2`、`opencv-contrib-python`
- `python-chess`、`numpy`、`scipy`
- [Stockfish](https://stockfishchess.org/) 引擎（推荐，也可用内置降级 AI）
- 不再依赖 `ultralytics` / `torch`（YOLO 已移除）

---

## 快速启动

```bash
# 1. 启动 Kortex 驱动
roslaunch kortex_driver kortex_driver.launch ip_address:=192.168.1.10
arm:=gen3_lite

# 2. 启动 ArUco 检测 + 运动规划器
roslaunch chess_robot_arm chess_robot.launch

# 3. 启动 AI 协调器（终端交互）
rosrun chess_robot_arm main.py _stockfish_path:=/usr/games/stockfish _think_time:=3.0
```

## 分步启动

**1) 启动机械臂并归位**
```bash
roslaunch kortex_driver kortex_driver.launch ip_address:=192.168.1.10 arm:=gen3_lite
rosrun chess_robot_arm motion_planner.py
```

**2) 启动 ArUco 棋盘检测**
```bash
rosrun chess_robot_arm board_detector.py
```

**3) 启动对弈协调器**
```bash
rosrun chess_robot_arm main.py _stockfish_path:=/usr/games/stockfish _think_time:=3.0
```

启动时序：`motion_planner`（归位）→ `board_detector`（ArUco 检测）→ `main.py`（等待输入）。

---

## 棋盘标定

棋盘首次放置或移动后需要标定。用夹爪末端依次对准棋盘四角表面，记录基座坐标，填入 [constants.py](chess_robot_arm/utils/constants.py) 的 `BOARD_CORNERS_BASE`：

```python
BOARD_CORNERS_BASE = {
    "top_left":     {"x": ..., "y": ..., "z": -0.011},  # h8
    "top_right":    {"x": ..., "y": ..., "z": -0.011},  # a8
    "bottom_left":  {"x": ..., "y": ..., "z": -0.011},  # h1
    "bottom_right": {"x": ..., "y": ..., "z": -0.011},  # a1
}
```

四角 z 统一取棋盘表面高度。修改后 `catkin_make` 重新编译。

---

## 对弈流程

1. 人类执白先走，在实体棋盘上移动棋子
2. 在终端输入走法（支持 UCI `e2e4`、SAN `e4`、FEN `fen:...`）
3. AI 通过 Stockfish 计算应招
4. 机械臂自动执行抓取-放置动作
5. 终端打印更新后的棋盘，循环直到对局结束
6. 输入 `quit` 退出

---

## 机械臂状态

| 状态码 | 含义 |
|--------|------|
| `0` | IDLE — 空闲，可接受指令 |
| `1` | BUSY — 执行中，忽略新指令 |
| `2` | ERROR — 故障，需清除后恢复 |

---

## 故障排查

**相机无法打开：** `rs-enumerate-devices` 检查连接，`pkill -f realsense` 释放占用。

**Stockfish 未找到：** `sudo apt install stockfish` 或指定 `_stockfish_path:=/path/to/stockfish`。

**机械臂连接失败：** 确认 `kortex_driver` 已安装，`ping <robot_ip>` 检查网络。

**Kortex ActionServer 报错：** 等待驱动完全就绪（`/my_gen3_lite/base_feedback` 有数据）后再启动 motion_planner。
