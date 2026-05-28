# 国际象棋机械臂对弈系统

基于 **Kinova Gen3 Lite** 六自由度机械臂 + **Intel RealSense** 深度相机 + **ROS 1** 框架实现的国际象棋人机对弈系统。

改编自 [EE368_Project](https://github.com/Lgx521/EE368_Project)（原项目为中国象棋），将棋盘改为 8×8 国际象棋，AI 引擎替换为 python-chess + Stockfish。

**当前版本已移除 YOLO 棋子视觉识别**，棋盘状态改为用户在终端手动输入，以规避 YOLO 检测不佳的问题。

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                     RealSense 相机 (眼在手)                   │
│                          │                                   │
│                          ▼                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Node 1: board_detector.py                           │   │
│  │  ArUco 标签 (ID 0-3) 检测 → 棋盘四角 3D 定位          │   │
│  │  发布: /chessboard_corners_3d, /camera/color/image_raw│   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Node 2: main.py (TerminalChessOrchestrator)         │   │
│  │  用户在终端手动输入走法 → python-chess + Stockfish AI  │   │
│  │  发布: /kinova_pick_place/goal_in_camera               │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Node 3: motion_planner.py                           │   │
│  │  手眼坐标变换 → DH 运动学 → 笛卡尔轨迹 → 夹爪控制      │   │
│  │  执行 Pick-and-Place 码放棋子                          │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**启动顺序：** motion_planner（归位） → board_detector（ArUco） → main.py（终端输入）

**数据流：** ArUco 定位 → 终端输入走法 → AI 计算应招 → 机械臂抓取放置 → 循环

---

## 项目结构

```
chess_robot_arm/
├── README.md                          # 本文件
├── requirements.txt                   # Python 依赖列表
├── config.py                          # 集中配置管理（从 ROS param 读取，回退到默认值）
├── main.py                            # 主协调节点：终端输入 + AI 对弈中枢
│
├── vision/                            # 视觉模块
│   ├── board_detector.py              # ArUco + RealSense 8×8 棋盘角点检测
│   ├── piece_classifier.py            # [已废弃] YOLOv8 棋子分类
│   └── piece_train.py                 # [已废弃] YOLOv8 训练脚本
│
├── chess_engine/                      # 对弈引擎模块
│   ├── game_state.py                  # python-chess 棋盘状态管理与坐标转换
│   └── ai_player.py                   # Stockfish AI 引擎 + 降级备用 AI
│
├── robot_arm/                         # 机械臂控制模块
│   ├── kinematics.py                  # DH 参数正/逆运动学
│   ├── arm_controller.py              # Kinova Kortex 驱动封装
│   └── motion_planner.py              # Pick-and-Place 运动规划节点
│
├── utils/                             # 工具模块
│   ├── constants.py                   # 物理常数、DH 参数、棋子映射表
│   └── calibration.py                 # 相机-机械臂手眼标定工具
│
└── launch/
    └── chess_robot.launch             # ROS launch 文件（启动 ArUco 检测 + 运动规划器）
```

---

## 各文件功能详解

### `utils/constants.py` — 物理常数与棋子映射
- 棋盘尺寸 (8×8)、格子大小 (55mm)
- ArUco 标签 ID 约定与物理尺寸
- 12 类国际象棋棋子 YOLO class_id → FEN 字符映射
- Kinova Gen3 Lite 标准 DH 参数表 (Craig 约定)
- 手眼变换矩阵 T_ee_camera 初值
- 夹爪姿态、运动参数、Stockfish 路径等默认配置

### `config.py` — 集中配置加载器
- 从 ROS 参数服务器读取所有可覆盖参数
- 参数不存在时自动回退到 `constants.py` 中的默认值
- 提供 `get_config()` 全局单例接口

### `vision/board_detector.py` — 棋盘角点检测节点
- **订阅**: 无（直接从 RealSense 读取图像）
- **发布**:
  - `/chessboard_corners_3d` (ChessboardCorners) — 棋盘四角在相机坐标系中的 3D 坐标
  - `/chessboard_corners_2d` (ChessboardPixelCorners) — 棋盘四角的 2D 像素坐标
  - `/camera/color/image_raw` (Image) — 原始彩色图像
- 使用 ArUco 标签定位棋盘四角，通过 PnP 解算输出 3D 坐标
- 显示 OpenCV 可视化窗口（可按 `q` 或 `ESC` 退出）

### `vision/piece_classifier.py` — [已废弃]
- 原 YOLOv8 棋子分类节点，因检测效果不佳已从流水线中移除
- 保留文件以供参考，不再被 launch 文件或 main.py 调用

### `vision/piece_train.py` — [已废弃]
- 原 YOLOv8 训练脚本，保留以供参考

### `chess_engine/game_state.py` — 棋盘状态管理器
- 封装 python-chess.Board 对象
- 提供棋盘矩阵 → FEN → UCI → (row, col) 坐标之间的转换
- 走法合法性校验、被吃子计数
- 棋盘方向：row 0 = 第 8 横排（黑方底线），col 0 = a 线

### `chess_engine/ai_player.py` — AI 对弈引擎
- 优先使用 Stockfish 引擎（需单独安装）
- Stockfish 不可用时自动降级为简易 AI（优先吃子 > 将军 > 随机）
- 提供与原项目 `elephant_fish.py` 兼容的 `ai_move_from_matrix()` 接口

### `robot_arm/kinematics.py` — DH 参数运动学
- 正运动学 (FK)：关节角度 → 末端位姿（4×4 齐次变换矩阵）
- 逆运动学 (IK)：目标位姿 → 关节角度（基于 SciPy SLSQP 数值优化）
- 可直接作为独立脚本运行测试

### `robot_arm/arm_controller.py` — 机械臂底层控制器
- 封装 `kortex_driver` ROS 服务（笛卡尔移动、夹爪控制、故障清除）
- 提供 `move_to_cartesian_pose()` 和 `move_gripper()` 等基础接口

### `robot_arm/motion_planner.py` — 运动规划节点
- **订阅**: `/kinova_pick_place/goal_in_camera` (PickAndPlaceGoalInCamera)
- **发布**: `/my_gen3_lite/arm_status` (String: "0"/"1"/"2")
- 使用正运动学实时计算 T_base_camera，将相机系坐标转为基座标系
- 执行完整的 Pick-and-Place 序列：预位 → 下降 → 抓取 → 抬升 → 移动 → 放置 → 归位

### `main.py` — AI 对弈主协调节点（终端输入版）
- **订阅**: `/chessboard_corners_3d`, `/my_gen3_lite/arm_status`
- **发布**: `/kinova_pick_place/goal_in_camera`, `/ai_eat_status`
- ArUco 定位完成后，在终端打印初始棋盘摆法
- **用户在终端手动输入走法**（支持 UCI / SAN / FEN 格式）
- 调用 Stockfish 计算 AI 应招
- 通过双线性插值将棋盘 (col, row) 映射为相机坐标系 3D 坐标
- 吃子时先将对方棋子移至弃子区，再执行主走法
- 自动处理王车易位（额外移动车）、过路兵、兵升变等特殊走法
- 每次 AI 走完后打印更新后的棋盘，等待用户再次输入，循环直到对局结束

### `utils/calibration.py` — 手眼标定工具
- 交互式标定：将 ArUco 标签放在已知基座标系位置，按 `s` 记录样本
- 输出 T_camera_marker 变换矩阵，辅助优化 T_ee_camera 参数

---

## 与原 EE368 项目的对应关系

| 原始文件 (中国象棋) | 新文件 (国际象棋) | 主要改动 |
|---|---|---|
| `chessboard_detector_node.py` | `vision/board_detector.py` | 适配 8×8 棋盘，发布到 `/chessboard_corners_3d` |
| `matrix_construction.py` | `vision/piece_classifier.py` | [已废弃] 12 类棋子，输出 FEN 字符串 |
| `piece_train.py` | `vision/piece_train.py` | [已废弃] 12 类国际象棋棋子训练 |
| `elephant_fish.py` | `chess_engine/ai_player.py` | python-chess + Stockfish 替代中国象棋引擎 |
| (新) | `chess_engine/game_state.py` | 新增 python-chess 封装层 |
| `inverse_kinematics.py` | `robot_arm/kinematics.py` | 基本不变，统一代码风格 |
| `move_cartesian.py` | `robot_arm/arm_controller.py` | 基本不变，去掉测试代码 |
| `kinova_grasp.py` / `kinova_cubic_cmd_grasp.py` | `robot_arm/motion_planner.py` | 合并为统一节点，简化架构 |
| `chess_ai_node.py` | `main.py` | 8×8 棋盘，终端输入替代 YOLO 视觉 |
| `forward_kinematics.py` | (并入 `kinematics.py`) | FK/IK 合并为一个模块 |
| `board_loc.py` | (并入 `board_detector.py`) | 棋盘定位直接集成到 ArUco 检测 |
| `frame_tf.py` | (并入 `motion_planner.py`) | 坐标变换在运动规划中直接完成 |

---

## 硬件依赖

| 设备 | 型号 | 用途 |
|---|---|---|
| 机械臂 | Kinova Gen3 Lite (6-DOF) | 棋子抓取与放置 |
| 深度相机 | Intel RealSense D435/D435i | 眼在手 (eye-in-hand) 棋盘视觉 |
| 夹爪 | Kinova 原装二指夹爪 | 抓取棋子 |
| ArUco 标签 | 4 个 (ID 0-3)，50mm 边长 | 棋盘四角定位 |
| 国际象棋盘 | 8×8 标准棋盘 (~55mm 格子) | 对弈棋盘 |

---

## 软件依赖

```bash
pip install -r requirements.txt
```

关键依赖:
- **ROS 1** (Noetic/Melodic) + `kortex_driver`
- **pyrealsense2**: Intel RealSense 相机 SDK
- **opencv-contrib-python**: ArUco 标签检测
- **python-chess**: 国际象棋规则引擎
- **numpy + scipy**: 数值计算与运动学求解
- **[Stockfish](https://stockfishchess.org/)** 引擎（可选，但强烈推荐）

> 注意：当前版本不再依赖 `ultralytics` 和 `torch`，YOLO 棋子识别已被移除。

安装 Stockfish:
```bash
sudo apt install stockfish          # Ubuntu/Debian
# 或从官网下载: https://stockfishchess.org/download/
```

---

## 使用方式

### 1. 硬件准备
1. 将 RealSense 相机安装在 Kinova 机械臂末端（眼在手配置）
2. 在棋盘四个角分别粘贴 ArUco 标签 (ID 0=左上, 1=右上, 2=左下, 3=右下)
3. 将棋盘放置在机械臂工作范围内
4. 连接机械臂和相机到计算机

### 2. 相机标定（如需要）
```bash
# 运行手眼标定工具
python utils/calibration.py

# 将 ArUco 标签放在基座标系中的已知位置
# 按 's' 键记录多个样本
# 按 'q' 键输出标定结果
```

### 3. 棋盘四角基座标系标定（棋盘移动后必须重做）

系统使用 `BOARD_CORNERS_BASE`（[constants.py](chess_robot_arm/utils/constants.py)）中手动测量的棋盘四角基座标，
直接双线性插值得到每个格子的抓取/放置坐标，不再依赖 `T_ee_camera` 手眼标定。

**标定步骤:**
1. 启动 Kortex 驱动和 motion_planner 让机械臂归位
2. 手动移动夹爪末端到棋盘左上角 (a8) 格子正上方，贴到棋盘表面
3. 用 `rosservice call` 的 `execute_waypoint_trajectory` 微调位置直到准确
4. 记录基座标 `(x, y, z)`，填入 `BOARD_CORNERS_BASE["top_left"]`
5. 对右上 (h8)、左下 (a1)、右下 (h1) 重复步骤 2-4
6. 更新 `constants.py` 后重新编译

```bash
# 微调示例（逐步降低 Z）
rosservice call /my_gen3_lite/base/clear_faults "input: {}" && sleep 3
rosservice call /my_gen3_lite/base/activate_publishing_of_action_topic \
  "input: {type: 0, rate_m_sec: 0, threshold_value: 0.0}" && sleep 1
rosservice call /my_gen3_lite/base/execute_waypoint_trajectory "
input:
  waypoints:
    - name: ''
      oneof_type_of_waypoint:
        cartesian_waypoint:
          - pose: {x: 0.446, y: -0.151, z: 0.05, theta_x: 10.7, theta_y: 177.9, theta_z: 82.7}
            reference_frame: 0
            blending_radius: 0.0
  duration: 0
  use_optimal_blending: false
"
```

> 注意: `BOARD_CORNERS_BASE` 内的 z 值统一使用 `-0.011`（棋盘表面高度），
> 代码中没有 `board_z` 参数，坐标直接以基座标系发布。

### 4. 启动完整系统

```bash
# 确保 ROS master 和 Kortex 驱动正在运行
roscore &
roslaunch kortex_driver gen3_lite.launch robot_name:=my_gen3_lite

# 终端 1: 启动机械臂运动规划器 + ArUco 棋盘检测
roslaunch chess_robot_arm chess_robot.launch

# 终端 2: 启动 AI 协调器（交互输入）
rosrun chess_robot_arm main.py \
    _stockfish_path:=/usr/games/stockfish \
    _think_time:=3.0
```

**启动时序：**
```
motion_planner  ──→  机械臂激活、归位、发布 arm_status="0"
                          │
board_detector  ──→  等待 arm_status="0" → 启动相机 → ArUco 检测
                          │
main.py         ──→  等待 /chessboard_corners_3d → 打印棋盘 → 等待输入
```

### 5. 终端输入格式

AI 协调器启动后，会先在终端打印初始棋盘摆法，然后等待用户输入。支持以下格式：

| 格式 | 示例 | 说明 |
|------|------|------|
| **UCI** | `e2e4`, `g1f3`, `e7e8q` | 起点+终点坐标（4-5 字符，升变时 5 字符） |
| **SAN** | `e4`, `Nf3`, `O-O`, `exd5` | 标准代数记谱法 |
| **FEN** | `fen:rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1` | 整盘局面覆盖（以 `fen:` 开头） |
| **退出** | `quit` | 结束对弈 |

### 6. 对弈流程
1. 启动 ArUco 检测和运动规划器，机械臂自动归位
2. 启动 AI 协调器，终端打印标准开局棋盘
3. **人类（执白）先走**：在实体棋盘上手动移动棋子
4. 在 AI 协调器终端输入你刚走的走法（如 `e4` 或 `e2e4`）
5. AI 调用 Stockfish 计算应招，将指令发送给运动规划器
6. 机械臂自动执行抓取-放置动作
7. 终端打印更新后的棋盘摆法
8. 循环回到步骤 3，直到对局结束

### 7. 手动测试移动指令
```bash
# 直接向运动规划器发送移动指令（调试用）
rostopic pub -1 /kinova_pick_place/goal_in_camera \
    chess_robot_arm/PickAndPlaceGoalInCamera \
    '{
      object_id_at_pick: "piece_e2",
      pick_position_in_camera: {x: 0.0, y: 0.0, z: 0.43},
      target_location_id_at_place: "square_e4",
      place_position_in_camera: {x: 0.0, y: -0.11, z: 0.43}
    }'
```

---

## ROS 话题总览

| 话题 | 类型 | 发布者 | 订阅者 | 说明 |
|---|---|---|---|---|
| `/chessboard_corners_3d` | `ChessboardCorners` | `board_detector.py` | `main.py` | 棋盘四角 3D 坐标 (相机系) |
| `/chessboard_corners_2d` | `ChessboardPixelCorners` | `board_detector.py` | (调试用) | 棋盘四角 2D 像素坐标 |
| `/camera/color/image_raw` | `Image` | `board_detector.py` | (调试用) | 原始彩色图像 |
| `/kinova_pick_place/goal_in_camera` | `PickAndPlaceGoalInCamera` | `main.py` | `motion_planner.py` | 抓取放置目标指令（基座标系） |
| `/my_gen3_lite/arm_status` | `String` | `motion_planner.py` | `main.py` | 机械臂状态 ("0"/"1"/"2") |
| `/ai_eat_status` | `String` | `main.py` | (日志用) | 吃子状态 ("BUSY"/"IDLE"/"PROMOTION") |

> 注意：`/chess_board_matrix` 话题已废弃（YOLO 棋子识别已移除）。棋盘状态由用户在终端手动输入。

---

## 机械臂状态码

| 状态码 | 含义 | 说明 |
|---|---|---|
| `0` / `IDLE` | 空闲 | 机械臂就绪，可接受新指令 |
| `1` / `BUSY` | 忙碌 | 正在执行抓取-放置操作，忽略新指令 |
| `2` / `ERROR` | 故障 | 发生错误，需人工干预清除故障后恢复 |

---

## 配置参数

所有参数均可在 ROS launch 文件或命令行中设置，支持 `rosparam` 动态覆盖。

### 关键参数列表
| 参数 | 默认值 | 说明 |
|---|---|---|
| `~robot_name` | `my_gen3_lite` | ROS 中机械臂的名称 |
| `~stockfish_path` | `/usr/games/stockfish` | Stockfish 引擎路径 |
| `~think_time` | `3.0` | AI 每步思考时间 (秒) |
| `~use_base_frame_coords` | `true` | 直接使用基座标系坐标（跳过手眼变换） |
| `~aruco_marker_size` | `0.05` | ArUco 标签边长 (米) |
| `~pre_action_z_lift` | `0.05` | 抓取/放置前 Z 轴抬升量 (米) |
| `~pick_z_offset` | `0.005` | 抓取 Z 轴微调偏移 (米) |
| `~place_z_offset` | `0.01` | 放置 Z 轴微调偏移 (米) |
| `~gripper_open` | `0.7` | 夹爪张开量 (0-1) |
| `~gripper_close` | `0.25` | 夹爪闭合量 (0-1) |
| `~show_cv_window` | `true` | 是否显示 OpenCV 可视化窗口 |

---

## 自定义消息类型

本系统自定义的 ROS 消息（位于 `msg/` 目录）:

| 消息类型 | 字段 |
|---|---|
| `ChessboardCorners` | `Header header`, `Point top_left`, `Point top_right`, `Point bottom_left`, `Point bottom_right` |
| `ChessboardPixelCorners` | `Header header`, `Point top_left_px`, `Point top_right_px`, `Point bottom_left_px`, `Point bottom_right_px` |
| `PickAndPlaceGoalInCamera` | `string object_id_at_pick`, `Point pick_position_in_camera`, `string target_location_id_at_place`, `Point place_position_in_camera` |

---

## 故障排查

### RealSense 相机无法打开
```bash
# 检查相机连接
rs-enumerate-devices
# 确保没有其他进程占用相机
pkill -f realsense
```

### Stockfish 未找到
```bash
# 安装 Stockfish
sudo apt install stockfish
# 或指定自定义路径
rosrun chess_robot_arm main.py _stockfish_path:=/path/to/stockfish
```

### 机械臂连接失败
```bash
# 确保 kortex_driver 正确安装
rospack find kortex_driver
# 检查机械臂 IP 和网络连接
ping <robot_ip>
```

### Kortex 驱动 ActionServer 报错
```
[ERROR] Attempt to get goal status on an uninitialized ServerGoalHandle
```
此错误表示 Kortex 驱动内部 ActionServer 尚未完全初始化。确保：
1. `roslaunch kortex_driver gen3_lite.launch` 已成功启动
2. `/my_gen3_lite/base_feedback` 话题有数据输出
3. 等待驱动完全就绪后再启动 motion_planner

---

## 许可

本项目基于 EE368_Project 改编，仅供学习和研究使用。
