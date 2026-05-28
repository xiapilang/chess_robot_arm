# 国际象棋机械臂对弈系统

基于 **Kinova Gen3 Lite** 六自由度机械臂 + **Intel RealSense** 深度相机 + **ROS 1** 框架实现的国际象棋人机对弈系统。

改编自 [EE368_Project](https://github.com/Lgx521/EE368_Project)（原项目为中国象棋），将棋盘改为 8×8 国际象棋，AI 引擎替换为 python-chess + Stockfish。

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
│  │  Node 2: piece_classifier.py                         │   │
│  │  YOLOv8 棋子检测 (12 类) → 8×8 棋盘矩阵 → FEN 字符串   │   │
│  │  发布: /chess_board_matrix (FEN)                      │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Node 3: main.py (ChessAIOrchestrator)               │   │
│  │  python-chess + Stockfish AI → 走法计算 → 坐标映射     │   │
│  │  发布: /kinova_pick_place/goal_in_camera               │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Node 4: motion_planner.py                           │   │
│  │  手眼坐标变换 → DH 运动学 → 笛卡尔轨迹 → 夹爪控制      │   │
│  │  执行 Pick-and-Place 码放棋子                          │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 项目结构

```
chess_robot_arm/
├── README.md                          # 本文件
├── requirements.txt                   # Python 依赖列表
├── config.py                          # 集中配置管理（从 ROS param 读取，回退到默认值）
├── main.py                            # 主协调节点：AI 对弈中枢
│
├── vision/                            # 视觉模块
│   ├── board_detector.py              # ArUco + RealSense 8×8 棋盘角点检测
│   ├── piece_classifier.py            # YOLOv8 棋子分类 (12 类) → 发布 FEN
│   └── piece_train.py                 # YOLOv8 训练脚本
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
    └── chess_robot.launch             # ROS launch 文件（一键启动全部节点）
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

### `vision/piece_classifier.py` — 棋子分类节点
- **订阅**: `/camera/color/image_raw`, `/chessboard_corners_3d`
- **发布**: `/chess_board_matrix` (String) — FEN 格式的棋盘状态字符串
- 加载训练好的 YOLOv8 模型，检测 12 类棋子
- 将检测结果映射到 8×8 棋盘格位置，输出 FEN 字符串

### `vision/piece_train.py` — YOLOv8 棋子检测训练脚本
- 训练 12 类国际象棋棋子目标检测模型
- 支持自定义数据集路径、epoch 数、batch size、学习率等
- 训练结束后自动导出 TorchScript 格式以加速推理

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

### `main.py` — AI 对弈主协调节点
- **订阅**: `/chess_board_matrix`, `/chessboard_corners_3d`, `/my_gen3_lite/arm_status`
- **发布**: `/kinova_pick_place/goal_in_camera`, `/ai_eat_status`
- 通过双线性插值将棋盘 (col, row) 映射为相机坐标系 3D 坐标
- 吃子时先将对方棋子移至弃子区，再执行主走法
- 自动检测人类走棋（棋盘状态变化），触发 AI 回合

### `utils/calibration.py` — 手眼标定工具
- 交互式标定：将 ArUco 标签放在已知基座标系位置，按 `s` 记录样本
- 输出 T_camera_marker 变换矩阵，辅助优化 T_ee_camera 参数

---

## 与原 EE368 项目的对应关系

| 原始文件 (中国象棋) | 新文件 (国际象棋) | 主要改动 |
|---|---|---|
| `chessboard_detector_node.py` | `vision/board_detector.py` | 适配 8×8 棋盘，发布到 `/chessboard_corners_3d` |
| `matrix_construction.py` | `vision/piece_classifier.py` | 12 类棋子，输出 FEN 字符串而非 RegionMatrix |
| `piece_train.py` | `vision/piece_train.py` | 12 类国际象棋棋子训练 |
| `elephant_fish.py` | `chess_engine/ai_player.py` | python-chess + Stockfish 替代中国象棋引擎 |
| (新) | `chess_engine/game_state.py` | 新增 python-chess 封装层 |
| `inverse_kinematics.py` | `robot_arm/kinematics.py` | 基本不变，统一代码风格 |
| `move_cartesian.py` | `robot_arm/arm_controller.py` | 基本不变，去掉测试代码 |
| `kinova_grasp.py` / `kinova_cubic_cmd_grasp.py` | `robot_arm/motion_planner.py` | 合并为统一节点，简化架构 |
| `chess_ai_node.py` | `main.py` | 8×8 棋盘，python-chess 逻辑 |
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
- **ultralytics + torch**: YOLOv8 棋子检测
- **python-chess**: 国际象棋规则引擎
- **numpy + scipy**: 数值计算与运动学求解
- **[Stockfish](https://stockfishchess.org/)** 引擎（可选，但强烈推荐）

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

### 2. 训练棋子检测模型
```bash
# 准备数据集（标注好的 12 类国际象棋棋子图片）
# 创建 configs/chess_data.yaml 描述数据集

python vision/piece_train.py \
    --data configs/chess_data.yaml \
    --epochs 200 \
    --batch 16 \
    --device 0
```

### 3. 相机标定（如需要）
```bash
# 运行手眼标定工具
python utils/calibration.py

# 将 ArUco 标签放在基座标系中的已知位置
# 按 's' 键记录多个样本
# 按 'q' 键输出标定结果
```

### 4. 启动完整系统
```bash
# 确保 ROS master 正在运行
roscore &

# 一键启动全部节点
roslaunch chess_robot_arm chess_robot.launch

# 自定义参数启动
roslaunch chess_robot_arm chess_robot.launch \
    robot_name:=my_gen3_lite \
    stockfish_path:=/usr/local/bin/stockfish \
    think_time:=5.0 \
    board_z:=0.39
```

### 5. 分步启动（调试用）
```bash
# 终端 1: 启动棋盘检测
rosrun chess_robot_arm board_detector.py

# 终端 2: 启动棋子分类
rosrun chess_robot_arm piece_classifier.py

# 终端 3: 启动 AI 协调器
rosrun chess_robot_arm motion_planner.py

# 终端 4: 启动运动规划器
rosrun chess_robot_arm main.py
```

### 6. 对弈流程
1. 系统启动后，机械臂自动归位 (home 位姿)
2. 棋盘检测节点持续发布棋盘角点位置
3. 棋子分类节点每秒检测一次棋盘状态，发布 FEN 字符串
4. **人类（执白）先走**：手动移动棋子
5. AI 协调器检测到人类走棋后，调用 Stockfish 计算走法
6. 将 AI 走法坐标发送给运动规划器执行机械臂码放
7. 循环直到对局结束

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
| `/chessboard_corners_3d` | `ChessboardCorners` | `board_detector.py` | `piece_classifier.py`, `main.py` | 棋盘四角 3D 坐标 (相机系) |
| `/chessboard_corners_2d` | `ChessboardPixelCorners` | `board_detector.py` | (调试用) | 棋盘四角 2D 像素坐标 |
| `/camera/color/image_raw` | `Image` | `board_detector.py` | `piece_classifier.py` | 原始彩色图像 |
| `/chess_board_matrix` | `String` | `piece_classifier.py` | `main.py` | 棋盘状态 (FEN 字符串) |
| `/kinova_pick_place/goal_in_camera` | `PickAndPlaceGoalInCamera` | `main.py` | `motion_planner.py` | 抓取放置目标指令 |
| `/my_gen3_lite/arm_status` | `String` | `motion_planner.py` | `main.py` | 机械臂状态 ("0"/"1"/"2") |
| `/ai_eat_status` | `String` | `main.py` | (日志用) | 吃子状态 ("BUSY"/"IDLE") |

---

## 机械臂状态码

| 状态码 | 含义 | 说明 |
|---|---|---|
| `0` / `IDLE` | 空闲 | 机械臂就绪，可接受新指令 |
| `1` / `BUSY` | 忙碌 | 正在执行抓取-放置操作，忽略新指令 |
| `2` / `ERROR` | 故障 | 发生错误，需人工干预清除故障后恢复 |

---

## 配置参数

所有参数均可在 ROS launch 文件中设置，支持 `rosparam` 动态覆盖。

### 关键参数列表
| 参数 | 默认值 | 说明 |
|---|---|---|
| `~robot_name` | `my_gen3_lite` | ROS 中机械臂的名称 |
| `~stockfish_path` | `/usr/games/stockfish` | Stockfish 引擎路径 |
| `~think_time` | `3.0` | AI 每步思考时间 (秒) |
| `~board_z` | `0.38` | 棋盘表面固定 Z 高度 (米) |
| `~aruco_marker_size` | `0.05` | ArUco 标签边长 (米) |
| `~pre_action_z_lift` | `0.05` | 抓取/放置前 Z 轴抬升量 (米) |
| `~pick_z_offset` | `0.005` | 抓取 Z 轴微调偏移 (米) |
| `~place_z_offset` | `0.01` | 放置 Z 轴微调偏移 (米) |
| `~gripper_open` | `0.7` | 夹爪张开量 (0-1) |
| `~gripper_close` | `0.25` | 夹爪闭合量 (0-1) |
| `~conf_threshold` | `0.5` | YOLO 检测置信度阈值 |
| `~show_cv_window` | `true` | 是否显示 OpenCV 可视化窗口 |
| `~show_visualization` | `true` | 是否显示棋子分类可视化 |

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
roslaunch chess_robot_arm chess_robot.launch stockfish_path:=/path/to/stockfish
```

### 机械臂连接失败
```bash
# 确保 kortex_driver 正确安装
rospack find kortex_driver
# 检查机械臂 IP 和网络连接
ping <robot_ip>
```

### YOLO 模型未训练
```bash
# 使用预训练模型进行快速测试
# 或运行训练脚本
python vision/piece_train.py --data configs/chess_data.yaml --epochs 50
```

---

## 许可

本项目基于 EE368_Project 改编，仅供学习和研究使用。
