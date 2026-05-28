"""
国际象棋机械臂项目的物理常数与棋子映射。
所有长度单位默认为米（除非特别注明）。
"""

# --- 8×8 国际象棋棋盘尺寸 ---
BOARD_ROWS = 8                     # 棋盘行数
BOARD_COLS = 8                     # 棋盘列数
SQUARE_SIZE = 0.04                # 标准比赛棋格边长（约40mm）
BOARD_WIDTH = SQUARE_SIZE * BOARD_COLS
BOARD_HEIGHT = SQUARE_SIZE * BOARD_ROWS

# --- 棋盘四角对应的 ArUco 标签 ID ---
# 约定：从机械臂 home 位置俯瞰棋盘的方向
ARUCO_TOP_LEFT_ID     = 0          # 棋盘左上角
ARUCO_TOP_RIGHT_ID    = 1          # 棋盘右上角
ARUCO_BOTTOM_LEFT_ID  = 2          # 棋盘左下角
ARUCO_BOTTOM_RIGHT_ID = 3          # 棋盘右下角
ARUCO_MARKER_SIZE     = 0.06       # ArUco 标签边长 6cm
ARUCO_DICT_NAME       = "DICT_6X6_250"

# --- 国际象棋棋子符号映射 ---
# YOLO 分类 class_id → FEN 字符
PIECE_ID_TO_FEN = {
    0:  'K',   # 白王
    1:  'Q',   # 白后
    2:  'R',   # 白车
    3:  'B',   # 白象
    4:  'N',   # 白马
    5:  'P',   # 白兵
    6:  'k',   # 黑王
    7:  'q',   # 黑后
    8:  'r',   # 黑车
    9:  'b',   # 黑象
    10: 'n',   # 黑马
    11: 'p',   # 黑兵
}

FEN_TO_PIECE_ID = {v: k for k, v in PIECE_ID_TO_FEN.items()}

# YOLO 分类名称（共 12 类）
PIECE_CLASS_NAMES = [
    "black_king", "black_queen", "black_rook", "black_bishop",
    "black_knight", "black_pawn",
    "white_king", "white_queen", "white_rook", "white_bishop",
    "white_knight", "white_pawn",
]

# 空格占位符
EMPTY_SQUARE = '0'

# --- Kinova Gen3 Lite 标准 DH 参数表 ---
# 格式：[alpha_{i-1}, a_{i-1}, d_i, theta_offset_i]
# 角度单位：弧度，长度单位：米
DH_PARAMETERS = [
    [0,            0,      0.2433, 0       ],
    [1.57079633,   0,      0.03,   1.57079633],
    [3.14159265,   0.28,   0.02,   1.57079633],
    [1.57079633,   0,      0.245,  1.57079633],
    [1.57079633,   0,      0.057,  3.14159265],
    [1.57079633,   0,      0.235,  1.57079633],
]
NUM_JOINTS = 6                      # 关节数

# --- 手眼变换矩阵（末端执行器 → 相机） ---
# T_ee_camera: 从末端执行器坐标系到相机光学坐标系的齐次变换
# 平移量单位：米
T_EE_CAMERA_TRANSLATION = [0.060, -0.040, -0.110]
# 旋转矩阵：相机光学坐标系在末端执行器坐标系中的姿态
T_EE_CAMERA_ROTATION = [
    [0.0, -1.0,  0.0],
    [1.0,  0.0,  0.0],
    [0.0,  0.0,  1.0],
]

# --- 机械臂初始/安全关节角度（度） ---
HOME_JOINT_ANGLES_DEG = [30.66, 346.57, 72.23, 270.08, 265.45, 345.69]

# --- 标定完成后机械臂避让位姿（笛卡尔坐标，避开相机视野） ---
# 用于 ArUco 角点检测完毕后、棋子识别开始前，机械臂移到此位置
POST_CALIB_HOME = {
    "x": 0.046,
    "y": 0.355,
    "z": 0.293,
    "rx": 141.7,
    "ry": 0.5,
    "rz": 38.2
}

# --- 机械臂运动参数 ---
DEFAULT_GRIPPER_ORIENTATION_DEG = [0.0, 180.0, 45.0]  # 夹爪姿态 (roll, pitch, yaw)
PRE_ACTION_Z_LIFT = 0.08        # 抓取/放置前在目标上方的抬升高度
PICK_Z_OFFSET = -0.005             # 抓取时的 Z 轴微调偏移
PLACE_Z_OFFSET = -0.005             # 放置时的 Z 轴微调偏移
GRIPPER_OPEN_VALUE = 0.34          # 夹爪闭合至 66%（接近物体前）
GRIPPER_CLOSE_VALUE = 0.04        # 夹爪闭合至 4%（夹住棋子）
MOTION_DURATION = 4.5             # 默认轨迹执行时间（秒）

# --- 相机参数（Intel RealSense D435 默认值，运行时可被 ROS param 覆盖） ---
CAMERA_WIDTH  = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS    = 30

# --- 被吃棋子放置参数 ---
# 被吃掉的棋子放置在棋盘右侧的弃子区
GARBAGE_OFFSET_FACTOR = 2.0       # 从左上角指向右上角向量的倍数

# --- AI 参数 ---
STOCKFISH_PATH = "/usr/games/stockfish"   # Stockfish 引擎路径
STOCKFISH_SKILL_LEVEL = 10                # 难度等级 1-20
STOCKFISH_THINK_TIME = 3.0                # 每步思考时间（秒）
