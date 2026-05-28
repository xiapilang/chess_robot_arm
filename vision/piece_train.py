#!/usr/bin/env python3
"""
YOLOv8 国际象棋棋子检测模型训练脚本。

训练 12 类目标检测模型：6 种棋子类型 × 2 种颜色（白/黑）。

使用方法:
    python piece_train.py --data configs/chess_data.yaml --epochs 200 --batch 16

数据集 YAML 配置文件需定义:
  - path: 数据集根目录
  - train: 训练图像目录
  - val:   验证图像目录
  - names: [white_king, white_queen, ..., black_pawn]（共 12 类）
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="训练 YOLOv8 国际象棋棋子检测模型")
    parser.add_argument("--data", default="configs/chess_data.yaml",
                        help="数据集 YAML 配置文件路径")
    parser.add_argument("--model", default="yolov8s.pt",
                        help="预训练模型权重文件")
    parser.add_argument("--epochs", type=int, default=200,
                        help="训练轮次")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="输入图像尺寸")
    parser.add_argument("--batch", type=int, default=16,
                        help="批次大小")
    parser.add_argument("--device", default="0",
                        help="训练设备: 0 表示 GPU，'cpu' 表示 CPU")
    parser.add_argument("--project", default=".",
                        help="输出项目目录（默认当前目录即项目根目录）")
    parser.add_argument("--name", default="runs/chess_piece",
                        help="实验名称")
    parser.add_argument("--lr0", type=float, default=0.01,
                        help="初始学习率")
    parser.add_argument("--lrf", type=float, default=0.1,
                        help="最终学习率因子")
    parser.add_argument("--optimizer", default="SGD",
                        help="优化器类型: SGD, Adam, AdamW")
    parser.add_argument("--patience", type=int, default=50,
                        help="早停耐心值（多少个 epoch 无提升后停止）")

    args = parser.parse_args()

    # 初始化模型（基于预训练权重）
    model = YOLO(args.model)

    # 开始训练
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=0.937,
        weight_decay=0.0005,
        patience=args.patience,
        save_period=10,          # 每 10 个 epoch 保存一次
        half=True,               # 启用 FP16 半精度加速
    )

    import shutil
    best_pt = f"{args.project}/{args.name}/weights/best.pt"
    root_pt = "./best.pt"
    shutil.copy(best_pt, root_pt)
    print(f"训练完成。最佳模型已复制到: {root_pt}")

    model = YOLO(best_pt)
    model.export(format="torchscript")


if __name__ == "__main__":
    main()
