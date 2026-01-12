# CIFAR-10 Image Classification using ResNet18

这是一个基于 **PyTorch Lightning** 实现的高性能 CIFAR-10 图像分类项目。项目使用针对小尺寸图像优化过的 **ResNet18** 架构，在测试集上达到了 **95.32%** 的准确率，超过了预设的 93% 目标。

##  实验结果

| 指标 (Metric)         | 结果 (Value) | 说明 (Note)               |
| :-------------------- | :----------- | :------------------------ |
| **Test Accuracy**     | **95.32%**   | 远超 93% 的目标           |
| **Best Val Accuracy** | **95.78%**   | 最佳模型出现在 Epoch 197  |
| **Test Loss**         | 0.1803       |                           |
| **Train Accuracy**    | 100.0%       | 模型充分收敛              |
| **Training Time**     | ~1.5 小时    | on NVIDIA RTX 4060 Laptop |

##  项目结构

项目采用了 PyTorch Lightning 的标准结构，实现了数据、模型和训练逻辑的解耦：

```text
cifar10_project/
├── data/
│   └── cifar_datamodule.py    # LightningDataModule: 处理数据下载、增强和加载
├── models/
│   └── resnet_module.py       # LightningModule: 定义 ResNet18 架构和训练步
├── checkpoints/               # 存放训练好的模型权重 (.ckpt)
├── logs/                      # TensorBoard 日志文件
├── train.py                   # 主训练脚本
├── requirements.txt           # 项目依赖
└── README.md                  # 项目说明
```

##  模型架构 

本项目使用了 **ResNet18**，并针对 CIFAR-10 数据集（32x32 像素）进行了关键修改，以避免信息丢失：

1.  **第一层卷积修改**：将原始 ResNet 的 `7x7 conv, stride=2` 修改为 `3x3 conv, stride=1`。
2.  **移除池化层**：移除了第一层卷积后的 `MaxPool2d`。
3.  **优势**：这些修改保留了特征图的空间分辨率，使其更适合处理 CIFAR-10 的低分辨率图像。

##  快速开始 

### 1. 环境要求

*   Python 3.8+
*   PyTorch 2.0+
*   PyTorch Lightning 2.0+
*   CUDA 

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 开始训练

```bash
python train.py
```

训练脚本会自动下载 CIFAR-10 数据集并开始训练。

### 4. 查看训练过程

使用 TensorBoard 监控 Loss 和 Accuracy 曲线：

```bash
tensorboard --logdir logs/
```

##  训练配置 

| 参数                  | 值                         | 说明                                   |
| :-------------------- | :------------------------- | :------------------------------------- |
| **Epochs**            | 200                        | 充分训练以保证收敛                     |
| **Batch Size**        | 128                        |                                        |
| **Optimizer**         | SGD                        | Momentum=0.9, Weight Decay=5e-4        |
| **Learning Rate**     | 0.1                        | 配合余弦退火调度器 (Cosine Annealing)  |
| **Precision**         | 16-mixed                   | 混合精度训练 (AMP)，加速并减少显存占用 |
| **Data Augmentation** | RandomCrop, HorizontalFlip | 仅在训练阶段启用                       |

##  训练日志摘要

根据最终训练日志：

*   **收敛情况**：在 200 个 Epoch 结束时，训练集准确率达到 100%，验证集准确率稳定在 95% 以上。
*   **最佳模型**：系统自动保存了验证集准确率最高的模型 (`epoch=197-val_acc=0.9578.ckpt`)。
*   **硬件利用**：使用了 RTX 4060 Laptop GPU，并开启了 AMP (Automatic Mixed Precision)。

##  作者

*   **Student**: He Wang
*   **Date**: 2026-01-12
*   **Framework**: PyTorch Lightning

