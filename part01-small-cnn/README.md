# Part 01 · 从零训练 CIFAR-10 小型 CNN

这是我的深度学习自学项目第一部分：使用 PyTorch 从零设计一个适合 `32×32` 图片的小型 CNN，完成从训练检查到本地图片推理的完整流程。

模型没有使用预训练权重，共有 **288,746 个可学习参数**。一次完整实验中，最佳验证准确率为 **89.08%**，最终 CIFAR-10 测试准确率为 **88.73%**。

## 完成了什么

- 设计三个双卷积块，推导并验证每层张量尺寸
- 通过固定单 batch 过拟合检查训练链路
- 实现训练、验证和最终测试
- 使用数据增强、AdamW 和余弦学习率调度
- 根据验证准确率保存最佳 checkpoint
- 加载 checkpoint，验证参数恢复正确
- 绘制训练与验证的 Loss、Accuracy 曲线
- 对本地 PNG/JPG 图片进行 Top-3 推理

这个项目保留了学习过程中的实验和思考，重点是理解各个环节，而不是追求榜单成绩。

## 网络结构

每个卷积块的结构为：

```text
Conv 3×3 → BatchNorm → ReLU
Conv 3×3 → BatchNorm → ReLU
MaxPool 2×2
```

卷积使用 `stride=1`、`padding=1`、`bias=False`。池化使用 `stride=2`。

| 阶段 | 输出形状 |
| --- | --- |
| 输入 RGB 图片 | `[N, 3, 32, 32]` |
| Block 1：3→32→32 | `[N, 32, 16, 16]` |
| Block 2：32→64→64 | `[N, 64, 8, 8]` |
| Block 3：64→128→128 | `[N, 128, 4, 4]` |
| AdaptiveAvgPool2d(1) | `[N, 128, 1, 1]` |
| Flatten | `[N, 128]` |
| Linear 128→10 | `[N, 10]` |

模型使用全局平均池化减少分类头参数量。输出为 logits，训练时直接输入 `CrossEntropyLoss`，推理时再使用 Softmax 展示类别概率。

## 数据与实验设置

使用 CIFAR-10 官方数据，将其中 50,000 张训练图片进一步划分：

| 划分 | 数量 | 用途 |
| --- | ---: | --- |
| 训练集 | 45,000 | 更新模型参数 |
| 验证集 | 5,000 | 选择最佳 checkpoint |
| 官方测试集 | 10,000 | 模型选择结束后评估 |

划分通过 `torch.randperm` 完成，使用局部随机数生成器固定种子 `42`，并检查训练与验证索引没有重叠。

训练数据增强：

```text
RandomCrop(32, padding=4, padding_mode="reflect")
→ RandomHorizontalFlip(p=0.5)
→ ToTensor
→ Normalize
```

验证和测试只使用 `ToTensor → Normalize`。训练与验证使用独立的 Dataset 对象，避免共享随机增强。

归一化参数：

```python
mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)
```

| 训练配置 | 数值 |
| --- | --- |
| Optimizer | AdamW |
| 初始学习率 | `1e-3` |
| Weight decay | `5e-4` |
| Scheduler | CosineAnnealingLR |
| 最低学习率 | `1e-5` |
| Epoch 数 | 30 |
| 训练 batch size | 128 |
| 验证 / 测试 batch size | 256 |
| Loss | CrossEntropyLoss |
| 最佳模型选择标准 | 验证准确率 |

## 实验结果

### 单 batch 过拟合

正式训练前，固定 64 张图片反复训练，不使用随机增强：

| Step | Loss | Accuracy |
| --- | ---: | ---: |
| 0 | 2.5074 | 12.50% |
| 20 | 0.0474 | 100% |
| 500 | 0.0003 | 100% |

切换至 `eval()` 后，这批图片仍达到 100% 准确率。

这个实验用于检查前向传播、损失、梯度和参数更新是否正常；它不代表模型能泛化到新图片。

### 完整训练

以下是自学过程中记录的一次运行结果：

| 指标 | 结果 |
| --- | ---: |
| 最佳验证 epoch | 29 |
| 最佳验证准确率 | 89.08% |
| 最终测试准确率 | **88.73%** |
| 最终测试 loss | 0.3457 |

训练结束后加载最佳验证 checkpoint，再执行最终测试。

当前只固定了数据划分种子，没有固定完整训练过程。随机初始化、数据顺序和数据增强会使重复运行结果有所不同。

## 项目结构

```text
part01-small-cnn/
├── smallcnn.py          # 卷积块和 SmallCNN 定义
├── train.py             # 数据划分、训练、验证、保存和最终测试
├── load_checkpoint.py   # 加载模型、测试集复测和单张样本推理
├── plot_history.py      # 绘制训练与验证曲线
├── predict_image.py     # 本地 PNG/JPG 图片推理
└── my images/           # 本地推理示例图片
```

运行时生成：

```text
data/
checkpoints/
├── best_checkpoint.pt
└── last_checkpoint.pt
training_curves.png
prediction_result.png
```

数据和 checkpoint 不包含在当前目录的源码中，需要下载或训练生成。

## 快速开始

### 准备环境

项目在 Ubuntu / WSL、Conda 环境 `d2l` 和 RTX 5060 Laptop GPU（8GB 显存）上完成。

主要依赖：

```text
torch
torchvision
matplotlib
Pillow
tqdm
```

先安装与本机环境匹配的 PyTorch 和 torchvision，然后安装其他依赖：

```bash
pip install matplotlib Pillow tqdm
```

克隆仓库并进入本部分目录：

```bash
git clone https://github.com/TimBergling07/deep-learning-from-scratch.git
cd deep-learning-from-scratch/part01-small-cnn
```

后续命令均在 `part01-small-cnn` 目录下运行。

### 训练

```bash
python train.py
```

首次运行自动下载 CIFAR-10。每轮训练后执行验证，并写入 `checkpoints/`：

- `best_checkpoint.pt`：验证准确率最高时的状态。
- `last_checkpoint.pt`：最近一轮的状态，包含完整历史记录。

训练结束后，程序加载最佳模型并输出最终测试结果。重新执行训练会覆盖同名 checkpoint；保留多次实验时需要自行更换保存目录。

### 加载与复测

```bash
python load_checkpoint.py
```

程序加载 `checkpoints/best_checkpoint.pt`，评估官方测试集，并对测试集中的一张图片输出真实类别、预测类别和 Top-3 概率。

训练期间应使用验证集进行模型选择，避免将反复复测的测试结果继续用于调参。

### 绘制训练曲线

```bash
python plot_history.py
```

读取 `checkpoints/last_checkpoint.pt` 中的完整历史，生成：

```text
training_curves.png
```

图中包含训练和验证的 Loss、Accuracy 曲线。

### 本地图片推理

```bash
python predict_image.py "/path/to/cat.jpg"
```

路径包含空格时使用引号。

图片经过以下处理：

```text
读取并转为 RGB
→ 短边缩放到 36
→ 中心裁剪为 32×32
→ ToTensor
→ Normalize
→ 添加 batch 维
→ 模型推理
```

程序输出 Top-3 类别及置信度，并生成 `prediction_result.png`。

没有图形窗口的环境中，可以将绘图脚本中的 `plt.show()` 去掉，直接查看保存的图片。

## checkpoint 内容

保存的状态包括：

- epoch
- 模型参数及 BatchNorm 运行统计量
- 优化器状态
- 学习率调度器状态
- 训练与验证指标
- 最佳验证准确率
- 历史记录 `history`

当前已经实现保存、加载和复测。虽然保存了续训需要的主要状态，`train.py` 尚未实现自动恢复训练入口，重新运行仍会从头训练。

## 踩过的坑

### 误用训练 DataLoader 进行“测试”

早期曾把 `evaluate()` 的输入误写为 `train_loader`，得到约 94% 的虚假“测试准确率”。修正后，真正测试准确率回到约 89%。

这让我意识到，指标名称并不能保证评估正确，必须检查数据来源。

### 按测试准确率选择 checkpoint

随后将流程从“每轮测试并选最好的一轮”，改成：

```text
训练集更新参数
→ 验证集选择模型
→ 测试集进行最终评估
```

测试图片即使没有参与反向传播，也可能通过模型选择影响实验结论。

开发早期已经查看过官方测试结果，因此最终流程虽然规范，也不能视为一次完全未接触测试集的盲测。

## 真实图片上的表现与限制

模型只认识以下十类：

```text
airplane, automobile, bird, cat, deer,
dog, frog, horse, ship, truck
```

面对其他物体，它也会从这十类中强行选择一个。

实际尝试中，一张猫图以约 95% 置信度识别正确；一张狗的侧脸却以约 65% 置信度被误判成 automobile，换成另一张正脸图片后，dog 置信度提高到 80% 以上。

这些现象提示模型对视角、背景和构图敏感。外部照片与 CIFAR-10 存在分布差异，缩放至 `32×32` 也会丢失细节。少量样本不足以判断误识别的具体原因。

**Softmax 置信度不等于预测正确的真实概率。**

## 学习收获

通过这个项目，我完成了一次从网络设计到实际推理的图像分类流程，并理解了：

- 小图像需要与输入尺寸匹配的网络结构。
- 单 batch 过拟合可以帮助排查训练链路。
- `eval()` 与 `inference_mode()` 有不同职责。
- 整个 epoch 的平均 Loss 应按样本数量加权累计。
- checkpoint 除了权重，还应保存训练状态和历史。
- 数据划分与模型选择方式直接影响结果是否可信。
- 较高的测试准确率，并不意味着模型理解了真实世界。