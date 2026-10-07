# 基于 U-Net 的人脸照片转素描项目报告

## 摘要

本项目将 RGB 人脸照片转换成 256×256 灰度素描，比较三种方法：**U-Net L1**、**U-Net L1 GAN** 和 **U-Net Edge**。三个模型均使用 FS2K 官方训练集全部 **1,058 对**配对样本，覆盖全部素描风格，各训练 150 轮。后两种方法共享条件 PatchGAN 对抗训练框架；U-Net Edge 额外输入照片的 Canny 边缘图。在个人照片的主观对比中，U-Net L1 GAN 的线条最清晰且五官较自然。

## 1. 数据与预处理

数据来自 [FS2K](https://github.com/DengPingFan/FS2K)。官方训练集 1,058 对，官方测试集 1,046 对。本次训练使用全部官方训练样本，**没有另划验证集，也没有使用官方测试集训练**。因此，下文列出的 L1 均为训练指标，不能视为泛化性能。

照片转 RGB，素描转灰度，分别用双三次插值缩放至 256×256，再将像素映射到 `[0,1]`。训练时对配对照片与素描同步随机水平翻转；亮度与对比度增强只作用于照片。真实素描标签保留灰度和线条细节。

## 2. 三种方法

| 方法 | 生成器输入 | 训练目标 | 主要区别 |
| --- | --- | --- | --- |
| **U-Net L1** | RGB，3 通道 | L1 | 学习预测素描与真实素描的像素差异。 |
| **U-Net L1 GAN** | RGB，3 通道 | GAN + 100 × L1 | 条件 PatchGAN 约束局部素描线条。 |
| **U-Net Edge** | RGB + Canny，4 通道 | GAN + 100 × L1 | 以对齐的边缘图辅助生成器和判别器。 |

### 2.1 网络结构

生成器使用五级 U-Net，通道数 `[32, 64, 128, 256, 512]`。编码器采用 3×3 卷积、BatchNorm、ReLU 和最大池化；解码器使用上采样与跳跃连接。Sigmoid 将输出映射到 `[0,1]`。RGB 生成器有 **7,849,601** 个参数，四通道 Edge 生成器有 **7,849,889** 个参数。

两种 GAN 方法使用条件 70×70 PatchGAN。判别器接收照片条件与真实或生成素描的通道拼接，输出局部真假预测图；推理时只使用生成器。目标函数为：

\[
L_G=L_{\mathrm{GAN}}+100L_{\mathrm{L1}},\qquad
L_D=\tfrac12(L_{D,\mathrm{real}}+L_{D,\mathrm{fake}}).
\]

U-Net Edge 先对照片做双边滤波（`d=5`、`sigmaColor=35`、`sigmaSpace=5`），再使用 Canny 阈值 **45/110** 生成第 4 个输入通道。边缘图不直接叠到生成素描上，素描标签也不经过边缘检测。

## 3. 训练设置

三个模型均训练 **150 轮**，批量大小 **8**，随机种子 **42**，Adam 学习率 **0.0002**，CUDA 混合精度开启。U-Net L1 使用余弦学习率调度；两种 GAN 方法使用恒定学习率，Adam 的 `β₁=0.5`、`β₂=0.999`。训练保存每轮损失、TensorBoard 日志、定期预览图，以及 `best` 和 `last` 权重。

因全部 1,058 对训练样本用于拟合，`best` 按**训练 L1 最低**选取。三个模型采用不同损失和训练过程，训练 L1 不能直接当作视觉质量排名。

| 方法 | 最低训练 L1 | 平均每步耗时 | GPU 峰值显存 |
| --- | ---: | ---: | ---: |
| U-Net L1 | 0.07047 | 0.0266 秒 | 1.19 GiB |
| U-Net L1 GAN | 0.08225 | 0.0303 秒 | 1.26 GiB |
| U-Net Edge | 0.08113 | 0.0311 秒 | 1.26 GiB |

## 4. 个人照片的可视化结果

下图使用 `image_2.jpg`，从左到右依次为**原图、U-Net L1、U-Net L1 GAN、U-Net Edge**：

![个人照片的三方法对比](results/image_2_all_styles_four_models.png)

在这张照片上，U-Net L1 输出平滑、线条较模糊；U-Net Edge 显示较多轮廓线，也有一些杂线；**U-Net L1 GAN 的眼镜、头发和五官线条较自然**。这是单张照片的主观观察，不代表独立测试集结论。

## 5. 局限

输出固定为 256×256，部分眼镜框和细发丝可能丢失。三种模型都使用包含不同素描风格的标签，单张输出风格不一定完全统一。Edge 方法依赖固定 Canny 阈值，可能将纹理或光照变化误当作轮廓。由于没有保留验证集，本次模型选择依据训练指标；更可靠的泛化比较需要另行使用独立数据评估。

## 6. 复现命令

完整数据下载与预处理说明见 [README.md](README.md)。在仓库根目录运行：

```bash
python scripts/check_dataset.py --data-root data/FS2K
python train_unet.py --config configs/unet_all_styles.yaml
python train_pix2pix.py --config configs/pix2pix_all_styles.yaml
python train_pix2pix.py --config configs/edge_pix2pix_all_styles.yaml
python compare_models.py --input image_2.jpg --output results/image_2_all_styles_four_models.png
```
