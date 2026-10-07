# 三种人脸照片转素描方法

本项目使用 FS2K 配对照片和素描训练三种方法。输入照片与输出素描都按 256×256 处理，输出是灰度图。代码、模型权重与本说明文件都在仓库根目录。

| 方法名称 | 输入与模型 | 训练损失 | 输出特点 |
| --- | --- | --- | --- |
| **U-Net L1** | RGB 照片 → U-Net | L1 | 脸部结构基本保留，但线条偏模糊。 |
| **U-Net L1 GAN** | RGB 照片 → U-Net 生成器；PatchGAN 判别照片与素描是否匹配 | GAN + 100 × L1 | 线条更清楚，头发和五官更像手绘素描。 |
| **U-Net Edge** | RGB 照片与 Canny 边缘图拼成 4 通道 → U-Net；再由 PatchGAN 训练 | GAN + 100 × L1 | 边缘辅助突出轮廓，但可能产生多余的线条。 |

U-Net Edge 使用双边滤波后进行 Canny 边缘检测，阈值为 **45/110**。边缘图是附加输入，不会直接叠到生成的素描上；真实素描标签也保持原样。

## 当前效果对比

[查看最新照片的结果](results/image_comparison.png)：原图｜U-Net L1｜U-Net Edge｜U-Net L1 GAN。

**就这张照片的视觉效果而言，U-Net L1 GAN 最好。** 相比 U-Net L1，它的线条更清楚；相比 U-Net Edge，面部细节更自然。这个结论是对当前照片的主观观察，不代表所有照片都会有相同排序。

## 对自己的照片推理

在仓库根目录运行，使用 U-Net L1 GAN 生成一张素描：

```bash
python inference.py --checkpoint checkpoints/pix2pix_best.pth --input image.jpg --output results/image_l1_gan.png
```

一次生成三种方法的对比图：

```bash
python compare_models.py --input image.jpg --output results/image_comparison.png
```

数据下载、预处理和训练步骤仍见[原 README](README.md)。
