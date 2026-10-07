# 三种人脸照片转素描方法

三个模型均使用 FS2K 官方训练集全部 **1,058 对**样本、256×256 输入输出和灰度素描标签。

| 方法名称 | 输入与模型 | 训练损失 | 在当前照片上的特点 |
| --- | --- | --- | --- |
| **U-Net L1** | RGB → U-Net | L1 | 脸部结构基本保留，但线条较模糊。 |
| **U-Net L1 GAN** | RGB → U-Net 生成器；条件 PatchGAN 判别器 | GAN + 100 × L1 | 头发、眼镜和五官线条较清楚。 |
| **U-Net Edge** | RGB 与 Canny 边缘图拼接为 4 通道 → U-Net；条件 PatchGAN 判别器 | GAN + 100 × L1 | 轮廓更突出，但部分线条可能偏多。 |

U-Net Edge 的 Canny 阈值是 **45/110**。边缘图只作辅助输入，不直接叠加到输出上；真实素描标签保持原样。

## 当前效果

[查看 `image_2.jpg` 的对比图](results/image_2_all_styles_four_models.png)：原图｜U-Net L1｜U-Net L1 GAN｜U-Net Edge。对这张照片的主观观察是 **U-Net L1 GAN 效果最好**；U-Net L1 较平滑，U-Net Edge 保留了轮廓但有些杂线。此结论不等于独立测试集排名。三个全量模型没有留出验证集，最低训练 L1 也不能替代泛化评价。

## 对自己的照片推理

```bash
python inference.py --checkpoint checkpoints/pix2pix_all_styles_best.pth --input image_2.jpg --output results/my_sketch.png
python compare_models.py --input image_2.jpg --output results/my_comparison.png
```

数据下载、预处理与训练命令见 [README.md](README.md)。
