# Face2Sketch：人脸照片转素描

本仓库包含三种基于 U-Net 的方法：**U-Net L1**、**U-Net L1 GAN** 和 **U-Net Edge**。三个现有权重均在 FS2K 官方训练集的全部 **1,058 对**照片与素描上训练 150 轮，涵盖全部素描风格。官方测试集没有参与训练。输入为 RGB 照片，输出为 256×256 灰度素描。

| 方法 | 输入 | 损失 | 配置 | 最佳权重 |
| --- | --- | --- | --- | --- |
| U-Net L1 | RGB | L1 | `configs/unet_all_styles.yaml` | `checkpoints/unet_all_styles_best.pth` |
| U-Net L1 GAN | RGB | GAN + 100 × L1 | `configs/pix2pix_all_styles.yaml` | `checkpoints/pix2pix_all_styles_best.pth` |
| U-Net Edge | RGB + Canny 边缘图 | GAN + 100 × L1 | `configs/edge_pix2pix_all_styles.yaml` | `checkpoints/edge_pix2pix_all_styles_best.pth` |

## 安装与下载数据

在仓库根目录运行：

```bash
python -m pip install -r requirements.txt
mkdir -p data
```

从 [FS2K 官方仓库](https://github.com/DengPingFan/FS2K)提供的 [Google Drive 页面](https://drive.google.com/file/d/1saIMhQ3dc5_ftkfGmBPbCluRn_zy7QQp/view?usp=sharing)下载 `FS2K.zip`，放到 `data/FS2K.zip`，然后解压：

```bash
unzip data/FS2K.zip -d data
python scripts/check_dataset.py --data-root data/FS2K
```

解压后应存在 `data/FS2K/anno_train.json`、`anno_test.json`、`photo/` 和 `sketch/`。检查脚本验证配对、文件可读性和训练/测试集不重叠，并保存 `results/dataset_preview.png`。预期官方训练集 1,058 对，测试集 1,046 对。`data/` 被 Git 忽略。

## 预处理

无需离线生成新数据集。读取时照片转 RGB、素描转灰度，均缩放为 256×256 并映射到 `[0,1]`。训练时照片与素描同步随机水平翻转；亮度和对比度增强只作用于照片。U-Net Edge 从处理后的照片生成 Canny 图：先做双边滤波（`d=5`、`sigmaColor=35`、`sigmaSpace=5`），再用阈值 **45/110** 检测边缘，作为第 4 个输入通道。真实素描标签不改变。

## 训练

```bash
python train_unet.py --config configs/unet_all_styles.yaml
python train_pix2pix.py --config configs/pix2pix_all_styles.yaml
python train_pix2pix.py --config configs/edge_pix2pix_all_styles.yaml
```

三个配置都使用完整官方训练集（`style: all`、`val_fraction: 0.0`），批量大小 8、训练 150 轮，并将权重保存到 `checkpoints/`、TensorBoard 日志和预览图保存到 `runs/`。由于没有留出验证集，`best.pth` 按**训练 L1** 选取，不能视为验证集最优。继续训练可用对应的 `--resume checkpoints/<模型>_last.pth`；查看日志可运行 `tensorboard --logdir runs`。

## 对自己的照片推理与比较

```bash
python inference.py --checkpoint checkpoints/pix2pix_all_styles_best.pth --input image_2.jpg --output results/my_sketch.png
python compare_models.py --input image_2.jpg --output results/my_comparison.png
```

画图顺序为 **原图｜U-Net L1｜U-Net L1 GAN｜U-Net Edge**。现有示例见[四图对比](results/image_2_all_styles_four_models.png)。非方形照片可加 `--resize-mode center_crop` 保持脸部比例；默认 `stretch` 与训练缩放一致。推理只加载生成器，不需要判别器。

方法说明见 [METHODS_README.md](METHODS_README.md)，实验报告见 [PROJECT_REPORT.md](PROJECT_REPORT.md)。
