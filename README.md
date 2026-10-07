# Face2Sketch：人脸照片转素描

代码、配置、数据读取、权重和实验结果统一放在 [`face2sketch/`](face2sketch/README.md)。项目提供三种模型：U-Net + L1、RGB Pix2Pix、RGB + Canny 边缘辅助 Pix2Pix（阈值 45/110）。训练入口有两个：`train_unet.py` 和 `train_pix2pix.py`；测试入口对应为 `evaluate_unet.py` 和 `evaluate_pix2pix.py`。Pix2Pix 是否使用边缘辅助由 YAML 配置决定。

以下命令均在**仓库根目录先执行 `cd face2sketch`** 后运行。

## 1. 安装依赖

```bash
cd face2sketch
python -m pip install -r requirements.txt
python -m pytest tests -q
```

默认自动选择 CUDA、MPS 或 CPU；可在训练、评估、推理命令后添加 `--device cpu` 指定 CPU。

## 2. 下载并放置 FS2K

从 [FS2K 官方仓库](https://github.com/DengPingFan/FS2K)提供的 [Google Drive 数据压缩包](https://drive.google.com/file/d/1saIMhQ3dc5_ftkfGmBPbCluRn_zy7QQp/view?usp=sharing)下载 **FS2K.zip**。数据包包含照片、素描和官方训练/测试标注。将下载的压缩包放到 `face2sketch/data/FS2K.zip`，然后在 `face2sketch/` 下执行：

```bash
mkdir -p data
unzip data/FS2K.zip -d data
```

压缩包内部已有 `FS2K/` 顶层目录；解压后应是下面的结构，避免额外套一层 `FS2K`：

```text
face2sketch/data/FS2K/
├── anno_train.json
├── anno_test.json
├── photo/
│   ├── photo1/
│   ├── photo2/
│   └── photo3/
└── sketch/
    ├── sketch1/
    ├── sketch2/
    └── sketch3/
```

本仓库的 `data/` 已被 Git 忽略；数据需在新环境中自行下载。默认配置选择官方标注里的 `style: 0`，从该风格的官方训练标注划出 10% 验证集，并保持官方测试集不变。当前配置对应 **321 训练、36 验证、619 测试** 对。

## 3. 数据检查与预处理

```bash
python scripts/check_dataset.py --data-root data/FS2K --style 0
```

检查脚本会验证照片与素描配对、缺失文件、图片可解码性和数据划分，并生成 `results/dataset_preview.png`。训练不需要预先导出 256×256 的新数据集；[`FS2KDataset`](face2sketch/datasets/fs2k.py) 在读取时完成处理：

- 照片转 RGB、素描转灰度，分别缩放到 256×256，并映射到 `[0,1]`。
- 训练时照片和素描同步随机水平翻转；亮度、对比度增强只用于照片。验证和测试不做随机增强。
- 边缘辅助模型在处理后的照片上执行双边滤波（`d=5`、`sigmaColor=35`、`sigmaSpace=5`）和 Canny（低阈值 `45`、高阈值 `110`），将边缘图作为第 4 个输入通道。真实素描标签不做边缘化。

数据读取和边缘计算分别见 `face2sketch/datasets/fs2k.py` 与 `face2sketch/datasets/edge_fs2k.py`。

## 4. 训练

三个模型使用各自独立的配置与权重文件。运行下列命令会重新训练相应模型；仓库中已有的权重可直接用于下一节的测试和推理。

```bash
# U-Net + L1
python train_unet.py --config configs/unet.yaml

# RGB Pix2Pix（无边缘辅助）
python train_pix2pix.py --config configs/pix2pix.yaml

# RGB + 边缘 Pix2Pix（45/110）
python train_pix2pix.py --config configs/edge_pix2pix.yaml
```

默认训练 150 轮，超参数、数据路径、样本风格和日志目录在各自的 YAML 中。训练输出 TensorBoard 日志和周期性预览图到 `runs/`，并在 `checkpoints/` 保存最佳和最后一轮权重：

| 模型 | 最佳权重 | 最新权重 |
| --- | --- | --- |
| U-Net | `best.pth` | `last.pth` |
| RGB Pix2Pix | `pix2pix_best.pth` | `pix2pix_last.pth` |
| 边缘 Pix2Pix | `edge_pix2pix_best.pth` | `edge_pix2pix_last.pth` |

断点续训示例：

```bash
python train_pix2pix.py --config configs/edge_pix2pix.yaml --resume checkpoints/edge_pix2pix_last.pth
tensorboard --logdir runs
```

## 5. 官方测试集评估

```bash
python evaluate_unet.py --config configs/unet.yaml
python evaluate_pix2pix.py --config configs/pix2pix.yaml
python evaluate_pix2pix.py --config configs/edge_pix2pix.yaml
```

三个命令都使用相同的官方测试标注和确定性预处理，分别输出 MAE、PSNR、SSIM，并保存预测图和照片／预测／真实素描对比图。结果位于 `results/`、`results/pix2pix/` 和 `results/edge_pix2pix/`；各自的 `metrics.json` 是指标汇总。

## 6. 单图推理与模型对比

以仓库根目录的最新照片 `111.jpg` 为例：

```bash
python inference.py --checkpoint checkpoints/edge_pix2pix_best.pth --input ../111.jpg --output results/111_edge_sketch.png
python compare_models.py --input ../111.jpg --output results/111_comparison.png
```

`compare_models.py` 会生成一张“原图｜U-Net｜RGB Pix2Pix｜45/110 边缘 Pix2Pix”的四栏图。输出为 [`face2sketch/results/111_comparison.png`](face2sketch/results/111_comparison.png)。输入不是正方形时，可加 `--resize-mode center_crop` 保持人脸比例；默认 `stretch` 与训练时缩放方式一致。

详细代码结构与已记录的测试指标见 [`face2sketch/README.md`](face2sketch/README.md)。
