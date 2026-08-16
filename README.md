# HAB-FastGS

HAB-FastGS is a budget-aware training extension for [FastGS](https://github.com/fastgs/FastGS). It explicitly controls the Gaussian population while preserving the original FastGS path when HAB is disabled.

HAB-FastGS 是 [FastGS](https://github.com/fastgs/FastGS) 的预算感知训练扩展。它在训练期间显式控制 Gaussian 数量，并在关闭 HAB 时保持原始 FastGS 运行路径不变。

This repository contains source code and evaluation utilities only. Datasets, trained models, and runtime logs are intentionally excluded.

本仓库仅包含源代码和评测工具，不包含数据集、训练模型和运行日志。

## Highlights / 主要特性

- **In-training Gaussian budget enforcement** / **训练期 Gaussian 预算控制**
- **Exact final-count mode for matched-budget comparisons** / **面向等预算比较的精确最终点数模式**
- **Configurable pruning priority and scheduling** / **可配置的裁剪优先级与调度策略**
- **Optional load-aware dynamic target controller** / **可选的负载感知动态目标控制器**
- **Deterministic seed selection and structured `hab_stats.csv` logging** / **确定性随机种子与结构化 `hab_stats.csv` 日志**
- **Strict PSNR, SSIM, LPIPS, and synchronized renderer-FPS utilities** / **严格的 PSNR、SSIM、LPIPS 与同步渲染 FPS 工具**

All HAB options are disabled by default. Commands without `--hab_*` arguments retain the original FastGS behavior.

所有 HAB 选项默认关闭；不传入 `--hab_*` 参数时，程序保持原始 FastGS 行为。

## Installation / 环境安装

The bundled `environment.yml` follows the upstream FastGS environment. A CUDA 11.8 environment can also be created manually.

仓库中的 `environment.yml` 沿用上游 FastGS 环境，也可以手动创建 CUDA 11.8 环境：

```bash
git clone https://github.com/Archaic-Atom/3DGS_WangFusheng_HAB-FastGS.git
cd 3DGS_WangFusheng_HAB-FastGS

conda create -y -n habfastgs python=3.8 pip
conda activate habfastgs

pip install torch==2.0.0 torchvision==0.15.1 \
  --index-url https://download.pytorch.org/whl/cu118
pip install plyfile tqdm websockets opencv-python scipy lpips

export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.6

pip install -e submodules/simple-knn
pip install -e submodules/fused-ssim
pip install -e submodules/diff-gaussian-rasterization_fastgs
```

If CUDA 11.8 is unavailable, select a PyTorch/CUDA combination compatible with your GPU.

如果无法使用 CUDA 11.8，请选择与本机 GPU 兼容的 PyTorch/CUDA 组合。

## Data layout / 数据布局

HAB-FastGS accepts the same COLMAP and 3DGS dataset layouts as FastGS. A typical local layout is shown below.

HAB-FastGS 接受与 FastGS 相同的 COLMAP 和 3DGS 数据布局。典型的本地目录结构如下：

```text
datasets/
├── mipnerf360/
│   ├── bicycle/
│   ├── garden/
│   └── room/
├── tanksandtemples/
│   ├── train/
│   └── truck/
└── db/
    ├── drjohnson/
    └── playroom/
```

Keep generated models outside the dataset directory. Dataset and output paths are ignored by Git.

请将生成的模型保存在数据集目录之外；数据和输出路径已由 Git 忽略。

## Training / 训练

### Original FastGS path / 原始 FastGS 路径

Use the scene-specific FastGS arguments in [`train_base.sh`](train_base.sh). The following example trains the `truck` scene.

请使用 [`train_base.sh`](train_base.sh) 中针对各场景的 FastGS 参数。下面以 `truck` 场景为例：

```bash
python train.py \
  -s /path/to/datasets/tanksandtemples/truck \
  -m output/truck_fastgs \
  --eval --iterations 30000 --seed 0 \
  --densification_interval 500 \
  --optimizer_type default \
  --test_iterations 30000 \
  --highfeature_lr 0.04 \
  --grad_abs_thresh 0.0009 \
  --mult 0.7
```

### Fixed Gaussian budget / 固定 Gaussian 预算

Choose an absolute target count for the scene and enable budget enforcement.

为场景指定一个绝对目标点数，并启用预算控制：

```bash
python train.py \
  -s /path/to/datasets/tanksandtemples/truck \
  -m output/truck_hab \
  --eval --iterations 30000 --seed 0 \
  --densification_interval 500 \
  --optimizer_type default \
  --test_iterations 30000 \
  --highfeature_lr 0.04 \
  --grad_abs_thresh 0.0009 \
  --mult 0.7 \
  --hab_mode gaussian_budget \
  --hab_target_gaussians 1700000 \
  --hab_priority_mode joint \
  --hab_prune_placement pre_densify \
  --hab_budget_schedule per_event \
  --hab_exact_final_count
```

`--hab_target_gaussians` is an absolute count rather than a percentage. Reuse the same target for runs that must be count-matched.

`--hab_target_gaussians` 表示绝对点数而不是百分比。需要进行等点数比较时，应在各次运行中复用同一目标值。

### Load-aware target / 负载感知目标

The optional controller adjusts the working target within configured bounds.

可选控制器会在设定的上下界内动态调整工作目标：

```bash
python train.py \
  -s /path/to/scene \
  -m output/scene_hab_load \
  --eval --seed 0 \
  --hab_mode load_budget \
  --hab_target_gaussians 1700000 \
  --hab_load_target_ratio 1.0 \
  --hab_load_min_scale 0.85 \
  --hab_load_max_scale 1.10
```

## HAB options / HAB 参数

| Option / 参数 | Default / 默认值 | Description / 说明 |
|---|---:|---|
| `--hab_mode` | `off` | `off`, `gaussian_budget`, or `load_budget` / 关闭、固定预算或负载预算模式 |
| `--hab_target_gaussians` | `0` | Absolute Gaussian target / Gaussian 绝对目标点数 |
| `--hab_min_target_gaussians` | `0` | Optional lower bound for dynamic targets / 动态目标的可选下界 |
| `--hab_budget_start_iter` | `500` | First iteration eligible for budget enforcement / 开始执行预算控制的迭代 |
| `--hab_max_prune_fraction` | `0.10` | Maximum fraction removed by one budget event / 单次预算事件的最大裁剪比例 |
| `--hab_priority_mode` | `joint` | `joint`, `opacity_only`, `score_only`, `radii_only`, or `random` / 联合、单信号或随机排序策略 |
| `--hab_prune_placement` | `pre_densify` | Apply the budget before or after densification / 在 densification 前或后执行预算控制 |
| `--hab_budget_schedule` | `per_event` | `per_event`, `ramp`, `final_only`, or `at_end` / 按事件、渐进、末次事件或训练末尾调度 |
| `--hab_exact_final_count` | disabled / 关闭 | Enforce the requested final count when feasible / 在可行时强制达到最终目标点数 |
| `--hab_log_interval` | `500` | Statistics logging interval / 统计日志记录间隔 |

Training writes controller and population statistics to `hab_stats.csv` in the model directory.

训练过程会将控制器状态和 Gaussian 数量统计写入模型目录下的 `hab_stats.csv`。

## Rendering / 渲染

Render the saved model with the matching scene and FastGS scale arguments.

使用对应的场景路径和 FastGS 缩放参数渲染已保存模型：

```bash
python render.py \
  -s /path/to/scene \
  -m output/truck_hab \
  --iteration 30000 \
  --skip_train \
  --quiet \
  --mult 0.7
```

## Unified image metrics / 统一图像指标

The evaluator pairs rendered and ground-truth images by filename. Missing files, mismatched names, or inconsistent resolutions are treated as errors.

评测器按照文件名配对渲染图和真值图；文件缺失、名称不匹配或分辨率不一致都会直接报错。

```bash
python benchmark/unified_metrics.py \
  --renders output/truck_hab/test/ours_30000/renders \
  --gt output/truck_hab/test/ours_30000/gt \
  --output output/truck_hab/unified_metrics.json \
  --tag truck_hab
```

## Synchronized FPS / 同步 FPS

Use CUDA-synchronized timing with explicit warm-up and measurement repeats.

使用 CUDA 同步计时，并显式设置预热次数和正式测量次数：

```bash
python benchmark/benchmark_sync_fps.py \
  -s /path/to/scene \
  -m output/truck_hab \
  --renderer fastgs \
  --eval --quiet --iteration 30000 \
  --mult 0.7 \
  --warmup_repeats 3 \
  --measure_repeats 30 \
  --output_json output/truck_hab/benchmark_sync_fps.json
```

The benchmark synchronizes CUDA around timed regions and stores every repeat-level measurement in JSON.

该工具会在计时区间前后同步 CUDA，并将每次重复测量结果写入 JSON。

## Repository structure / 仓库结构

```text
.
├── arguments/                  # Command-line options / 命令行参数
├── scene/gaussian_model.py     # Gaussian and budget operations / Gaussian 与预算操作
├── train.py                    # Training, controller, seed, logs / 训练、控制器、种子与日志
├── gaussian_renderer/          # FastGS renderer integration / FastGS 渲染器集成
├── benchmark/
│   ├── unified_metrics.py      # PSNR, SSIM, and LPIPS / 图像质量指标
│   └── benchmark_sync_fps.py   # CUDA-synchronized FPS / CUDA 同步 FPS
├── submodules/                 # Rasterizer, KNN, fused SSIM / 光栅器、KNN、融合 SSIM
├── train_base.sh               # Scene command examples / 场景命令示例
└── environment.yml             # Upstream-compatible environment / 上游兼容环境
```

## Acknowledgements and license / 致谢与许可证

HAB-FastGS builds on [FastGS](https://github.com/fastgs/FastGS) and [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).

HAB-FastGS 基于 [FastGS](https://github.com/fastgs/FastGS) 和 [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) 构建。

See [`LICENSE`](LICENSE) and [`LICENSE_ORIGINAL.md`](LICENSE_ORIGINAL.md) for the applicable terms. Retain the upstream notices when redistributing modified source.

适用条款请参阅 [`LICENSE`](LICENSE) 和 [`LICENSE_ORIGINAL.md`](LICENSE_ORIGINAL.md)。重新分发修改后的源代码时，请保留上游版权与许可声明。
