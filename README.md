<div align="center">

# HAB-FastGS

### Hardware-Aware Budgeting for Fast 3D Gaussian Splatting

在 [FastGS](https://github.com/fastgs/FastGS) 训练框架内实施显式 Gaussian 预算控制，面向受限硬件上的模型规模、质量与渲染速度折中。

</div>

> [!IMPORTANT]
> 本仓库公开的是论文实验冻结版本 `20260803-opacity-per-event-82-v4`。正式结果支持“固定 18% Gaussian 压缩与有限场景域的资源—质量折中”，不支持广义感知质量非劣、必然训练加速或跨渲染器的普遍领先。完整门槛与失败项均在本文和 [`docs/experiments`](docs/experiments/) 中披露。

## 方法概览

HAB-FastGS 不改变 3D Gaussian Splatting 的表示和 FastGS rasterizer，而是在训练期的 densification 事件中加入预算约束：

1. 先按原始 FastGS 配方训练 seed 0 baseline，得到场景基准点数 `N_base`。
2. 固定该场景目标 `B = floor(0.82 × N_base)`，并对其他 seed 复用同一个目标。
3. 每次 densification 前，如果当前 Gaussian 数超过 `B`，从统一候选池中优先删除低 opacity Gaussian。
4. 最终保存前执行精确点数检查，保证输出达到固定目标。
5. `hab_stats.csv` 记录点数、可见率、候选数、预算裁剪量以及 rasterizer 负载代理量。

开发阶段比较了 joint、opacity-only、score-only、radii-only、random、前/后置裁剪、per-event/ramp/final-only 等构造基线。预注册选择最终得到：

```text
hab_mode             = gaussian_budget
hab_priority_mode    = opacity_only
hab_prune_placement  = pre_densify
hab_budget_schedule  = per_event
target ratio         = 0.82
exact final count    = enabled
```

joint 优先级在开发集上未通过预注册 FPS、PSNR 和最差 seed 门槛，因此不作为正式默认方法；load-feedback 只保留为敏感性实验。

## 正式实验结论

### FastGS 配对主实验

确认集为 `train / drjohnson / garden / room`，每个场景使用 seed `{0,1,2}`。质量指标统一采用 FastGS PSNR/SSIM 和 VGG-LPIPS；FPS 使用同一 GPU、同一进程、共享 Scene 的交错配对测量。以下均为 `HAB − FastGS`：

| 指标 | 层级配对估计 | 95% CI | 预注册门槛 |
|---|---:|---:|:---:|
| PSNR | -0.0626 dB | [-0.1304, +0.0090] | PASS |
| SSIM | -0.00299 | [-0.00455, -0.00139] | PASS |
| LPIPS（越低越好） | +0.00750 | [+0.00383, +0.01168] | **FAIL** |
| 训练 wall time | -6.08% | [-8.15%, -4.08%] | **FAIL**，门槛为至少 -10% |
| native FPS | +5.76% | [+4.15%, +7.63%] | **FAIL**，CI 下界须 >5% |

精确点数、无场景平均变慢和无灾难性质量下降检查均通过。因 LPIPS、wall-time 和 FPS 置信区间门槛未全部通过，论文级结论限定为：

> 在冻结的四场景确认域内，HAB-FastGS 用约 18% Gaussian 资源削减换取小幅 PSNR/SSIM 代价，并观察到约 5.8% 的配对 native FPS 提升。

### 统一第三方对照

下表覆盖 `truck / train / playroom / drjohnson`。质量列为四场景算术均值；FPS 与 wall 为几何均值；Gaussian 数为场景均值。不同方法使用各自原生 rasterizer，因此跨方法 FPS 是统一空闲 GPU 上的描述性比较，而不是同进程配对推断。

| 方法 | PSNR | SSIM | LPIPS ↓ | FPS ↑ | wall (s) ↓ | Gaussian 数 |
|---|---:|---:|---:|---:|---:|---:|
| FastGS | 27.0932 | 0.87404 | 0.23748 | 395.2 | 242.9 | 229,989 |
| **HAB-FastGS** | **26.9339** | **0.87172** | **0.24333** | **408.1** | **233.3** | **188,801** |
| Vanilla 3DGS | 26.7892 | 0.88022 | 0.20360 | 100.4 | 1580.5 | 2,027,865 |
| Speedy-Splat | 26.4860 | 0.86140 | 0.25391 | 437.2 | 915.5 | 216,048 |
| Taming-3DGS | 26.4435 | 0.85735 | 0.26204 | 219.6 | 362.4 | 188,795 |
| Mini-Splatting | 26.7178 | 0.87855 | 0.21058 | 482.0 | 1202.6 | 427,047 |
| DashGaussian | 27.1462 | 0.87982 | 0.21391 | 126.6 | 371.9 | 1,573,590 |
| ShorterSplatting | 26.4482 | 0.87158 | 0.20815 | 209.6 | 276.3 | 2,317,280 |

相对几乎精确点数匹配的冻结 Taming-3DGS 配置，HAB-FastGS 在四个场景逐场景获得：

- PSNR：+0.102 至 +0.961 dB；
- LPIPS：改善 0.0109 至 0.0345；
- native FPS：+65.6% 至 +103.9%；
- 训练 wall ratio：0.542 至 0.742。

质量与 FPS 门槛逐场景通过，但预注册的 `wall ≤ 0.5 × Taming` 门槛全部失败。因此允许的表述是“冻结配置下相对点数匹配 Taming 的质量/native-FPS 优势”，不能表述为至少 2× 训练加速。

### 机制证据

- 34/34 机制 collector 和 12/12 同步 FPS 作业完成。
- 点数匹配时，opacity-only 相对 joint 在 `truck` 高 0.2387 dB，而在 `playroom` 低 0.2693 dB；优先级效果具有场景依赖性。
- score-only / radii-only 在精确点数下损失 5.4–8.8 dB，random 在 `truck` 损失 0.565 dB。
- 将一次性精确裁剪从 27k 推迟到 30k，会在 `truck` 损失 1.700–1.826 dB，在 `playroom` 损失 0.654 dB。
- `ratio ∈ {1.0, 0.9, 0.82, 0.7}` Pareto 扫描已完成。

完整统计、逐场景结果、协议、失败恢复和哈希清单见 [`docs/experiments/SUBMISSION_EXPERIMENT_REPORT.md`](docs/experiments/SUBMISSION_EXPERIMENT_REPORT.md)。

## 环境配置

正式运行环境：

| 组件 | 版本 |
|---|---|
| OS | WSL2 / Ubuntu 22.04 |
| Python | 3.8.20 |
| PyTorch | 2.0.0+cu118 |
| CUDA Toolkit | 11.8 |
| GCC/G++ | 11.4 |
| 正式实验 GPU | NVIDIA RTX 3080 20 GB (`sm_86`) |

推荐在 WSL-Ubuntu 中创建独立 conda 环境：

```bash
git clone https://github.com/Archaic-Atom/3DGS_WangFusheng_HAB-FastGS.git
cd 3DGS_WangFusheng_HAB-FastGS

conda create -y -n habfastgs python=3.8.20 pip
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

`environment.yml` 保留上游 FastGS 环境供兼容参考；正式论文数值来自上表所列的 Python 3.8 / PyTorch 2.0 / CUDA 11.8 冻结环境。

## 数据准备

支持标准 COLMAP/3DGS 数据布局。正式实验使用 Tanks&Temples、Deep Blending 和 Mip-NeRF 360 场景。

```text
datasets/
├── tandt/
│   ├── truck/
│   └── train/
├── db/
│   ├── playroom/
│   └── drjohnson/
└── m360/
    ├── bicycle/
    ├── garden/
    ├── room/
    └── counter/
```

不要在权威数据目录中生成 COLMAP 文件、派生 PLY、缓存或训练输出。建议将输入数据只读复制到 WSL ext4，再将输出写入独立目录：

```bash
rsync -a --chmod=a-w /mnt/<drive>/datasets/tandt/truck/ ~/hab-data/tandt/truck/
mkdir -p ~/hab-runs
```

正式审计覆盖 8 个场景，source / WSL mirror / derived 清单逐字节一致；原始数据没有被原地修改。

## 训练

### 1. 训练 FastGS seed-0 基准

不同场景应沿用上游 `train_base.sh` 中对应的 FastGS 参数。以 `truck` 为例：

```bash
python train.py \
  -s ~/hab-data/tandt/truck \
  -m ~/hab-runs/truck_fastgs_s0 \
  --eval --iterations 30000 --seed 0 \
  --densification_interval 500 --optimizer_type default \
  --test_iterations 30000 --save_iterations 30000 \
  --highfeature_lr 0.04 --grad_abs_thresh 0.0009 --mult 0.7
```

从最终 PLY 或 `hab_stats.csv` 读取 FastGS seed-0 Gaussian 数 `N_base`，并计算：

```text
target = floor(0.82 * N_base)
```

### 2. 训练冻结 HAB 配置

```bash
python train.py \
  -s ~/hab-data/tandt/truck \
  -m ~/hab-runs/truck_hab_s0 \
  --eval --iterations 30000 --seed 0 \
  --densification_interval 500 --optimizer_type default \
  --test_iterations 30000 --save_iterations 30000 \
  --highfeature_lr 0.04 --grad_abs_thresh 0.0009 --mult 0.7 \
  --hab_mode gaussian_budget \
  --hab_target_gaussians <TARGET_FROM_FASTGS_SEED0> \
  --hab_priority_mode opacity_only \
  --hab_prune_placement pre_densify \
  --hab_budget_schedule per_event \
  --hab_exact_final_count
```

正式多 seed 实验必须复用同一个场景 seed-0 target，不能为每个 seed 重新计算目标。

## 渲染与统一评测

```bash
# 生成严格 test renders / GT
python render.py -m ~/hab-runs/truck_hab_s0 \
  --iteration 30000 --skip_train --quiet --mult 0.7

# 统一 FastGS PSNR / SSIM / VGG-LPIPS；文件名集合或分辨率不一致会直接失败
python benchmark/unified_metrics.py \
  --renders ~/hab-runs/truck_hab_s0/test/ours_30000/renders \
  --gt ~/hab-runs/truck_hab_s0/test/ours_30000/gt \
  --output ~/hab-runs/truck_hab_s0/unified_metrics.json \
  --tag truck_hab_s0

# 同步 native FPS，正式口径为 3 warmup + 30 measured passes
python benchmark/benchmark_sync_fps.py \
  -m ~/hab-runs/truck_hab_s0 --renderer fastgs --iteration 30000 \
  --mult 0.7 --warmup_repeats 3 --measure_repeats 30 \
  --output_json ~/hab-runs/truck_hab_s0/benchmark_sync_fps.json --quiet
```

## 主要参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--hab_mode` | `off` | `off` 保持原始 FastGS；正式 HAB 使用 `gaussian_budget` |
| `--hab_target_gaussians` | `0` | 场景固定 Gaussian 预算 |
| `--hab_budget_start_iter` | `500` | 开始施加预算的 iteration |
| `--hab_max_prune_fraction` | `0.10` | 单次事件最多裁剪当前点数比例 |
| `--hab_priority_mode` | `joint` | 正式冻结配置必须显式设为 `opacity_only` |
| `--hab_prune_placement` | `pre_densify` | 预算裁剪相对 densification 的位置 |
| `--hab_budget_schedule` | `per_event` | `per_event / ramp / final_only / at_end` |
| `--hab_exact_final_count` | false | 正式运行必须启用精确最终点数 |
| `--hab_log_interval` | `500` | `hab_stats.csv` 记录间隔 |

HAB 参数默认关闭，不传任何 `--hab_*` 参数时保持 FastGS baseline 行为。

## 仓库结构

```text
.
├── arguments/                  # HAB CLI 参数
├── scene/gaussian_model.py     # 预算候选、排序与精确裁剪
├── train.py                    # 控制器、seed、日志与调度
├── gaussian_renderer/          # FastGS rasterizer 接口与负载统计
├── benchmark/
│   ├── unified_metrics.py      # 统一 PSNR/SSIM/LPIPS
│   └── benchmark_sync_fps.py   # 同步 render-only FPS
└── docs/experiments/           # 协议、正式表格、失败账本、哈希清单
```

## 可复现记录

- [`SUBMISSION_EXPERIMENT_REPORT.md`](docs/experiments/SUBMISSION_EXPERIMENT_REPORT.md)：独立投稿实验报告。
- [`EXPERIMENT_PROTOCOL.md`](docs/experiments/EXPERIMENT_PROTOCOL.md)：冻结前选择、确认门槛与统一评测协议。
- [`FORMAL_MAIN_V4.md`](docs/experiments/FORMAL_MAIN_V4.md)：FastGS/HAB 正式主表。
- [`FORMAL_THIRDPARTY_COMPARISON.md`](docs/experiments/FORMAL_THIRDPARTY_COMPARISON.md)：4 场景 × 8 方法统一表。
- [`FORMAL_MECHANISMS_V4.md`](docs/experiments/FORMAL_MECHANISMS_V4.md)：构造基线、恢复窗口、Pareto 和 load sensitivity。
- [`FAILURE_LEDGER.md`](docs/experiments/FAILURE_LEDGER.md)：失败、OOM 恢复、排除规则和可用性边界。
- [`ARTIFACT_SHA256SUMS.txt`](docs/experiments/ARTIFACT_SHA256SUMS.txt)：全部紧凑记录的 SHA-256 清单。

## 局限

- 正式统计确认域只有四个场景，不能外推为所有 3DGS 数据分布上的普适结论。
- 当前主方法本质上仍是 Gaussian-count budget；`num_rendered / num_buckets` 是负载代理，尚未形成 tile-list/FPS 闭环。
- LPIPS 预注册上界失败，说明 opacity-only 预算不包含显式感知质量保护。
- 减少 Gaussian 数不保证任意场景、任意 rasterizer 都按同比例提升 FPS。
- 跨方法 wall time 包含各仓库自身流程差异，只作完整官方调用的描述性比较。

## 致谢与许可

本项目建立在 [FastGS](https://github.com/fastgs/FastGS) 和原始 [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) 代码基础上。请同时遵守 [`LICENSE`](LICENSE) 与 [`LICENSE_ORIGINAL.md`](LICENSE_ORIGINAL.md)，并在学术使用时引用对应上游论文。

HAB-FastGS 的实验记录冻结于 2026-08-03；仓库中的数值均来自同一 RTX 3080 20 GB、WSL-Ubuntu 和 FastGS 统一评测口径。
