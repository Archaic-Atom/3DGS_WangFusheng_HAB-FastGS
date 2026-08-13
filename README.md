### Hardware-Aware Budgeting for Fast 3D Gaussian Splatting

在 [FastGS](https://github.com/fastgs/FastGS) 训练框架内实施显式 Gaussian 预算控制，面向受限硬件上的模型规模、质量与渲染速度折中。

## 方法概览

HAB-FastGS 不改变 3D Gaussian Splatting 的表示和 FastGS rasterizer，而是在训练期的 densification 事件中加入预算约束：

1. 先按原始 FastGS 配方训练 seed 0 baseline，得到场景基准点数 `N_base`。
2. 固定该场景目标 `B = floor(0.82 × N_base)`，并对其他 seed 复用同一个目标。
3. 每次 densification 前，如果当前 Gaussian 数超过 `B`，从统一候选池中优先删除低 opacity Gaussian。
4. 最终保存前执行精确点数检查，保证输出达到固定目标。
5. `hab_stats.csv` 记录点数、可见率、候选数、预算裁剪量以及 rasterizer 负载代理量。

```text
hab_mode             = gaussian_budget
hab_priority_mode    = opacity_only
hab_prune_placement  = pre_densify
hab_budget_schedule  = per_event
target ratio         = 0.82
exact final count    = enabled
```

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

创建独立 conda 环境：

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

## 致谢与许可

本项目建立在 [FastGS](https://github.com/fastgs/FastGS) 和原始 [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) 代码基础上。
