# HAB-FastGS

HAB-FastGS is a budget-aware training extension for [FastGS](https://github.com/fastgs/FastGS). It adds explicit control over the Gaussian population while preserving the original FastGS path when HAB is disabled.

This repository contains source code and evaluation utilities only. Datasets, trained models, and runtime logs are intentionally excluded.

## Highlights

- In-training Gaussian budget enforcement.
- Exact final-count mode for matched-budget comparisons.
- Configurable pruning priority and scheduling.
- Optional load-aware dynamic target controller.
- Deterministic seed selection and structured `hab_stats.csv` logging.
- Strict PSNR, SSIM, LPIPS, and synchronized renderer-FPS utilities.

All HAB options default to disabled. Commands that do not pass `--hab_*` arguments retain the original FastGS behavior.

## Installation

The bundled `environment.yml` follows the upstream FastGS environment. A CUDA 11.8 setup can also be created manually:

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

Use a PyTorch/CUDA combination compatible with your GPU if CUDA 11.8 is not available.

## Data layout

HAB-FastGS accepts the same COLMAP and 3DGS dataset layouts as FastGS. A typical local structure is:

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

## Training

### Original FastGS path

Use the scene-specific FastGS arguments from [`train_base.sh`](train_base.sh). For example:

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

### Fixed Gaussian budget

Choose an absolute target count for the scene and enable budget enforcement:

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

`--hab_target_gaussians` is an absolute count, not a percentage. Use the same target for every run that must be count-matched.

### Load-aware target

The optional controller adjusts the working target inside configured bounds:

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

## HAB options

| Option | Default | Description |
|---|---:|---|
| `--hab_mode` | `off` | `off`, `gaussian_budget`, or `load_budget` |
| `--hab_target_gaussians` | `0` | Absolute Gaussian target |
| `--hab_min_target_gaussians` | `0` | Optional lower bound for dynamic targets |
| `--hab_budget_start_iter` | `500` | First iteration eligible for budget enforcement |
| `--hab_max_prune_fraction` | `0.10` | Maximum fraction removed by one budget event |
| `--hab_priority_mode` | `joint` | `joint`, `opacity_only`, `score_only`, `radii_only`, or `random` |
| `--hab_prune_placement` | `pre_densify` | Apply the budget before or after densification |
| `--hab_budget_schedule` | `per_event` | `per_event`, `ramp`, `final_only`, or `at_end` |
| `--hab_exact_final_count` | disabled | Enforce the requested final count when feasible |
| `--hab_log_interval` | `500` | Statistics logging interval |

Training writes controller and population statistics to `hab_stats.csv` in the model directory.

## Rendering

```bash
python render.py \
  -s /path/to/scene \
  -m output/truck_hab \
  --iteration 30000 \
  --skip_train \
  --quiet \
  --mult 0.7
```

## Unified image metrics

The evaluator pairs render and ground-truth images by filename and rejects missing or mismatched images:

```bash
python benchmark/unified_metrics.py \
  --renders output/truck_hab/test/ours_30000/renders \
  --gt output/truck_hab/test/ours_30000/gt \
  --output output/truck_hab/unified_metrics.json \
  --tag truck_hab
```

## Synchronized FPS

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

The benchmark synchronizes CUDA around timed regions and stores all repeat-level measurements in JSON.

## Repository structure

```text
.
├── arguments/                  # FastGS and HAB command-line options
├── scene/gaussian_model.py     # Gaussian population and budget operations
├── train.py                    # Training loop, controller, seed, and logging
├── gaussian_renderer/          # FastGS renderer integration
├── benchmark/
│   ├── unified_metrics.py      # PSNR, SSIM, and LPIPS
│   └── benchmark_sync_fps.py   # CUDA-synchronized renderer timing
├── submodules/                 # Rasterizer, KNN, and fused SSIM extensions
├── train_base.sh               # Upstream scene-specific command examples
└── environment.yml             # Upstream-compatible environment
```

## Acknowledgements and license

HAB-FastGS builds on [FastGS](https://github.com/fastgs/FastGS) and [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).

See [`LICENSE`](LICENSE) and [`LICENSE_ORIGINAL.md`](LICENSE_ORIGINAL.md) for the applicable terms. Retain the upstream notices when redistributing modified source.
