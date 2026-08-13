# HAB-FastGS submission experiment report

Status: complete formal record, 2026-08-03 (Asia/Shanghai).

## Executive disposition

HAB-FastGS uses the frozen profile `opacity_only / per_event / pre_densify`
with an exact per-scene target `floor(0.82 * FastGS seed-0 N)`. The method is
reproducibly useful as an 18% Gaussian compression and resource-quality
trade-off. On four independent confirmation scenes it preserves PSNR and SSIM
within the preregistered bounds and improves paired synchronized FPS by 5.76%
on average, but the lower confidence bound is 4.15%, LPIPS narrowly misses its
upper-bound gate, and training wall improves by 6.08% rather than the required
10%. Therefore a broad non-inferiority, guaranteed render acceleration, or
>=10% training-speed claim is not supported.

Against the frozen count-matched Taming-3DGS configuration on truck, train,
playroom, and drjohnson, HAB improves PSNR, LPIPS, and descriptive native FPS
on every scene. The preregistered broad comparison gate nevertheless fails
because it additionally requires HAB training wall to be at most half of
Taming's; observed ratios are 0.542--0.742. The defensible comparison claim is
same-count quality/native-FPS superiority over this Taming configuration, not
>=2x training acceleration.

## Completeness

- Development selection: 18/18 runs; truck/playroom, seeds 0/1/2, priority and
  schedule decisions frozen before confirmation.
- Main formal matrix: 28/28 strict collectors; four confirmation scenes x two
  arms x three seeds plus two fixed-seed extensions.
- Mechanism matrix: 34/34 strict collectors and 12/12 interleaved FPS jobs.
- Five-method external matrix: 20/20 train/render/evaluation runs and 20/20
  native-FPS strict collectors.
- ShorterSplatting: 4/4 original-scene collectors and 4/4 synchronized native
  FPS records.
- Unified original-scene comparison: exactly 32 rows = 4 scenes x 8 methods.

The authoritative dataset tree was never modified in place. Failures and
recoveries are enumerated in `FAILURE_LEDGER.md` rather than silently replaced.

## Reproducible runtime and data

| Component | Accepted value |
|---|---|
| WSL | Ubuntu 22.04 |
| Python | 3.8.20 |
| PyTorch | 2.0.0+cu118 |
| CUDA toolkit | 11.8 |
| GCC/G++ | 11.4 |
| GPU | NVIDIA RTX 3080 20 GB, `sm_86` |
| FastGS/HAB freeze | `20260803-opacity-per-event-82-v4` |
| Five-method freeze | `20260803-thirdparty-five-v1` |
| Shorter freeze | `20260803-shorter-splatting-s0-v1` |
| Authoritative input | `H:\WorkSpace\3DGS-GPT56\datasets` (read-only policy) |
| WSL training mirror | `/home/cute_cat/hab-fastgs/datasets` |

FastGS/HAB frozen source is
`/home/cute_cat/hab-paper-freeze/20260803-opacity-per-event-82-v4/source`;
its conda prefix is
`/home/cute_cat/anaconda3/envs/habfastgs-paper-20260803-opacity-per-event-82-v4`.
The freeze manifest digest is
`03de8357f7faa6d1634c4fd15b7f1df7d0ec6b762cd05825e3829fb2c7c203b6`.

All eight used scenes have exact source/mirror file and byte equality. The
accepted data-manifest digest is
`7e10ff3ad3d708dba029ac64a307532f7bc9eca7322aa11df3fc571313041062`;
its JSON and TSV SHA-256 values are respectively
`40caa1de988f9e806327367577fbcf2a390c3ef750891d1422358be0d0fb9ab9`
and `7ed5fa24d56ca415af4d218f946f90d3837f822dc4f76e29f484a0c7bb3817b`.
Mip-NeRF 360 `points3D.ply` files are explicit derived overlays, not edits to
the authoritative tree.

## Frozen method selection

Joint priority failed the preregistered default gate: paired FPS versus opacity
was +3.347% with hierarchical 95% CI [1.337%, 5.128%], below the required +5%
lower bound; truck mean PSNR was 0.2131 dB lower and the worst seed lost 0.2883
dB. The fixed ramp also failed: aggregate PSNR changed by -0.1119 dB, 95% CI
[-0.3485, 0.0296], and FPS changed by +0.131%, CI [-2.010%, 2.048%]. The frozen
default is consequently opacity-only with per-event budgeting. No confirmation
result was used to retune it.

## Main confirmation result

Hierarchical paired inference resamples scenes first and seeds within scene
(50,000 resamples). FPS uses a shared Scene and one process with balanced arm
order, three warmups, and 30 measured passes. Per-run screening FPS is excluded.

| Metric, HAB - FastGS | Estimate | 95% CI | Gate |
|---|---:|---:|---|
| PSNR | -0.0626 dB | [-0.1304, +0.0090] | PASS |
| SSIM | -0.00299 | [-0.00455, -0.00139] | PASS |
| LPIPS | +0.00750 | [+0.00383, +0.01168] | FAIL |
| Training wall | -6.08% | [-8.15%, -4.08%] | FAIL (required >=10%) |
| Paired FPS | +5.76% | [+4.15%, +7.63%] | FAIL (lower bound required >5%) |

Exact target count, no-slower-confirmation-scene, and no-catastrophic-scene
gates pass. Absolute per-scene/seed summaries and every paired row are in
`FORMAL_MAIN_V4.md`, `FORMAL_MAIN_V4_STATS.json`, and
`FORMAL_MAIN_V4_PAIRS.csv`.

## Mechanism evidence

- Count-matched opacity versus joint is scene-dependent: opacity is +0.2387 dB
  on truck; joint is +0.2693 dB on playroom.
- Score-only and radii-only priorities lose 5.4--8.8 dB at identical count.
  Random pruning loses 0.565 dB on truck. These are strong negative controls.
- Moving one-shot exact pruning from 27k to 30k loses 1.700--1.826 dB on truck
  over seeds 0--2 and 0.654 dB on playroom; late one-shot recovery is rejected.
- Frozen Pareto points cover ratios 1.00/0.90/0.82/0.70 on truck and playroom.
- Reversed load feedback reduces achieved point count on all four tested scenes
  but changes quality inconsistently; it is sensitivity evidence only.

The full tables are in `FORMAL_MECHANISMS_V4.md` and the machine-readable rows
and trace hashes are in `FORMAL_MECHANISMS_V4.json`.

## Unified external comparison

The following are four-scene descriptive summaries. Quality uses one frozen
FastGS evaluator. FPS is each method's native renderer on the same idle GPU,
but cross-rasterizer values are not same-process paired inference. Geometric
means are used for FPS and wall.

| Method | mean PSNR | mean SSIM | mean LPIPS | geo. FPS | geo. train s | mean N |
|---|---:|---:|---:|---:|---:|---:|
| FastGS | 27.0932 | 0.87404 | 0.23748 | 395.2 | 242.9 | 229,989 |
| HAB | 26.9339 | 0.87172 | 0.24333 | 408.1 | 233.3 | 188,801 |
| Vanilla 3DGS | 26.7892 | 0.88022 | 0.20360 | 100.4 | 1,580.5 | 2,027,865 |
| Speedy-Splat | 26.4860 | 0.86140 | 0.25391 | 437.2 | 915.5 | 216,048 |
| Taming matched | 26.4435 | 0.85735 | 0.26204 | 219.6 | 362.4 | 188,795 |
| Mini-Splatting | 26.7178 | 0.87855 | 0.21058 | 482.0 | 1,202.6 | 427,047 |
| DashGaussian | 27.1462 | 0.87982 | 0.21391 | 126.6 | 371.9 | 1,573,590 |
| ShorterSplatting | 26.4482 | 0.87158 | 0.20815 | 209.6 | 276.3 | 2,317,280 |

HAB versus count-matched Taming:

| Scene | dFPS | dPSNR | dLPIPS | HAB/Taming wall | all preregistered gates |
|---|---:|---:|---:|---:|---|
| truck | +85.5% | +0.961 | -0.0155 | 0.650 | FAIL |
| train | +65.6% | +0.668 | -0.0345 | 0.742 | FAIL |
| playroom | +103.9% | +0.231 | -0.0140 | 0.542 | FAIL |
| drjohnson | +90.5% | +0.102 | -0.0109 | 0.656 | FAIL |

Every failure is solely the preregistered `wall <= 0.5x` condition; FPS, PSNR,
and LPIPS conditions pass on all four scenes. The complete 32 rows, identities,
collector/FPS hashes, and gate booleans are in
`FORMAL_THIRDPARTY_COMPARISON.json` and its CSV/Markdown views.

## Supported and unsupported manuscript language

Supported:

- fixed 18% Gaussian compression under exact scene targets;
- limited-domain quality-resource trade-off versus FastGS;
- same-count quality and descriptive native-FPS superiority over the frozen
  Taming configuration on all four original scenes;
- reproducible negative findings for priority, pruning time, and load feedback.

Unsupported:

- broad perceptual non-inferiority versus FastGS;
- guaranteed per-scene or CI-lower-bound >5% rendering acceleration;
- >=10% FastGS training acceleration;
- >=2x Taming training acceleration;
- universal state-of-the-art quality or speed across rasterizers.

## Reproduction and audit entry points

- Preregistered protocol: `EXPERIMENT_PROTOCOL.md`
- Frozen target ledger: `TARGET_LEDGER_V4.json`
- Main statistics: `FORMAL_MAIN_V4_STATS.json`
- Mechanisms: `FORMAL_MECHANISMS_V4.json`
- Unified 32-row comparison: `FORMAL_THIRDPARTY_COMPARISON.json`
- Source/data/runtime narrative: `WORKLOG.md`
- Failures and recoveries: `FAILURE_LEDGER.md`
- Final file digests: `ARTIFACT_SHA256SUMS.txt`

Raw immutable runtime and record roots remain under `/home/cute_cat` in WSL;
the Windows `paper_artifacts` tree is the submission-facing index and compact
evidence package.
