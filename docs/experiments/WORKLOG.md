# HAB-FastGS experiment worklog

All timestamps are Asia/Shanghai unless explicitly marked UTC. This is a
progress ledger, not the final accepted-results table.

## 2026-08-03

- Read and reconciled the HAB-FastGS Notion overview and WSL rerun record.
- Audited the WSL2/CUDA/conda environment, the WSL-native source tree, existing
  experiment outputs, comparison repositories, and authoritative dataset root.
- Declared `/home/cute_cat/hab-fastgs/FastGS` authoritative; the Windows FastGS
  working copy is stale and is not a result-producing source tree.
- Verified Ubuntu 22.04 WSL2, Python 3.8.20, Torch 2.0.0+cu118, CUDA toolkit
  11.8, GCC/G++ 11.4, and RTX 3080 20 GB (`sm_86`).
- Added immutable-data Mip-NeRF 360 overlays for bicycle, garden, room, and
  counter. The raw-copy metadata digest was unchanged before/after overlay
  creation: `e58cc46b65a668b526d33a97664e4a83c3db6f37115004ae1b337efa9d377aec`.
- Fixed the 27k final-prune score/mask alignment and exact-budget prune
  attribution. Runtime assertions were exercised by bicycle and garden HAB
  pilots; both hit the requested count exactly.
- Fixed the Taming zero-sample boundary, corrected Dash's training entry point,
  corrected Mini's benchmark import root, and redesigned third-party smoke/full
  output isolation and strict acceptance. Formal reruns had not yet started at
  this point in the chronology and are completed in the later sections below.
- Strengthened paired FPS with balanced rotation and paired bootstrap; prepared
  immutable run IDs and per-scene GPU-state evidence.
- Strengthened the unified evaluator to require exact stem/resolution equality.
  The optimized/provenance-complete evaluator was deployed after the M360 queue.
  One-view equivalence testing produced exact zero differences for PSNR, SSIM,
  and LPIPS; the evaluator now hashes code, LPIPS/VGG assets, image manifests,
  resolutions, framework versions, CUDA/cuDNN, and GPU identity.

### M360 pilot observations

| Scene | Arm | Gaussians | Train wall (s) | PSNR | SSIM | LPIPS | screening FPS |
|---|---|---:|---:|---:|---:|---:|---:|
| bicycle | FastGS | 538,802 | 454.31 | 24.8520 | 0.71546 | 0.30828 | 266.13 |
| bicycle | joint HAB | 441,817 | 387.22 (recovered) | 24.7857 | recorded in unified JSON | recorded in unified JSON | 327.39 |
| garden | FastGS | 742,126 | 520.69 | 27.10944 | 0.83629 | 0.17372 | 244.80 |
| garden | joint HAB | 608,543 | 465.34 | 26.95145 | 0.83012 | 0.18740 | 285.82 |
| room | FastGS | 208,498 | 355.60 | 31.87477 | 0.92042 | 0.21675 | 327.53 |
| room | joint HAB | 170,968 | 322.69 | 31.41848 | 0.91434 | 0.22425 | 359.73 |
| counter | FastGS | 207,848 | 386.80 | 29.12235 | 0.90784 | 0.20297 | 306.72 |
| counter | joint HAB | 170,435 | 345.63 | 28.68882 | 0.90138 | 0.21317 | 323.52 |

Garden's joint pilot reduced count by 18.0%, wall time by 10.6%, and increased
the non-formal screening FPS by 16.8%, but its SSIM and LPIPS deltas fail the
predeclared perceptual-quality gates. It is therefore a pilot trade-off result,
not an accepted "nearly lossless" claim. Opacity-only and fixed-ramp profiles
will be selected on truck/playroom before confirmation scenes are used.

### Known recovery event

The bicycle joint-HAB Python training process completed normally and saved an
exact 441,817-point 30k PLY. While that shell was alive, its launcher file was
updated and Bash subsequently encountered a transient parse error in the
post-training section. No training code failed. The dedicated no-training
recovery path completed the render, strict unified evaluation, FPS screening,
and ledger tail. Training wall time is derived from the run-log birth timestamp
to PLY mtime (387.22 s) and remains marked as recovered provenance rather than
an instrumented launcher time. The PLY hash was recorded before recovery.

### Frozen development selection

- Completed all 18 preregistered selection runs: truck/playroom, seeds 0/1/2,
  joint, opacity-only, and fixed ramp. Every PLY exactly matched its fixed
  scene target (207,698 truck; 150,170 playroom).
- Completed six interleaved paired-FPS jobs (one per scene/seed), each with one
  shared Scene, balanced opacity/joint/ramp rotation, three warmups, and 30
  paired measured passes. The accepted FPS run ID is
  `dev-selection-v2-20260803T0736CST`; two earlier run IDs failed before timing
  and are retained as path-validation evidence.
- Joint minus opacity achieved +3.347% aggregate paired FPS, hierarchical 95%
  CI [1.337%, 5.128%], below the preregistered +5% lower-bound requirement.
  Truck mean PSNR was 0.2131 dB lower and the worst seed was 0.2883 dB lower.
- Ramp minus opacity/per-event changed aggregate PSNR by -0.1119 dB, 95% CI
  [-0.3485, 0.0296], and paired FPS by +0.131%, 95% CI [-2.010%, 2.048%].
- Frozen main profile before confirmation: `opacity_only`, `per_event`, exact
  target `floor(0.82 * FastGS seed-0 count)`. Joint and ramp remain negative
  ablations and will not be retuned.

### Third-party smoke validation

- Vanilla, Speedy-Splat, DashGaussian, and Taming-3DGS completed isolated
  train/render/exact-Ply/FPS/unified-metric/strict-collector smoke paths.
- Mini-Splatting trained and saved its requested 1k checkpoint, but its method
  intentionally stores only DC features before the 15k simplification stage;
  the official shared renderer therefore cannot render a 1k checkpoint. Its
  first end-to-end render validation must use a post-15k or formal 30k run.
- Fixed paired-FPS import-root and optional-argument normalization defects
  exposed by the first two no-timing startup attempts.

### Accepted immutable paper runtime

- Freeze attempts v1 and v2 are rejected: v1 retained absolute dependencies on
  the mutable base environment; v2 could not archive the unpublished exact
  `lit==15.0.7` wheel. v3 completed its byte manifest but is rejected for
  training because post-link `patchelf` mutation caused `simple-knn.distCUDA2`
  to raise a false CUDA OOM even at 1,000 points.
- A controlled three-way loader diagnostic showed that the original extension
  worked, while both post-link RUNPATH and forced-RPATH variants failed. A
  separate build proved that a relative RPATH emitted directly by the linker
  remained byte-stable and GPU-correct.
- Accepted freeze ID: `20260803-opacity-per-event-82-v4`. Its source and
  execution trees are mode 555, it contains no editable distributions, all
  three CUDA extensions contain `sm_86`, and their linker-emitted RPATH is
  `$ORIGIN/../torch/lib:$ORIGIN/torch/lib:$ORIGIN/../../..:$ORIGIN/../..`.
- The full freeze manifest and the source-file lock both verified after runtime
  QA. `simple-knn` succeeded on 500,000 points; fused SSIM produced finite
  forward values and backward gradients.
- The read-only frozen source completed a 20-iteration truck train/render/native
  metric/unified metric/synchronized-FPS pipeline. The exact 32-view unified
  smoke result was PSNR 12.6826657, SSIM 0.4737333, LPIPS 0.6482973; its FPS
  path completed 30 measured passes. These are runtime-QA values, not paper
  performance results.

### Frozen v4 confirmation result

- Completed all 28 required FastGS/HAB formal collectors: four confirmation
  scenes x two arms x three seeds, plus two extension scenes x two arms x seed
  zero. All 14 same-seed pairs are complete and every HAB PLY exactly equals
  its fixed `floor(0.82 * FastGS seed-0 N)` target.
- Formal paired-FPS run ID:
  `formal-paired-fps-v4-confirmation-s012-r3`. It contains 12 one-process,
  shared-Scene FastGS/HAB jobs, each with three warmups and 30 synchronized,
  interleaved measured passes. Wallpaper Engine was paused through its official
  control interface for this FPS window and immediately resumed; the five idle
  samples passed the <=10% utilization guard.
- Hierarchical paired confirmation inference (50,000 resamples): PSNR -0.0626
  dB, 95% CI [-0.1304, +0.0090]; SSIM -0.00299, CI [-0.00455, -0.00139];
  LPIPS +0.00750, CI [+0.00383, +0.01168]; training wall -6.08%, CI
  [-8.15%, -4.08%]; paired FPS +5.76%, CI [+4.15%, +7.63%].
- The exact-count, PSNR, SSIM, no-slower-scene, and no-catastrophic-scene gates
  pass. LPIPS upper-bound, FPS lower-bound >+5%, and wall-time >=10% gates fail.
  The paper claim is therefore limited to an 18% compression/resource-quality
  trade-off; broad non-inferiority or guaranteed render/training acceleration
  is not supported.
- Final machine-readable evidence is `FORMAL_MAIN_V4_STATS.json`; paper table
  is `FORMAL_MAIN_V4.md`. The provisional per-run screening FPS is excluded
  from inference.

### Formal failure/recovery ledger additions

- Confirmation queue outer cell reached its fixed 7,200-second terminal after
  the strict `room/FastGS/seed2` collector row had been written, but before the
  wrapper's final echo and evidence checksum. No training or metric process was
  killed. A validation-only recovery wrote `wrapper_recovery.json`, appended a
  recovered terminal marker, and generated `evidence.sha256`; PLY, metrics, FPS,
  and collector were not changed.
- Formal paired-FPS `r1` failed before any measurement because the launcher PATH
  omitted WSL's `/usr/lib/wsl/lib` `nvidia-smi`; `r2` then correctly refused a
  non-idle 18% GPU sample. Both directories remain failure evidence. `r3` fixed
  PATH/runtime idempotency and passed the original idle threshold without
  relaxing it.

### Authoritative data manifest

- Audited all eight used scenes against
  `H:\WorkSpace\3DGS-GPT56\datasets` without modifying that tree. Every source
  file and byte digest matched its WSL training mirror; the scene count is
  exactly eight.
- Accepted manifest digest:
  `7e10ff3ad3d708dba029ac64a307532f7bc9eca7322aa11df3fc571313041062`.
  Machine-readable JSON SHA-256 is
  `40caa1de988f9e806327367577fbcf2a390c3ef750891d1422358be0d0fb9ab9`;
  tabular TSV SHA-256 is
  `7ed5fa24d56ca415af4d218f946f90d3837f822dc4f76e29f484a0c7bb3817b`.
- Recomputed the complete eight-scene manifest after all experiments into
  `used-data-final-audit-20260803-v1`. Its JSON and TSV are byte-identical to
  the pre-run files and reproduce the same digest, proving that the
  authoritative data and WSL inputs did not drift during the experiment.

### ShorterSplatting contemporaneous baseline

- Freeze ID `20260803-shorter-splatting-s0-v1` uses PyTorch 2.0.0+cu118,
  CUDA 11.8, RTX 3080, and a dedicated conda prefix. Import/build probes and a
  3k densification smoke passed (926,592 points).
- The upstream truck training completed at 30k, but its bundled
  `example_metrics.py` exhausted GPU memory because TorchMetrics state was
  accumulated across the full training set and called twice. The failure is
  retained. A test-only, float32, memory-safe evaluation overlay was separately
  sealed (`execution-eval-v2.tar` SHA-256
  `8266907b4ec3bbd7634931961dd220a9b0b6038574914620083650460914e097`);
  pre/post PLY hashes prove that recovery did not alter the model.
- Unified seed-0 results (truck/train/playroom/drjohnson) are respectively:
  PSNR 25.2251/21.5239/29.9136/29.1304, SSIM
  0.87864/0.81261/0.90076/0.89431, LPIPS
  0.14244/0.19973/0.24246/0.24796, point counts
  2,584,064/1,085,440/2,326,016/3,273,600, and full training invocation wall
  310.53/283.38/250.39/264.64 seconds.
- Native-renderer synchronized 30-pass FPS is
  199.25/207.70/249.80/186.76 for the same scene order. These cross-rasterizer
  numbers are descriptive and are not treated as paired FastGS inference.

### Frozen v4 mechanisms

- Completed all 34 preregistered strict collectors and 12 shared-Scene,
  interleaved 30-pass FPS jobs. Machine-readable and paper-table evidence is
  `FORMAL_MECHANISMS_V4.json` / `FORMAL_MECHANISMS_V4.md`.
- Count-matched opacity versus joint is scene-dependent: opacity is +0.2387 dB
  on truck, while joint is +0.2693 dB on playroom. Score-only and radii-only
  priorities collapse PSNR by 5.4--8.8 dB at the identical point count;
  random pruning also loses 0.565 dB on truck. This supports the use of opacity
  as a quality-bearing signal but does not justify a universal superiority
  claim over joint priority.
- Moving one-shot exact pruning from 27k to 30k consistently worsens truck PSNR
  by 1.700--1.826 dB over seeds 0--2 and playroom by 0.654 dB. The late recovery
  arm is therefore a negative result.
- The frozen Pareto rows cover ratios 1.00/0.90/0.82/0.70 on truck and
  playroom. The 0.82 row retains 25.8079 dB at 207,698 points on truck and
  30.1234 dB at 150,170 points on playroom; all targets are exact.
- Reversing load feedback lowers achieved point count on all four tested scenes
  but trades quality inconsistently. FPS changes range from +1.74% to +4.98%;
  the mechanism is reported as sensitivity evidence, not a default method.
- The training wrapper's first checksum manifest accidentally included itself;
  exactly that entry failed while all seven substantive files passed. The
  original is retained, an explicit audit note was added, and the corrected
  non-self-referential `evidence_v2.sha256` verifies all seven files. The
  launcher is patched so subsequent manifests cannot self-reference.

### Formal unified third-party comparison

- Freeze ID `20260803-thirdparty-five-v1` completed all 20 seed-0
  train/render/unified-evaluation runs and all 20 idle-GPU native-renderer FPS
  plus strict-collector runs: Vanilla 3DGS, Speedy-Splat, count-matched
  Taming-3DGS, Mini-Splatting, and DashGaussian on truck, train, playroom, and
  drjohnson. No formal training or collector failed.
- Taming PLY counts were within the frozen 0.1%/10-point tolerance on every
  scene: 207,693/207,698 (truck), 189,335/189,337 (train),
  150,163/150,170 (playroom), and 207,989/207,997 (drjohnson).
- Together with FastGS, HAB, and ShorterSplatting, the unified table contains
  exactly 32 rows and one evaluator SHA-256. Output hashes are:
  JSON `862af9875af245a5de99b5d1d4a65e1515a52692a45b3554a9b81e6af63f602a`,
  CSV `7a40754360a7e8ac87b78ff963aba7899e2561469b1b1521b8417e9faa409b03`,
  Markdown `d7070309702f719604bf71ab05e161e6a3e7a34928e42fafbf584f4e2820db85`.
- Against count-matched Taming, HAB improves PSNR on all four scenes by
  +0.102 to +0.961 dB, lowers LPIPS by 0.0109 to 0.0345, and raises descriptive
  native-renderer FPS by +65.6% to +103.9%. The preregistered broad gate still
  fails because its fourth condition requires training wall <=0.5x; observed
  ratios are 0.542--0.742. The admissible claim is same-count quality/FPS
  superiority over this Taming configuration, not >=2x training speed.
- Mini has the highest geometric-mean descriptive FPS (482.0), while Dash has
  the highest four-scene mean PSNR (27.146). Both use substantially more points
  than HAB on most scenes; cross-rasterizer FPS is not paired inference.
- During truck/vanilla, one read-only full freeze-manifest hash ran for about
  88 seconds and was terminated by exact PID to avoid host I/O contention. It
  did not start CUDA or write source/data/model files. The event is sealed in
  `THIRDPARTY_TRAIN_BACKGROUND_AUDIT.txt`; truck/vanilla wall remains
  descriptive and is not used for a paired wall-time claim.
- Wallpaper Engine was paused through the official control interface for clean
  mechanism and third-party FPS windows, then resumed after all GPU timing
  completed.
