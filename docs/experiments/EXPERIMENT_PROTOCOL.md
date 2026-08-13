# HAB-FastGS submission experiment protocol

Status: development profile and immutable paper runtime frozen before the final
confirmation matrix on 2026-08-03. This document separates development
decisions from confirmation evidence and prevents post-hoc tuning on the
confirmation scenes.

## Source record

The protocol is derived from the HAB-FastGS Notion overview and the WSL rerun
record:

- https://app.notion.com/p/3aa4b816e53b819fbc68d1c9343e27b2
- https://app.notion.com/p/3af4b816e53b81939ce3cec74e9cef32

The authoritative execution tree is `/home/cute_cat/hab-fastgs` in the
`Ubuntu-22.04` WSL distribution. The authoritative input-data tree is
`H:\WorkSpace\3DGS-GPT56\datasets`; launchers must never train from or write to
that DrvFS tree. WSL ext4 copies or explicitly derived overlays are required.

## Scene roles

- Development/selection: `truck`, `playroom`.
- Confirmation: `train`, `drjohnson`, `garden`, `room`.
- Fixed-seed extension: `bicycle`, `counter`.

The development scenes have already been inspected repeatedly and cannot be
described as independent confirmation evidence. After the method profile is
frozen, no priority weight, schedule, target ratio, or threshold may be changed
using confirmation-scene results. A failed confirmation narrows the claim; it
does not authorize tuning on the failed scene.

## Pilot-result disposition

The 27k final-prune score/mask alignment defect was fixed on 2026-08-03. Old
`hab_fixed`, `cb_score_only`, `cb_post_densify`, `cb_final_only`, and the 27k vs
30k placement comparison are pilot-only because their exact-budget ranking can
depend on the pre-fix index order. FastGS baselines are unaffected. Load-aware
arms without exact-count enforcement are not affected by this defect, but their
quality must still be recomputed by the frozen evaluator and their FPS remeasured.

No legacy RTX 2080 Ti result or native-repository metric may be merged into the
RTX 3080 submission table.

## Development decision 1: priority profile

Run `joint/per_event` and `opacity_only/per_event` on truck and playroom with
seeds 0, 1, and 2. For a scene, every arm must have the exact same target count,
defined as `floor(0.82 * FastGS seed-0 count)`.

`joint` can be the sole default only if all conditions hold:

1. Interleaved paired FPS improvement over opacity-only has an aggregate 95%
   confidence lower bound of at least +5%, and each scene CI excludes a loss.
2. Three-seed mean PSNR loss versus opacity-only is at most 0.10 dB per scene.
3. LPIPS increases by at most 0.005 and SSIM decreases by at most 0.002.
4. No seed loses more than 0.20 dB PSNR.
5. Point counts and all non-priority settings are identical.

Otherwise opacity-only is the default quality profile. A statistically faster
joint arm below the +5% gate may be reported only as a speed-first profile. If
there is no significant paired-FPS gain, the joint formula is a falsified
ablation, not a default-method success.

## Development decision 2: budget schedule

After selecting the priority, compare its current per-event schedule against
one predeclared linear ramp on truck and playroom, seeds 0, 1, and 2. There is no
schedule grid search. Ramp is accepted only if:

1. PSNR is non-negative on both scenes and the paired aggregate improvement is
   at least 0.05 dB with a 95% confidence lower bound above zero.
2. LPIPS worsens by no more than 0.002 and SSIM by no more than 0.001.
3. Paired FPS loss has a 95% confidence lower bound no worse than -2%.
4. Training wall time increases by no more than 5%.
5. Every run reaches the exact target count.

Failure retains per-event as the final schedule. Ramp is then reported once as
a negative or statistically null ablation and is not retuned.

## Frozen development outcome

All 18 required development runs completed at the exact fixed scene counts.
The accepted paired-FPS evidence is run ID
`dev-selection-v2-20260803T0736CST`, with 30 interleaved measured passes per
scene and seed after warm-up.

- Priority: `opacity_only`. Joint achieved +3.347% aggregate paired FPS with a
  hierarchical 95% CI of [1.337%, 5.128%], so its lower bound did not reach
  +5%. Joint also lost 0.2131 dB mean PSNR on truck, and its worst seed lost
  0.2883 dB. The FPS, per-scene PSNR, and worst-seed gates failed.
- Schedule: `per_event`. The fixed ramp changed aggregate PSNR by -0.1119 dB
  with a 95% CI of [-0.3485, 0.0296] and paired FPS by +0.131% with a 95% CI
  of [-2.010%, 2.048%]. Its PSNR and FPS gates failed.
- Ratio: fixed `0.82`, with scene targets defined from the FastGS seed-0 count
  exactly as preregistered.

No confirmation-scene result may change this profile. The machine-readable
decision evidence is `paper_artifacts/records/DEV_SELECTION_V2.json`, and the
human-readable table is `paper_artifacts/records/DEV_SELECTION_V2.md`.

## Runtime authority

The only accepted FastGS/HAB paper runtime is freeze ID
`20260803-opacity-per-event-82-v4`:

- source: `/home/cute_cat/hab-paper-freeze/20260803-opacity-per-event-82-v4/source`;
- execution layer: `/home/cute_cat/hab-paper-freeze/20260803-opacity-per-event-82-v4/execution`;
- conda prefix:
  `/home/cute_cat/anaconda3/envs/habfastgs-paper-20260803-opacity-per-event-82-v4`;
- Windows evidence index:
  `paper_artifacts/freeze/20260803-opacity-per-event-82-v4`.

The freeze manifest digest is
`03de8357f7faa6d1634c4fd15b7f1df7d0ec6b762cd05825e3829fb2c7c203b6`.
The full manifest, source lock, linker-emitted relative RPATH, CUDA extension
probes, real 20-iteration training, 32-view render, strict unified evaluation,
and 30-pass synchronized-FPS smoke all passed. v1-v3 are failure evidence and
must never produce accepted rows.

## Final matrix

The accepted main method uses the fixed 0.82 ratio.

| Purpose | Scenes | Arms | Seeds |
|---|---|---|---|
| Priority selection | truck, playroom | joint, opacity-only | 0, 1, 2 |
| Schedule selection | truck, playroom | selected per-event, fixed ramp | 0, 1, 2 |
| Confirmation | train, drjohnson, garden, room | FastGS, frozen method | 0, 1, 2 |
| Extension | bicycle, counter | FastGS, frozen method | 0 |
| Count-matched ablation | truck, playroom | default, joint, opacity, score, radii, random, post-densify, final-only | 0 |
| Recovery-window ablation | truck (0,1,2), playroom (0) | exact cut at 27k, exact cut at 30k | listed |
| Budget Pareto | truck, playroom | ratios 0.70, 0.82, 0.90 and FastGS 1.00 | 0 |
| Load feedback | truck, train, playroom, drjohnson | forward, reversed | 0 |
| Notion comparison matrix | truck, train, playroom, drjohnson | vanilla, speedy, taming, mini, Dash | 0 |

The five-method comparison matrix must be complete. At least one additional
contemporaneous workload/training-efficiency baseline (ShorterSplatting or
Faster-GS) is required on all four original scenes before claiming broad
state-of-the-art coverage.

## Metrics and statistics

- Quality: the frozen FastGS PSNR/SSIM implementation plus VGG-LPIPS, computed
  from an exact render/GT stem match. Native metrics are compatibility evidence
  only.
- Seeds: exactly `{0,1,2}` where the matrix specifies three seeds. Seeds are not
  added or removed after inspecting significance.
- Quality summaries: scene mean +/- sample SD and same-seed paired deltas.
- Aggregate inference: hierarchical paired bootstrap, resampling scenes first
  and seeds within scene. Test views are not independent replicates.
- FPS: one process and one shared Scene, balanced/interleaved arm order, at least
  30 measured passes after warm-up, paired percentage differences and 95% CIs.
- Wall time: full 30k train invocation, with identical scheduled evaluation and
  saving boundaries for every method.

## Confirmation acceptance gates

Against FastGS on the four confirmation scenes, the frozen method must meet:

1. Exact 18% Gaussian-count reduction.
2. Aggregate paired PSNR 95% CI lower bound above -0.30 dB.
3. Aggregate paired SSIM lower bound above -0.005.
4. Aggregate paired LPIPS upper bound below +0.01.
5. Paired FPS improvement lower bound above +5%.
6. Training wall time improves by at least 10%, with no confirmation scene
   slower on average.
7. No scene loses more than 0.75 dB PSNR or worsens LPIPS by more than 0.02.

If a gate fails, the manuscript must narrow the claim (compression-only,
quality-resource trade-off, or limited scene domain) instead of changing the
method after confirmation.

## Frozen confirmation disposition

The v4 confirmation matrix is complete and the profile is not retuned. The
hierarchical paired estimates are PSNR -0.0626 dB (95% CI [-0.1304, +0.0090]),
SSIM -0.00299 ([-0.00455, -0.00139]), LPIPS +0.00750 ([+0.00383,
+0.01168]), training wall -6.08% ([-8.15%, -4.08%]), and paired FPS +5.76%
([+4.15%, +7.63%]).

The exact-count, PSNR, SSIM, no-slower-scene, and no-catastrophic-scene gates
pass. LPIPS upper-bound, paired-FPS lower-bound, and training-wall gates fail.
The admissible manuscript claim is therefore a fixed 18% Gaussian compression
and limited-domain quality-resource trade-off. It is not a broad
non-inferiority, guaranteed render acceleration, or >=10% training-speed claim.
The authoritative evidence is `paper_artifacts/records/FORMAL_MAIN_V4_STATS.json`.

## Mechanism-result disposition

The preregistered mechanism matrix is complete: 34 strict training/evaluation
collectors and 12 shared-Scene, interleaved synchronized-FPS jobs. The selected
opacity priority remains the frozen default; these experiments do not reopen
development selection.

- At the exact same count, opacity exceeds joint by 0.2387 dB on truck, while
  joint exceeds opacity by 0.2693 dB on playroom. Priority performance is
  scene-dependent, so no universal opacity-over-joint claim is admissible.
- Score-only and radii-only priorities lose 5.4--8.8 dB PSNR at fixed count.
  Random pruning loses 0.565 dB on truck. These are strong negative controls.
- Delaying one-shot exact pruning from 27k to 30k loses 1.700--1.826 dB on
  truck across all three seeds and 0.654 dB on playroom. Late one-shot recovery
  is rejected.
- Load-feedback reversal changes both achieved count and quality. It is
  sensitivity evidence only and is not used for the main exact-count claim.

The authoritative files are `FORMAL_MECHANISMS_V4.json`,
`FORMAL_MECHANISMS_V4.md`, `ABLATION_V4.csv`, `RECOVERY_V4.csv`,
`PARETO_V4.csv`, and `LOAD_FEEDBACK_V4.csv`.

## Contemporaneous baseline policy

ShorterSplatting and the five Notion-listed repositories are run from sealed
source manifests and dedicated conda prefixes. Every formal baseline uses seed
zero, 30k iterations, the original method's native renderer, and the exact
frozen FastGS PSNR/SSIM + VGG-LPIPS evaluator over strict test-view stem and
resolution equality. FPS is measured only after training, on an idle GPU, with
three warmup and 30 measured passes. Cross-rasterizer FPS is descriptive rather
than same-process paired inference.

The upstream ShorterSplatting metric helper's training-set TorchMetrics state
accumulation caused an OOM after a successful truck training run. Its accepted
recovery overlay is test-only, explicit float32, memory-safe, and sealed; PLY
hash equality proves that no trained model was changed. All four original
scenes have accepted Shorter collectors and synchronized native-renderer FPS.

## Third-party comparison disposition

The five-method matrix is complete with 20 accepted seed-0 collectors; adding
FastGS, HAB, and ShorterSplatting yields exactly 32 rows over the four original
scenes. All rows use the same frozen evaluator. Count-matched Taming differs
from the HAB target by no more than eight points on any scene.

HAB exceeds count-matched Taming on every scene in PSNR (+0.102 to +0.961 dB),
LPIPS (-0.0109 to -0.0345), and descriptive native-renderer FPS (+65.6% to
+103.9%). However, the preregistered broad gate also requires HAB training wall
to be at most half of Taming's; the observed ratios are 0.542--0.742, so the
broad 3-of-4 gate fails in all four scenes. The paper may claim same-count
quality and native-FPS superiority over the frozen Taming configuration, but
must not claim >=2x training speed or universal cross-rasterizer dominance.

The authoritative comparison files are `FORMAL_THIRDPARTY_COMPARISON.json`,
`FORMAL_THIRDPARTY_COMPARISON.csv`, and
`FORMAL_THIRDPARTY_COMPARISON.md`.

## Per-run acceptance

An accepted row requires the frozen source/environment/data-manifest hashes,
complete command and seed, start/end times, 30k PLY, exact expected test views,
exact render/GT stems, strict unified metrics with per-view values, full log and
HAB trace, and an idle-GPU paired-FPS record. Tracebacks, NaNs, OOMs, point-count
mismatches, score/mask assertion failures, missing FPS, or duplicate run keys
are hard failures and remain in the failure ledger.
