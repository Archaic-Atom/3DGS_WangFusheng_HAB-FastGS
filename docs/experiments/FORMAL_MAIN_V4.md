# Frozen v4 FastGS vs HAB main results

All quality values use the frozen FastGS evaluator and exact render/GT stem matching. FPS inference uses 30 paired, synchronized, interleaved passes after three warmups; per-run screening FPS is excluded.

The fixed HAB target is `floor(0.82 * FastGS seed-0 N)` for each scene and is reused across seeds. Therefore the target is exactly preregistered, while its percentage relative to seed-1/2 FastGS can differ from 18%.

## Absolute results

| scene | arm | seeds | Gaussians mean +/- sd | PSNR | SSIM | LPIPS | train seconds |
|---|---|---|---|---|---|---|---|
| train | fastgs | 0,1,2 | 232214 +/- 3553 | 22.486 +/- 0.123 | 0.8077 +/- 0.0019 | 0.2397 +/- 0.0020 | 264.0 +/- 5.0 |
| train | hab_main | 0,1,2 | 189337 +/- 0 | 22.353 +/- 0.083 | 0.8039 +/- 0.0007 | 0.2473 +/- 0.0005 | 246.2 +/- 12.4 |
| drjohnson | fastgs | 0,1,2 | 256160 +/- 3825 | 29.452 +/- 0.022 | 0.9008 +/- 0.0008 | 0.2712 +/- 0.0008 | 288.1 +/- 7.8 |
| drjohnson | hab_main | 0,1,2 | 207997 +/- 0 | 29.419 +/- 0.093 | 0.9001 +/- 0.0004 | 0.2739 +/- 0.0008 | 269.2 +/- 1.2 |
| garden | fastgs | 0,1,2 | 749611 +/- 18962 | 27.136 +/- 0.106 | 0.8363 +/- 0.0006 | 0.1740 +/- 0.0006 | 533.7 +/- 2.3 |
| garden | hab_main | 0,1,2 | 606922 +/- 0 | 27.148 +/- 0.022 | 0.8313 +/- 0.0007 | 0.1875 +/- 0.0018 | 495.3 +/- 11.9 |
| room | fastgs | 0,1,2 | 208245 +/- 1443 | 31.924 +/- 0.049 | 0.9207 +/- 0.0005 | 0.2162 +/- 0.0004 | 356.8 +/- 3.5 |
| room | hab_main | 0,1,2 | 170528 +/- 0 | 31.827 +/- 0.046 | 0.9181 +/- 0.0009 | 0.2225 +/- 0.0008 | 343.0 +/- 3.5 |
| bicycle | fastgs | 0 | 538989 +/- 0 | 24.832 +/- 0.000 | 0.7140 +/- 0.0000 | 0.3098 +/- 0.0000 | 432.9 +/- 0.0 |
| bicycle | hab_main | 0 | 441970 +/- 0 | 24.822 +/- 0.000 | 0.7124 +/- 0.0000 | 0.3167 +/- 0.0000 | 407.4 +/- 0.0 |
| counter | fastgs | 0 | 208314 +/- 0 | 29.168 +/- 0.000 | 0.9079 +/- 0.0000 | 0.2030 +/- 0.0000 | 397.4 +/- 0.0 |
| counter | hab_main | 0 | 170817 +/- 0 | 29.104 +/- 0.000 | 0.9055 +/- 0.0000 | 0.2087 +/- 0.0000 | 361.0 +/- 0.0 |

## Same-seed paired deltas (HAB - FastGS)

| scene | role | n seeds | PSNR | SSIM | LPIPS | train wall | paired FPS |
|---|---|---|---|---|---|---|---|
| train | confirmation | 3 | -0.1326 | -0.00379 | +0.00763 | -6.7% | +5.1% |
| drjohnson | confirmation | 3 | -0.0331 | -0.00071 | +0.00265 | -6.5% | +4.3% |
| garden | confirmation | 3 | +0.0122 | -0.00492 | +0.01350 | -7.2% | +8.4% |
| room | confirmation | 3 | -0.0969 | -0.00254 | +0.00621 | -3.9% | +5.3% |
| bicycle | extension | 1 | -0.0102 | -0.00162 | +0.00688 | -5.9% | - |
| counter | extension | 1 | -0.0641 | -0.00243 | +0.00569 | -9.2% | - |

## Confirmation hierarchical inference

- PSNR: -0.0626 dB, 95% CI [-0.1304, +0.0090]
- SSIM: -0.00299, 95% CI [-0.00455, -0.00139]
- LPIPS: +0.00750, 95% CI [+0.00383, +0.01168]
- Train wall: -6.08%, 95% CI [-8.15, -4.08]
- Paired FPS: +5.76%, 95% CI [+4.15, +7.63]

## Preregistered gates

| gate | status |
|---|---|
| exact_fixed_target_from_seed0_0.82 | PASS |
| psnr_ci_lower_above_-0.30 | PASS |
| ssim_ci_lower_above_-0.005 | PASS |
| lpips_ci_upper_below_+0.01 | FAIL |
| paired_fps_ci_lower_above_+5pct | FAIL |
| wall_ci_upper_at_most_-10pct | FAIL |
| no_confirmation_scene_slower_on_average | PASS |
| no_catastrophic_confirmation_scene | PASS |

Claim disposition: **limited-domain resource-quality trade-off; broad non-inferiority claim not supported**.

Catastrophic-scene flags: none.
