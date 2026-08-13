# HAB-FastGS development selection report

Generated: `2026-08-02T23:45:42.532687+00:00`

All 18 selection runs completed at the preregistered exact counts: truck `207698` and playroom `150170`.

## Arm means

| Scene | Arm | PSNR | SSIM | LPIPS | train wall (s) |
|---|---|---:|---:|---:|---:|
| truck | fastgs_baseline | 25.7984 | 0.876859 | 0.177056 | 276.37 |
| truck | dev_joint_v2 | 25.5347 | 0.872338 | 0.186766 | 229.38 |
| truck | dev_opacity_v2 | 25.7479 | 0.873890 | 0.185620 | 228.91 |
| truck | dev_opacity_ramp_v2 | 25.7487 | 0.873895 | 0.185801 | 233.84 |
| playroom | fastgs_baseline | 30.6345 | 0.910060 | 0.262882 | 258.80 |
| playroom | dev_joint_v2 | 30.4152 | 0.909287 | 0.267179 | 219.13 |
| playroom | dev_opacity_v2 | 30.4307 | 0.907866 | 0.265693 | 229.94 |
| playroom | dev_opacity_ramp_v2 | 30.2060 | 0.908281 | 0.266619 | 226.17 |

## Priority decision

Joint is rejected; `opacity_only/per_event` is the frozen quality profile.

| Gate | Observed | Pass |
|---|---:|:---:|
| Aggregate paired FPS (joint - opacity) | 3.347% CI [1.337, 5.128] | False |
| Truck PSNR (joint - opacity) | -0.2131 dB | False |
| Playroom PSNR (joint - opacity) | -0.0155 dB | False |
| Worst seed PSNR delta | -0.2883 dB | False |
| LPIPS / SSIM tolerances | max LPIPS 0.00149, min SSIM -0.00155 | True |
| Identical exact counts/settings | 6/6 pairs | True |

## Schedule decision

Ramp is rejected; `per_event` is the frozen schedule.

| Gate | Observed | Pass |
|---|---:|:---:|
| Aggregate PSNR (ramp - per-event) | -0.1119 dB CI [-0.3485, 0.0296] | False |
| Scene PSNR means | truck 0.0008, playroom -0.2247 dB | False |
| Paired FPS (ramp - per-event) | 0.131% CI [-2.010, 2.048] | False |
| Aggregate wall-time change | 0.248% | True |
| LPIPS / SSIM tolerances | max LPIPS 0.00093, min SSIM 0.00000 | True |
| Exact counts | 6/6 pairs | True |

## Dev-only baseline context

The selected opacity profile changes aggregate PSNR by -0.1272 dB (95% CI [-0.2525, -0.0339]) versus same-seed FastGS and changes training wall time by -14.011%. These are selection-scene diagnostics, not confirmation evidence.

FPS CIs resample scenes, seeds, and the 30 paired interleaved repeats. Quality CIs resample scenes and seeds only; test views are not treated as independent replicates.
