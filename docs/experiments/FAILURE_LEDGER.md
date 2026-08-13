# HAB-FastGS formal failure and recovery ledger

This ledger is part of the submission record. Failures are retained rather
than overwritten. A recovery is accepted only when it is validation-only or
its new run ID cleanly separates it from the failed attempt.

| Scope | Failure or irregularity | Disposition | Accepted-result impact |
|---|---|---|---|
| M360 bicycle pilot | Launcher was edited while its shell remained alive; Bash later hit a parse error after Python had saved the exact PLY. | Dedicated no-training recovery rendered/evaluated the unchanged PLY; pre-recovery hash retained. | Pilot only; no formal row uses the recovered wall as ordinary instrumented time. |
| Development FPS startup | Two startup attempts exposed import-root and optional-argument normalization defects before timing. | New immutable ID `dev-selection-v2-20260803T0736CST`; failed IDs retained. | No failed timing enters inference. |
| Paper freeze v1 | Absolute dependencies on mutable base environment. | Rejected. | Produces no accepted row. |
| Paper freeze v2 | Exact unpublished `lit==15.0.7` wheel could not be archived. | Rejected. | Produces no accepted row. |
| Paper freeze v3 | Post-link `patchelf` mutation caused `simple-knn.distCUDA2` to raise false CUDA OOM. | Rejected after controlled loader diagnostic. | Produces no accepted row. |
| Confirmation wrapper | Outer 7,200 s cell ended after the final strict collector was written but before final echo/checksum. | Validation-only terminal recovery wrote `wrapper_recovery.json`; PLY/metrics/FPS/collector unchanged. | All 28 collectors accepted; recovery is explicit provenance. |
| Confirmation paired FPS r1 | PATH omitted WSL `/usr/lib/wsl/lib`, so `nvidia-smi` was unavailable before timing. | Failed directory retained. | No measurement accepted. |
| Confirmation paired FPS r2 | Idle guard observed 18% GPU utilization and correctly refused. | Failed directory retained; threshold not relaxed. | No measurement accepted. |
| Confirmation paired FPS r3 | Clean idle run with corrected PATH/runtime idempotency. | Accepted ID `formal-paired-fps-v4-confirmation-s012-r3`. | Sole confirmation FPS inference source. |
| ShorterSplatting upstream evaluation | `example_metrics.py` accumulated TorchMetrics state over the training set and called metrics twice, causing CUDA OOM after a successful truck 30k training. | Test-only float32 memory-safe overlay `execution-eval-v2` was externally sealed; PLY hash before/after recovery is identical. | All four Shorter rows accepted from the unchanged 30k PLYs; upstream failure retained. |
| Shorter count smoke | An early smoke interpreted the wrong tensor axis as Gaussian count. | Excluded; collector and full-Ply smoke corrected to vertex/count axis. | No incorrect-count row accepted. |
| Mechanism evidence manifest | Original `evidence.sha256` included itself and exactly that entry failed; seven substantive files passed. | Original retained; audit note plus non-self-referential `evidence_v2.sha256` verify all seven files; launcher patched. | 34 mechanism collectors and 12 FPS jobs remain accepted. |
| Third-party truck/vanilla wall | A read-only full freeze-manifest hash overlapped training for about 88 s before exact-PID termination. No CUDA or writes occurred. | `THIRDPARTY_TRAIN_BACKGROUND_AUDIT.txt` is included in `evidence_v2.sha256`. | Quality/count/FPS accepted; truck/vanilla wall is descriptive and not exact paired inference. |

No formal third-party training, render, unified evaluation, native-FPS job, or
strict collector failed. The five-method formal matrix is 20/20 complete.
