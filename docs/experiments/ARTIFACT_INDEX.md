# HAB-FastGS artifact index

## Submission-facing records

| File | Purpose |
|---|---|
| `SUBMISSION_EXPERIMENT_REPORT.md` | Standalone result and claim-disposition report |
| `EXPERIMENT_PROTOCOL.md` | Preregistered selection, confirmation, mechanisms, metrics, and gates |
| `WORKLOG.md` | Chronological environment, pilot, freeze, formal, and recovery ledger |
| `FAILURE_LEDGER.md` | Explicit failure/recovery inclusion and exclusion decisions |
| `TARGET_LEDGER_V4.json/.csv/.md` | Frozen per-scene target derivation |
| `DEV_SELECTION_V2.json/.md` | Development priority and schedule decision |
| `FORMAL_MAIN_V4_STATS.json` | Authoritative main statistics, CIs, gates, identities, hashes |
| `FORMAL_MAIN_V4_PAIRS.csv` | Same-seed main pair rows |
| `FORMAL_MAIN_V4_SCENES.csv` | Scene summaries |
| `FORMAL_MAIN_V4.md` | Paper-readable main table |
| `FORMAL_MECHANISMS_V4.json/.md` | Ablation, recovery, Pareto, load-feedback evidence |
| `ABLATION_V4.csv` | Count-matched priority/placement/random controls |
| `RECOVERY_V4.csv` | 27k versus 30k one-shot recovery |
| `PARETO_V4.csv` | Frozen count-quality-speed Pareto rows |
| `LOAD_FEEDBACK_V4.csv` | Forward/reversed load-feedback sensitivity |
| `FORMAL_THIRDPARTY_COMPARISON.json/.csv/.md` | Unified 4-scene x 8-method comparison |
| `ARTIFACT_SHA256SUMS.txt` | Final compact-record SHA-256 manifest |

## Provenance and freeze indexes

| Path | Purpose |
|---|---|
| `paper_artifacts/provenance/used-data-20260803-v1/` | Eight-scene source/mirror byte manifest |
| `paper_artifacts/provenance/used-data-final-audit-20260803-v1/` | Post-run recomputation, byte-identical to the pre-run manifest |
| `paper_artifacts/freeze/20260803-opacity-per-event-82-v4/` | Accepted FastGS/HAB source/env/runtime freeze index |
| `paper_artifacts/freeze/20260803-thirdparty-five-v1/` | Five-method source/env locks and source archives |
| `paper_artifacts/freeze/20260803-shorter-splatting-s0-v1/` | Shorter freeze and sealed evaluation overlay |
| `paper_artifacts/formal/` | Audited formal launchers and recovery notes |

## Raw WSL evidence roots

| Root | Contents |
|---|---|
| `/home/cute_cat/hab-paper-freeze/20260803-opacity-per-event-82-v4` | Read-only accepted HAB source and execution layer |
| `/home/cute_cat/hab-paper-records/20260803-opacity-per-event-82-v4` | FastGS/HAB collectors, phases, paired FPS |
| `/home/cute_cat/hab-paper-runs/20260803-opacity-per-event-82-v4` | FastGS/HAB model outputs and traces |
| `/home/cute_cat/hab-thirdparty-freeze/20260803-thirdparty-five-v1` | Five-method execution freeze |
| `/home/cute_cat/hab-thirdparty-records/20260803-thirdparty-five-v1` | Five-method strict collectors and phase evidence |
| `/home/cute_cat/hab-thirdparty-runs/20260803-thirdparty-five-v1` | Five-method models, renders, metrics, native FPS |
| `/home/cute_cat/hab-thirdparty-records/20260803-shorter-splatting-s0-v1` | Shorter strict collectors and phases |
| `/home/cute_cat/hab-thirdparty-runs/20260803-shorter-splatting-s0-v1` | Shorter models, recovery evaluation, native FPS |

The authoritative input tree is
`H:\WorkSpace\3DGS-GPT56\datasets`; it is not an output root and was never
modified in place.
