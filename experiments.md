# Experiment Log

| ID | Date | Model | Features | CV | Local Score | Public LB | Notes |
|---|---|---|---|---|---:|---:|---|
| 000 | 2026-05-04 | CatBoost GPU | prefix + interactions + action stacking | single split by match | 0.4187 | - | README current best |
| 001 | TBD | CatBoost GPU | same as 000 | 5-fold GroupKFold by match | TBD | - | stability check |
| 002 | TBD | CatBoost GPU | + transition probability features | 5-fold GroupKFold by match | TBD | - | improve actionId/pointId |
| 003 | TBD | CatBoost seed ensemble | best tabular features | 5-fold GroupKFold by match | TBD | - | reduce variance |
| 004 | 2026-05-08 | CatBoost GPU | prefix + interactions + action stacking | single split by match | 0.4143 | reset | refreshed `test_new.csv`; current best local run |
| 005 | 2026-05-08 | CatBoost GPU | 004 + fold-safe transition priors | single split by match | 0.3043 | - | negative result; action/point F1 collapsed |
| 006 | 2026-05-08 | CatBoost CPU | 004 without action/point class weights | single split by match | 0.4037 | - | below 004; GPU attempt OOMed |
| 007 | 2026-05-15 | Transformer (raw seq + FocalLoss) | raw stroke features + TransformerEncoder (3L/4H) | single GroupShuffleSplit by match | 0.2369 | - | First Transformer attempt; F1_action/point collapsed (no prefix engineering); baseline for future hybrid |
| 008 | 2026-05-15 | CatBoost GPU 5-fold | prefix + interactions + action stacking (best params) | 5-fold GroupKFold by match | 0.4176 ± 0.0050 (OOF 0.4087) | - | Most reliable CV for current SOTA tabular features; single-split 0.4143 was slightly optimistic |

## Naming

- Metrics: `outputs/metrics_<experiment_id>_<short_name>.json`
- OOF predictions: `outputs/oof_<experiment_id>_<short_name>`
- Submissions: `outputs/submission_<experiment_id>_<short_name>.csv`
- Models: `outputs/models_<experiment_id>_<short_name>/`
- Error analysis: `outputs/error_analysis_<experiment_id>_<short_name>/`

Use `test_new` in short names for submissions generated from the refreshed 2026-05-06
test data, for example `outputs/submission_004_catboost_gpu_test_new.csv`.

## Current Comparison

| Run | Metrics | Submission | Models |
|---|---|---|---|
| 004 | `outputs/metrics_catboost_gpu_test_new_v1.json` | `outputs/submission_catboost_gpu_test_new_v1.csv` | `outputs/models_catboost_gpu_test_new_v1/` |
| 005 | `outputs/metrics_catboost_gpu_test_new_v2_transition.json` | `outputs/submission_catboost_gpu_test_new_v2_transition.csv` | `outputs/models_catboost_gpu_test_new_v2_transition/` |
| 006 | `outputs/metrics_catboost_cpu_test_new_v3_no_class_weights.json` | `outputs/submission_catboost_cpu_test_new_v3_no_class_weights.csv` | `outputs/models_catboost_cpu_test_new_v3_no_class_weights/` |

Run 004 remains the recommended submission candidate. Run 005 shows that the first
transition-prior implementation overfits next-shot targets badly. Run 006 indicates that
balanced action/point class weights are still useful under the current feature set.

## Priorities

1. Validate current best with 5-fold GroupKFold.
2. Use OOF error analysis to identify weak `actionId` and `pointId` classes.
3. Rework transition probability features before reusing them; the 005 version is a negative result.
4. Compare CatBoost, LightGBM, and LSTM under the same validation protocol.
5. Build seed/model ensembles after validation is stable.
6. Regenerate all leaderboard submissions from `test_new.csv`; older `test.csv`
   outputs no longer count after the leaderboard reset.

## 2026-05-15 Night Autonomous Run (until ~7 AM)

- Hybrid RNN pipeline fix validated: 1-epoch small test passed (score 0.4037) after adding `total_length=x.size(1)` in pad_packed_sequence. Committed.
- Launched medium hybrid (5 epochs, hidden 256, 2 layers) as next step.
- Error analysis on CatBoost 5-fold OOF completed: action accuracy ~0.442, point ~0.271. Zero F1 action classes: 15 (traditional serve), 16 (hook serve). Weak serve-related actions identified for future focus.
- LightGBM 5-fold still running (very slow on GPU).

Next: Ensemble script, more hybrid scaling, weak-class targeted experiments.

## Night Run Update (2026-05-15 ~19:00+)

- Hybrid RNN medium (5 epochs, hidden 256, 2L): completed, best 0.4004 (epoch 2), but F1_action collapsed to ~0.001 after epoch 1 (heavy bias to actionId=8). Pipeline now stable (no shape crash), but learning issue for rare actions remains. Not competitive.
- LightGBM 5-fold: killed after ~81 min (still stuck in Fold 0 action training, too slow on this GPU setup).
- New CatBoost depth10/800trees/lr0.025 variant launched (single split).
- Ensemble script ready, blended CatBoost+Hybrid submission created (using CatBoost for action/point + hybrid for serverGetPoint).
- Error analysis done: action classes 15 (traditional serve) and 16 (hook serve) have zero F1 in 5-fold OOF.

Next autonomous: Improve hybrid (higher action loss weight, stronger rare class handling), re-test 1-epoch, more CatBoost variants, targeted modeling for weak serve classes.

- Improved hybrid 1-epoch test (action loss 0.55, focal 2.5): completed 0.3313 (worse than previous 0.4037), F1a still ~0.01, heavy bias persists. Weight change did not help rare action learning. Will de-prioritize heavy hybrid fixes for now; use as rally specialist for blending only.


- CatBoost depth10/800trees/lr0.025 (single split): completed, score 0.4062 (F1a=0.3177, F1p=0.1982). Slightly below v1 0.4143.
- CatBoost seed123 (same best params): completed, score 0.3911 (lower). Diversity for ensemble is moderate.


- CatBoost seed456 (same best params): completed, score **0.4163** (F1a=0.3395, F1p=0.2016). One of the strongest single-split runs so far.

