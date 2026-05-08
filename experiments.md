# Experiment Log

| ID | Date | Model | Features | CV | Local Score | Public LB | Notes |
|---|---|---|---|---|---:|---:|---|
| 000 | 2026-05-04 | CatBoost GPU | prefix + interactions + action stacking | single split by match | 0.4187 | - | README current best |
| 001 | TBD | CatBoost GPU | same as 000 | 5-fold GroupKFold by match | TBD | - | stability check |
| 002 | TBD | CatBoost GPU | + transition probability features | 5-fold GroupKFold by match | TBD | - | improve actionId/pointId |
| 003 | TBD | CatBoost seed ensemble | best tabular features | 5-fold GroupKFold by match | TBD | - | reduce variance |
| 004 | TBD | CatBoost GPU | prefix + interactions + action stacking | single split by match | TBD | reset | refreshed `test_new.csv` submission |

## Naming

- Metrics: `outputs/metrics_<experiment_id>_<short_name>.json`
- OOF predictions: `outputs/oof_<experiment_id>_<short_name>`
- Submissions: `outputs/submission_<experiment_id>_<short_name>.csv`
- Error analysis: `outputs/error_analysis_<experiment_id>_<short_name>/`

Use `test_new` in short names for submissions generated from the refreshed 2026-05-06
test data, for example `outputs/submission_004_catboost_gpu_test_new.csv`.

## Priorities

1. Validate current best with 5-fold GroupKFold.
2. Use OOF error analysis to identify weak `actionId` and `pointId` classes.
3. Add fold-safe transition probability features.
4. Compare CatBoost, LightGBM, and LSTM under the same validation protocol.
5. Build seed/model ensembles after validation is stable.
6. Regenerate all leaderboard submissions from `test_new.csv`; older `test.csv`
   outputs no longer count after the leaderboard reset.
