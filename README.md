# AI CUP 2026 Spring

Project workspace for the AI CUP 2026 Spring table-tennis rally prediction task.

## Environment

This repo uses Ryo's Miniconda with a project-local conda environment:

```bash
source /mnt/8tb_hdd/ryo/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/8tb_hdd/ryo/ai-cup-2026spring/conda-envs/ai-cup-ryo
```

Install the package after dependencies are ready:

```bash
python -m pip install -e .
```

## Baseline

Run the LSTM baseline:

```bash
aicup-lstm-baseline \
  --train train.csv \
  --test test.csv \
  --out outputs/submission_lstm_baseline.csv
```

Run the sklearn tabular experiment:

```bash
aicup-tabular-baseline \
  --train train.csv \
  --test test.csv \
  --out outputs/submission_tabular_baseline.csv
```

Run the current-best CatBoost tabular experiment on both local GPUs:

```bash
aicup-tabular-baseline \
  --model catboost \
  --device gpu \
  --gpu-devices 0:1 \
  --n-estimators 600 \
  --learning-rate 0.03 \
  --cat-depth 8 \
  --train train.csv \
  --test test.csv \
  --out outputs/submission_catboost_gpu_v5.csv \
  --metrics-out outputs/metrics_catboost_gpu_v5.json \
  --val-predictions-out outputs/val_catboost_gpu_v5
```

Current local validation best:

- Model: CatBoost GPU with stacked point prediction and shot interaction features.
- Settings: `--n-estimators 600 --learning-rate 0.03 --cat-depth 8`.
- Score: `0.4187` (`f1_action=0.3341`, `f1_point=0.2129`, `auc=0.9995`).

Audit validation signals and simple leakage risks:

```bash
aicup-audit-validation \
  --train train.csv \
  --test test.csv \
  --out outputs/audit_validation.json
```
