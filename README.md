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

For local checks and CI-equivalent commands:

```bash
make install-dev
make check
```

## Docker

Build a CPU image that includes the package and CLI entry points:

```bash
docker build -t ai-cup-ryo:local .
```

Run the default help command:

```bash
docker run --rm ai-cup-ryo:local
```

Build a GPU image for local CUDA training hosts such as this 2x4090 machine:

```bash
docker build -f Dockerfile.gpu -t ai-cup-ryo:gpu-local .
```

Run the GPU image with NVIDIA runtime:

```bash
docker run --rm --gpus all ai-cup-ryo:gpu-local
```

Mount local competition data when you want to run an experiment from the container:

```bash
docker run --rm \
  -v "$PWD:/app" \
  ai-cup-ryo:local \
  --train train.csv \
  --test test.csv \
  --out outputs/submission_tabular_baseline.csv
```

`Dockerfile` is the portable CPU image used by CI. `Dockerfile.gpu` is the CUDA image for local GPU training and GPU-capable deployment hosts.

## CI/CD

GitHub Actions now covers:

- `CI`: install, `ruff`, `py_compile`, CLI `--help` smoke checks, and Docker build validation.
- `Docker Publish`: manually build and publish both CPU and GPU images to `ghcr.io/<owner>/<repo>` via workflow dispatch.

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

Analyze OOF prediction errors after running with `--save-oof`:

```bash
aicup-error-analysis \
  --next-shot-oof outputs/oof_catboost_5fold_v1_next_shot.csv \
  --rally-oof outputs/oof_catboost_5fold_v1_rally.csv \
  --out-dir outputs/error_analysis_catboost_5fold_v1
```
