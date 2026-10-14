# AI CUP 2026 Spring

Project workspace for the AI CUP 2026 Spring table-tennis rally prediction task.

## Environment

Create an isolated Python 3.10 or 3.11 environment with the tool of your
choice. For example, with conda:

```bash
conda create -n ai-cup-2026spring python=3.11
conda activate ai-cup-2026spring
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

The competition released a refreshed test set on 2026-05-06. Put it at `test_new.csv`;
the CLIs now use `test_new.csv` by default when it exists, falling back to `test.csv`
only for older local runs. Treat `rally_uid` only as an identifier. Its numeric order
is randomized and must not be used as rally sequence information.

Run the LSTM baseline:

```bash
aicup-lstm-baseline \
  --train train.csv \
  --test test_new.csv \
  --out outputs/submission_lstm_test_new.csv
```

Run the sklearn tabular experiment:

```bash
aicup-tabular-baseline \
  --train train.csv \
  --test test_new.csv \
  --out outputs/submission_tabular_test_new.csv
```

Run the current-best CatBoost tabular experiment on both local GPUs:

```bash
aicup-tabular-baseline \
  --model catboost \
  --device gpu \
  --gpu-devices 1 \
  --n-estimators 600 \
  --learning-rate 0.03 \
  --cat-depth 8 \
  --train train.csv \
  --test test_new.csv \
  --out outputs/submission_catboost_gpu_test_new_v1.csv \
  --metrics-out outputs/metrics_catboost_gpu_test_new_v1.json \
  --val-predictions-out outputs/val_catboost_gpu_test_new_v1 \
  --models-out-dir outputs/models_catboost_gpu_test_new_v1
```

Current local validation best:

- Model: CatBoost GPU with stacked point prediction and shot interaction features.
- Settings: `--n-estimators 600 --learning-rate 0.03 --cat-depth 8`.
- Refreshed `test_new.csv` run: `outputs/submission_catboost_gpu_test_new_v1.csv`.
- Saved models: `outputs/models_catboost_gpu_test_new_v1/`.
- Score: `0.4143` (`f1_action=0.3311`, `f1_point=0.2050`, `auc=0.9995`).

Recent optimization comparison on the same single match-group split:

| Run | Change | Score | F1 action | F1 point | AUC | Submission |
|---|---|---:|---:|---:|---:|---|
| v1 | CatBoost GPU, balanced action/point weights | 0.4143 | 0.3311 | 0.2050 | 0.9995 | `outputs/submission_catboost_gpu_test_new_v1.csv` |
| v2 | + fold-safe transition priors | 0.3043 | 0.1789 | 0.0821 | 0.9994 | `outputs/submission_catboost_gpu_test_new_v2_transition.csv` |
| v3 | no action/point class weights | 0.4037 | 0.3140 | 0.1955 | 0.9995 | `outputs/submission_catboost_cpu_test_new_v3_no_class_weights.csv` |

Do not use v2/v3 as the primary submission unless leaderboard evidence contradicts local validation.

Audit validation signals and simple leakage risks:

```bash
aicup-audit-validation \
  --train train.csv \
  --test test_new.csv \
  --out outputs/audit_validation_test_new.json
```

Analyze OOF prediction errors after running with `--save-oof`:

```bash
aicup-error-analysis \
  --next-shot-oof outputs/oof_catboost_5fold_v1_next_shot.csv \
  --rally-oof outputs/oof_catboost_5fold_v1_rally.csv \
  --out-dir outputs/error_analysis_catboost_5fold_v1
```
