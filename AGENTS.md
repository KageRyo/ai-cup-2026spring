# Repository Guidelines

## Project Structure & Module Organization

This repository contains AI CUP 2026 Spring table-tennis rally prediction experiments.

- `src/`: Python module code and runnable training scripts.
- `src/train_lstm_baseline.py`: PyTorch LSTM baseline for sequence modeling.
- `src/train_tabular_baseline.py`: Tabular baseline experiments.
- `src/data_contract.py`: shared train/test/submission schema checks.
- `outputs/`: generated submissions and experiment artifacts; ignored by git.
- `train.csv`, `test.csv`, `test_new.csv`, `baseline_official/`, `conda-envs/`, `conda-pkgs/`, `.cache/`: local-only data, environments, and caches; ignored by git.
- Tests live under `tests/`.

## Build, Test, and Development Commands

Activate the project environment:

```bash
source /mnt/8tb_hdd/ryo/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/8tb_hdd/ryo/ai-cup-2026spring/conda-envs/ai-cup-ryo
```

Install the package in editable mode after changing `pyproject.toml` entry points:

```bash
python -m pip install --no-build-isolation --no-deps -e .
```

Run the LSTM baseline on the refreshed test set:

```bash
aicup-lstm-baseline --train train.csv --test test_new.csv --out outputs/submission_lstm_test_new.csv
```

Run the current best tabular baseline and save submission, metrics, validation predictions, and models:

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

Quick syntax check:

```bash
python -m py_compile src/*.py
```

## Coding Style & Naming Conventions

Use Python 3.10+, 4-space indentation, type hints where they clarify interfaces, and concise function names in `snake_case`. Keep scripts deterministic by exposing seeds through CLI arguments. Prefer structured pandas/sklearn/PyTorch APIs over ad hoc string handling. `ruff` is configured in `pyproject.toml` with line length 100.

## Testing Guidelines

Validate changes with `ruff`, `pytest`, `py_compile`, `--help` checks for CLI entry points, and a short smoke run using reduced epochs or estimators. Tests should live in `tests/` and be named `test_*.py`.

## Current Experiment Notes

The 2026-05-06 refreshed test set is `test_new.csv`. CLIs default to `test_new.csv` when present. Treat `rally_uid` only as an identifier; its numeric order is randomized and must not be used as sequence information.

Current best local run is `outputs/submission_catboost_gpu_test_new_v1.csv` with score `0.4143` (`f1_action=0.3311`, `f1_point=0.2050`, `auc=0.9995`). Models are saved in `outputs/models_catboost_gpu_test_new_v1/`.

Recent negative/secondary runs:

- `outputs/submission_catboost_gpu_test_new_v2_transition.csv`: fold-safe transition priors, score `0.3043`; do not use as primary submission.
- `outputs/submission_catboost_cpu_test_new_v3_no_class_weights.csv`: no action/point class weights, score `0.4037`; below current best.

## Commit & Pull Request Guidelines

Use concise Conventional Commit style, matching existing history:

- `init: scaffold lstm baseline`
- `feat(baseline): add tabular experiment`
- `fix(baseline): correct submission formatting`
- `chore: update ignored artifacts`

Before committing, run `git status --short --ignored` and ensure datasets, submissions, caches, conda environments, and official zip extracts are not staged. PRs should summarize model changes, validation metrics, generated submission paths, and any dependency or environment changes.

## Security & Configuration Tips

Do not commit competition data, generated predictions, model checkpoints, credentials, `.env`, `.codex`, or local conda/cache directories. GPU access may require running commands outside the Codex sandbox; verify with `nvidia-smi` and `torch.cuda.is_available()`.
