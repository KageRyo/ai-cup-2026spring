# Repository Guidelines

## Project Structure & Module Organization

This repository contains AI CUP 2026 Spring table-tennis rally prediction experiments.

- `src/aicup_ryo/`: Python package code and runnable training scripts.
- `src/aicup_ryo/train_lstm_baseline.py`: PyTorch LSTM baseline for sequence modeling.
- `src/aicup_ryo/train_tabular_baseline.py`: Tabular baseline experiments.
- `outputs/`: generated submissions and experiment artifacts; ignored by git.
- `train.csv`, `test.csv`, `baseline_official/`, `conda-envs/`, `conda-pkgs/`, `.cache/`: local-only data, environments, and caches; ignored by git.
- Tests are not yet present. Add future tests under `tests/`.

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

Run the LSTM baseline:

```bash
aicup-lstm-baseline --train train.csv --test test.csv --out outputs/submission_lstm_baseline.csv
```

Run the tabular baseline:

```bash
aicup-tabular-baseline --model catboost --train train.csv --test test.csv --out outputs/submission_catboost.csv
```

Quick syntax check:

```bash
python -m py_compile src/aicup_ryo/*.py
```

## Coding Style & Naming Conventions

Use Python 3.10+, 4-space indentation, type hints where they clarify interfaces, and concise function names in `snake_case`. Keep scripts deterministic by exposing seeds through CLI arguments. Prefer structured pandas/sklearn/PyTorch APIs over ad hoc string handling. `ruff` is configured in `pyproject.toml` with line length 100.

## Testing Guidelines

No formal test suite exists yet. For now, validate changes with `py_compile`, `--help` checks for CLI entry points, and a short smoke run using reduced epochs or estimators. Future tests should live in `tests/` and be named `test_*.py`.

## Commit & Pull Request Guidelines

Use concise Conventional Commit style, matching existing history:

- `init: scaffold lstm baseline`
- `feat(baseline): add tabular experiment`
- `fix(baseline): correct submission formatting`
- `chore: update ignored artifacts`

Before committing, run `git status --short --ignored` and ensure datasets, submissions, caches, conda environments, and official zip extracts are not staged. PRs should summarize model changes, validation metrics, generated submission paths, and any dependency or environment changes.

## Security & Configuration Tips

Do not commit competition data, generated predictions, model checkpoints, credentials, `.env`, `.codex`, or local conda/cache directories. GPU access may require running commands outside the Codex sandbox; verify with `nvidia-smi` and `torch.cuda.is_available()`.

