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
