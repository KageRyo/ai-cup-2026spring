#!/usr/bin/env python3
"""
Simple Ensemble script for AI CUP 2026 Spring table-tennis rally prediction.

NEW FILE (does not replace existing scripts).

Purpose:
- Combine predictions from multiple strong models (CatBoost 5-fold, LightGBM 5-fold, Hybrid RNN, etc.)
- Support simple probability averaging or weighted averaging.
- Can also do basic stacking if OOF predictions are available.

This is a high-ROI, low-risk way to improve the final score after we have several 5-fold models.

Usage examples:
    # Simple average of two submissions
    python src/train_ensemble.py \
        --submissions outputs/submission_catboost_5fold_v1.csv outputs/submission_lightgbm_5fold_v1.csv \
        --weights 0.6 0.4 \
        --out outputs/submission_ensemble_catboost_lgbm_v1.csv

    # Weighted ensemble with hybrid
    python src/train_ensemble.py \
        --submissions outputs/submission_catboost_5fold_v1.csv outputs/submission_hybrid_rnn_medium_v1.csv \
        --weights 0.7 0.3 \
        --out outputs/submission_ensemble_v2.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data_contract import SUBMISSION_COLUMNS, validate_submission, load_test_csv, default_test_path


def load_submission(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure correct column order and types
    df = df[SUBMISSION_COLUMNS]
    df["rally_uid"] = df["rally_uid"].astype(int)
    df["actionId"] = df["actionId"].astype(int)
    df["pointId"] = df["pointId"].astype(int)
    df["serverGetPoint"] = df["serverGetPoint"].astype(float)
    return df.sort_values("rally_uid").reset_index(drop=True)


def ensemble_submissions(
    submission_paths: list[str | Path],
    weights: list[float] | None = None,
) -> pd.DataFrame:
    """
    Average (weighted) the serverGetPoint probabilities and take argmax for actionId/pointId.
    For actionId and pointId we currently take the prediction from the highest-weight model
    (simple and stable). A more advanced version could average the underlying logits if available.
    """
    if weights is None:
        weights = [1.0 / len(submission_paths)] * len(submission_paths)

    assert len(submission_paths) == len(weights), "Number of submissions and weights must match"
    assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1.0"

    # Load all submissions
    subs = [load_submission(p) for p in submission_paths]

    # Use the first submission as the base (for rally_uid and structure)
    base = subs[0].copy()
    n = len(base)

    # Weighted average for serverGetPoint
    server_prob = np.zeros(n, dtype=np.float64)
    for sub, w in zip(subs, weights):
        server_prob += sub["serverGetPoint"].values * w

    base["serverGetPoint"] = server_prob

    # For actionId and pointId, take the prediction from the model with highest weight (stable)
    best_idx = int(np.argmax(weights))
    base["actionId"] = subs[best_idx]["actionId"].values
    base["pointId"] = subs[best_idx]["pointId"].values

    return base[SUBMISSION_COLUMNS].sort_values("rally_uid").reset_index(drop=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensemble multiple rally prediction submissions")
    parser.add_argument(
        "--submissions", nargs="+", required=True,
        help="Paths to submission CSV files to ensemble"
    )
    parser.add_argument(
        "--weights", nargs="+", type=float, default=None,
        help="Optional weights for each submission (must sum to 1.0). If not provided, equal weight is used."
    )
    parser.add_argument("--test", default=default_test_path(), help="Test file for validation")
    parser.add_argument("--out", required=True, help="Output path for ensembled submission")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    print(f"Loading {len(args.submissions)} submissions for ensembling...", flush=True)
    submission = ensemble_submissions(args.submissions, args.weights)

    # Validate against test set
    test = load_test_csv(args.test)
    validate_submission(submission, test)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False, encoding="utf-8")

    print(f"Ensemble submission saved to: {out_path}", flush=True)
    print(submission.head(8), flush=True)


if __name__ == "__main__":
    main()
