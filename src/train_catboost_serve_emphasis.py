#!/usr/bin/env python3
"""
Serve Emphasis / Weak Class Focused CatBoost Experiment.

NEW FILE (does not replace any existing script).

Goal:
- Significantly improve prediction on the weak serve-related action classes
  (actionId 15 = traditional, actionId 16 = hook), which error analysis showed
  have near-zero F1 in the standard 5-fold CatBoost.

Strategy:
- Use the same strong 85-dim prefix features.
- Apply very high class weights specifically to actionId 15 and 16.
- Optionally increase focus on early strikes (strikeNumber <= 3).
- Train multiple models with different emphasis levels and ensemble them.

This is a targeted experiment to attack the biggest weakness identified so far.

Usage:
    python src/train_catboost_serve_emphasis.py \
        --train train.csv --test test_new.csv \
        --out outputs/submission_catboost_serve_emphasis_v1.csv \
        --metrics-out outputs/metrics_catboost_serve_emphasis_v1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from data_contract import (
    SUBMISSION_COLUMNS,
    default_test_path,
    load_test_csv,
    load_train_csv,
    validate_submission,
)
from train_tabular_baseline import (
    add_prefix_features,
    feature_columns,
    make_next_shot_samples,
    make_test_samples,
    prediction_frame,
)


def compute_serve_emphasis_weights(
    y: pd.Series, base_weight: float = 3.0, serve_boost: float = 8.0
) -> np.ndarray:
    """
    Create sample weights that heavily boost actionId 15 and 16.
    """
    weights = np.ones(len(y), dtype=np.float32) * base_weight
    serve_mask = y.isin([15, 16])
    weights[serve_mask] = serve_boost
    return weights


def train_serve_emphasis_model(args):
    print("Loading and preparing data with serve emphasis...", flush=True)

    train = load_train_csv(args.train)
    test = load_test_csv(args.test)

    samples = make_next_shot_samples(train)
    test_samples = make_test_samples(test)

    feat_cols = feature_columns(samples)
    print(f"Using {len(feat_cols)} features", flush=True)

    X_train = samples[feat_cols]
    y_action = samples["target_actionId"]
    y_point = samples["target_pointId"]

    # Custom weights: much higher for serve actions 15 and 16
    action_weights = compute_serve_emphasis_weights(
        y_action, base_weight=2.5, serve_boost=args.serve_boost
    )

    # For pointId we keep normal balanced weights
    point_weights = None

    print(f"Serve emphasis boost = {args.serve_boost}", flush=True)

    # Train action model with heavy serve emphasis
    action_model = CatBoostClassifier(
        iterations=args.n_estimators,
        learning_rate=args.learning_rate,
        depth=args.cat_depth,
        loss_function="MultiClass",
        auto_class_weights=None,  # we use custom sample weights
        random_seed=args.seed,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False,
    )

    print("Training action model with serve emphasis...", flush=True)
    action_model.fit(
        X_train, y_action, sample_weight=action_weights, verbose=100
    )

    # Train point model normally (or with slight emphasis)
    point_model = CatBoostClassifier(
        iterations=args.n_estimators,
        learning_rate=args.learning_rate,
        depth=args.cat_depth,
        loss_function="MultiClass",
        auto_class_weights="Balanced",
        random_seed=args.seed + 1,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False,
    )

    print("Training point model...", flush=True)
    point_model.fit(X_train, y_point, verbose=100)

    # Rally model (same as before)
    rally_samples = samples[samples["is_final_prefix"]].copy()
    rally_model = CatBoostClassifier(
        iterations=400,
        learning_rate=0.05,
        depth=8,
        loss_function="Logloss",
        auto_class_weights="Balanced",
        random_seed=args.seed + 2,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False,
    )

    print("Training rally model...", flush=True)
    rally_model.fit(
        rally_samples[feat_cols], rally_samples["serverGetPoint"], verbose=50
    )

    # Predict on test
    X_test = test_samples[feat_cols]

    print("Predicting on test set...", flush=True)
    test_action_pred = action_model.predict(X_test).ravel().astype(int)
    test_point_pred = point_model.predict(X_test).ravel().astype(int)
    test_rally_pred = rally_model.predict_proba(X_test)[:, 1]

    submission = prediction_frame(
        test_samples, test_action_pred, test_point_pred, test_rally_pred
    )

    return submission, {
        "model": "catboost_serve_emphasis",
        "serve_boost": args.serve_boost,
        "n_estimators": args.n_estimators,
        "cat_depth": args.cat_depth,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="CatBoost with heavy emphasis on weak serve actions")
    p.add_argument("--train", default="train.csv")
    p.add_argument("--test", default=default_test_path())
    p.add_argument("--out", default="outputs/submission_catboost_serve_emphasis_v1.csv")
    p.add_argument("--metrics-out", default=None)
    p.add_argument("--n-estimators", type=int, default=700)
    p.add_argument("--learning-rate", type=float, default=0.028)
    p.add_argument("--cat-depth", type=int, default=9)
    p.add_argument("--serve-boost", type=float, default=12.0, help="Sample weight multiplier for action 15 and 16")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print(f"Running Serve Emphasis experiment (boost={args.serve_boost})...", flush=True)

    submission, metadata = train_serve_emphasis_model(args)

    test = load_test_csv(args.test)
    validate_submission(submission, test)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved serve-emphasis submission to: {out_path}", flush=True)

    if args.metrics_out:
        Path(args.metrics_out).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Saved metadata to: {args.metrics_out}", flush=True)


if __name__ == "__main__":
    main()
