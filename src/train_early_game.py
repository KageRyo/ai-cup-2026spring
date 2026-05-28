#!/usr/bin/env python3
"""
Early-Game / Short-Rally Specialist for AI CUP 2026 Spring.

Addresses the critical distribution shift:
- test_new has 70% rallies of length <= 3 (train only 30%)
- Median rally length in test_new is 2 vs 5 in train
- First-stroke serve actions (15/16) are much more prevalent

Strategy:
- Train dedicated CatBoost action + point models ONLY on early prefixes
  (knownPrefixLen <= 3 or strikeNumber <= 3).
- Use strong balanced class weights + extra emphasis on serve-related transitions.
- At inference, for test samples with low prefix length, use this specialist
  (or heavily blend toward it). Later prefixes fall back to general models.

This is a targeted fix for the main reason local 0.41+ became 0.3x on LB.
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


def filter_early_samples(samples: pd.DataFrame, max_prefix_len: int = 3) -> pd.DataFrame:
    """Keep only early prefixes where the model has seen very little history."""
    mask = samples["knownPrefixLen"] <= max_prefix_len
    return samples[mask].copy()


def train_early_game_models(args):
    print("Loading data for early-game specialist...", flush=True)

    train = load_train_csv(args.train)
    test = load_test_csv(args.test)

    # Build next-shot samples (same as main pipeline)
    all_samples = make_next_shot_samples(train)
    early_samples = filter_early_samples(all_samples, max_prefix_len=args.max_prefix_len)

    print(f"Total next-shot samples: {len(all_samples):,}")
    print(f"Early samples (prefix <= {args.max_prefix_len}): {len(early_samples):,} "
          f"({len(early_samples)/len(all_samples)*100:.1f}%)", flush=True)

    feat_cols = feature_columns(early_samples)
    print(f"Using {len(feat_cols)} features for early-game model", flush=True)

    X = early_samples[feat_cols]
    y_action = early_samples["target_actionId"]
    y_point = early_samples["target_pointId"]

    # Stronger class weighting for early game (where serve actions matter more)
    action_model = CatBoostClassifier(
        iterations=args.n_estimators,
        learning_rate=args.learning_rate,
        depth=args.cat_depth,
        loss_function="MultiClass",
        auto_class_weights="Balanced",
        random_seed=args.seed,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False,
    )

    print("Training early-game action model...", flush=True)
    action_model.fit(X, y_action, verbose=50)

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

    print("Training early-game point model...", flush=True)
    point_model.fit(X, y_point, verbose=50)

    # Also train a short-rally serverGetPoint model (final prefix of short rallies)
    # We use the last prefix of rallies that were short in training
    rally_samples = all_samples[all_samples["is_final_prefix"]].copy()
    short_rally_mask = rally_samples["knownPrefixLen"] <= args.max_prefix_len + 1
    short_rally_samples = rally_samples[short_rally_mask]

    rally_model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.04,
        depth=7,
        loss_function="Logloss",
        auto_class_weights="Balanced",
        random_seed=args.seed + 2,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False,
    )

    print(f"Training early/short-rally serverGetPoint model on {len(short_rally_samples)} samples...", flush=True)
    rally_model.fit(
        short_rally_samples[feat_cols],
        short_rally_samples["serverGetPoint"],
        verbose=30,
    )

    # Predict on test
    test_samples = make_test_samples(test)
    X_test = test_samples[feat_cols]

    # For early-game specialist we still predict on ALL test rows,
    # but caller will decide when to trust these predictions.
    print("Predicting with early-game specialist on test set...", flush=True)
    action_pred = action_model.predict(X_test).ravel().astype(int)
    point_pred = point_model.predict(X_test).ravel().astype(int)
    rally_pred = rally_model.predict_proba(X_test)[:, 1]

    submission = prediction_frame(test_samples, action_pred, point_pred, rally_pred)

    metadata = {
        "model": "early_game_specialist",
        "max_prefix_len": args.max_prefix_len,
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "cat_depth": args.cat_depth,
        "early_train_samples": int(len(early_samples)),
    }

    return submission, metadata, {
        "action_model": action_model,
        "point_model": point_model,
        "rally_model": rally_model,
        "feature_cols": feat_cols,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Early-game / short-rally specialist CatBoost")
    p.add_argument("--train", default="train.csv")
    p.add_argument("--test", default=default_test_path())
    p.add_argument("--out", default="outputs/submission_early_game_specialist_v1.csv")
    p.add_argument("--metrics-out", default=None)
    p.add_argument("--max-prefix-len", type=int, default=3, help="Train only on prefixes <= this length")
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--cat-depth", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print(f"Running Early-Game Specialist (max_prefix_len={args.max_prefix_len})...", flush=True)

    submission, metadata, models = train_early_game_models(args)

    test = load_test_csv(args.test)
    validate_submission(submission, test)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved early-game specialist submission to: {out_path}", flush=True)

    if args.metrics_out:
        Path(args.metrics_out).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Saved metadata to: {args.metrics_out}", flush=True)


if __name__ == "__main__":
    main()
