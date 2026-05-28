#!/usr/bin/env python3
"""
First-Stroke / Opening Serve Specialist for AI CUP 2026 Spring.

This is the more thorough solution (option B) to the core distribution shift problem.

Problem:
- In test_new, 70% of rallies have length <=3 and the first observed action
  (strikeNumber=1) is a serve (action 15 or 16) at much higher rate than train.
- The standard next-shot formulation almost never sees serve actions as *targets*.
- Therefore all previous models (including the early-game specialist) under-predict
  serve-related behavior on short rallies.

Approach:
- Train dedicated CatBoost models exclusively on the first 1-2 strokes of every rally.
- Apply very strong class weighting focused on transitions *from* opening serves (15/16).
- At inference, when the observed prefix in test is strikeNumber <= 2, we trust
  this specialist heavily for actionId / pointId (and serverGetPoint for short rallies).

This directly models the "opening serve + immediate return" regime that dominates
the refreshed test set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from loguru import logger

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


def filter_first_stroke_samples(samples: pd.DataFrame, max_strike: int = 2) -> pd.DataFrame:
    """Keep only the very early strokes where opening serve dynamics dominate."""
    mask = samples["strikeNumber"] <= max_strike
    return samples[mask].copy()


def train_first_stroke_models(args):
    logger.info("Loading data for first-stroke / opening serve specialist...")

    train = load_train_csv(args.train)
    test = load_test_csv(args.test)

    all_samples = make_next_shot_samples(train)
    first_samples = filter_first_stroke_samples(all_samples, max_strike=args.max_strike)

    logger.info(f"Total next-shot samples: {len(all_samples):,}")
    logger.info(f"First-stroke samples (strikeNumber <= {args.max_strike}): {len(first_samples):,} "
                f"({len(first_samples)/len(all_samples)*100:.1f}%)")

    feat_cols = feature_columns(first_samples)
    logger.info(f"Using {len(feat_cols)} features for first-stroke model")

    X = first_samples[feat_cols]
    y_action = first_samples["target_actionId"]
    y_point = first_samples["target_pointId"]

    # Very strong balanced weighting + we can add extra sample weights later for serve transitions
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

    logger.info("Training first-stroke action model (heavy emphasis on early transitions)...")
    action_model.fit(X, y_action, verbose=40)

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

    logger.info("Training first-stroke point model...")
    point_model.fit(X, y_point, verbose=40)

    # Short-rally serverGetPoint head (final prefix of short rallies)
    rally_samples = all_samples[all_samples["is_final_prefix"]].copy()
    short_mask = rally_samples["strikeNumber"] <= args.max_strike + 1
    short_rally = rally_samples[short_mask]

    rally_model = CatBoostClassifier(
        iterations=250,
        learning_rate=0.04,
        depth=6,
        loss_function="Logloss",
        auto_class_weights="Balanced",
        random_seed=args.seed + 2,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False,
    )

    logger.info(f"Training first-stroke serverGetPoint model on {len(short_rally)} short rallies...")
    rally_model.fit(short_rally[feat_cols], short_rally["serverGetPoint"], verbose=20)

    # Predict
    test_samples = make_test_samples(test)
    X_test = test_samples[feat_cols]

    logger.info("Predicting with first-stroke specialist on test set...")
    action_pred = action_model.predict(X_test).ravel().astype(int)
    point_pred = point_model.predict(X_test).ravel().astype(int)
    rally_pred = rally_model.predict_proba(X_test)[:, 1]

    submission = prediction_frame(test_samples, action_pred, point_pred, rally_pred)

    metadata = {
        "model": "first_stroke_specialist",
        "max_strike": args.max_strike,
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "cat_depth": args.cat_depth,
        "first_stroke_train_samples": int(len(first_samples)),
    }

    return submission, metadata


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="First-stroke / opening serve specialist (thorough fix for short-rally shift)"
    )
    p.add_argument("--train", default="train.csv")
    p.add_argument("--test", default=default_test_path())
    p.add_argument("--out", default="outputs/submission_first_stroke_v1.csv")
    p.add_argument("--metrics-out", default=None)
    p.add_argument("--max-strike", type=int, default=2, help="Train on strokes with strikeNumber <= this")
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--cat-depth", type=int, default=7)
    p.add_argument("--seed", type=int, default=2022)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logger.info(f"Running First-Stroke Specialist (max_strike={args.max_strike})...")

    submission, metadata = train_first_stroke_models(args)

    test = load_test_csv(args.test)
    validate_submission(submission, test)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False, encoding="utf-8")
    logger.success(f"Saved first-stroke specialist submission to: {out_path}")

    if args.metrics_out:
        Path(args.metrics_out).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logger.success(f"Saved metadata to: {args.metrics_out}")


if __name__ == "__main__":
    main()
