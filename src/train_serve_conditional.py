#!/usr/bin/env python3
"""
Serve-Conditional Return Specialist — The Deeper Architectural Fix (更下一層的徹底解)

Core Problem (Root Cause Analysis):
- Opening serves (actionId 15/16) dominate the first stroke in test_new (~81.6% combined).
- 70% of test rallies are very short (length <= 3).
- In the standard next-shot formulation, serve actions almost never appear as *targets*.
- Even "first-stroke" or "early-game" specialists trained on strikeNumber <= 2/3 still use the
  generic next-shot framing and are heavily diluted by long-rally patterns from train.

This module attacks the problem at a deeper level:

1. Explicitly isolate "post-serve" prediction problems:
   - Training samples where the *current* observed stroke (the prefix we condition on)
     has actionId in {15, 16} (i.e., we just saw the opening serve).
   - These are exactly the cases of predicting the return shot + point outcome
     right after a serve.

2. Train a dedicated, high-capacity specialist only on these post-serve transitions,
   with strong emphasis on short-rally dynamics and serve characteristics
   (handId, spinId, strengthId, positionId, and their interactions).

3. Provide routing logic: at inference, when the last observed action in a test rally
   is a serve, route the prediction through this specialist.

This is more targeted than previous early/first-stroke specialists and directly models
the dominant regime in the refreshed test set.

Usage after training any new model:
    aicup-make-final \
        --base ... \
        --first-stroke ... \
        --serve-conditional outputs/submission_serve_conditional_v1.csv \
        --out outputs/submission_final_vX.csv
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
    SHOT_COLUMNS,
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


SERVE_ACTIONS = {15, 16}


def make_post_serve_samples(samples: pd.DataFrame) -> pd.DataFrame:
    """
    Create training samples specifically for the moment right after an opening serve.

    We keep only next-shot samples where the *current* action (the one we just observed,
    i.e. the prefix) was a serve action (15 or 16). This means we are predicting
    the immediate return shot + the final serverGetPoint after a serve.

    These samples are the most critical for test_new because:
    - Most short rallies in test are decided right after the opening serve.
    - The distribution of returns and point winners after a strong serve differs
      significantly from the average long-rally behavior in train.
    """
    # The "current" action for a next-shot sample is the actionId of the current row.
    # We want cases where that current action was a serve.
    post_serve_mask = samples["actionId"].isin(SERVE_ACTIONS)
    post_serve = samples[post_serve_mask].copy()

    logger.info(f"Post-serve samples: {len(post_serve):,} / {len(samples):,} "
                f"({len(post_serve)/len(samples)*100:.1f}%)")
    return post_serve.reset_index(drop=True)


def add_serve_conditional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features that are especially powerful right after a serve.
    These emphasize the quality and type of the opening serve.
    """
    df = df.copy()

    # Serve quality signals (very important after action 15/16)
    df["serve_is_traditional"] = (df["actionId"] == 15).astype("int8")
    df["serve_is_hook"] = (df["actionId"] == 16).astype("int8")
    df["serve_hand_strong"] = ((df["actionId"].isin(SERVE_ACTIONS)) &
                               (df["handId"] == 1)).astype("int8")  # assuming 1 = strong hand, adjust if needed
    df["serve_spin_heavy"] = ((df["actionId"].isin(SERVE_ACTIONS)) &
                              (df["spinId"] >= 2)).astype("int8")   # example encoding

    # Interaction between serve characteristics and current score situation
    df["serve_after_leading"] = df["serve_is_traditional"] * df["isLeading"]
    df["serve_after_trailing"] = df["serve_is_hook"] * df["isTrailing"]

    return df


def train_serve_conditional_models(args):
    logger.info("=== Serve-Conditional Return Specialist (Deeper Fix) ===")
    logger.info("Loading and preparing post-serve specific training data...")

    train = load_train_csv(args.train)
    test = load_test_csv(args.test)

    # Build the usual next-shot samples, then filter to post-serve only
    all_samples = make_next_shot_samples(train)
    post_serve_samples = make_post_serve_samples(all_samples)

    # Enrich with serve-specific features
    post_serve_samples = add_serve_conditional_features(post_serve_samples)

    feat_cols = feature_columns(post_serve_samples)
    # Add our new conditional features if they are not already excluded
    extra_serve_feats = [
        "serve_is_traditional", "serve_is_hook",
        "serve_hand_strong", "serve_spin_heavy",
        "serve_after_leading", "serve_after_trailing",
    ]
    for f in extra_serve_feats:
        if f in post_serve_samples.columns and f not in feat_cols:
            feat_cols.append(f)

    logger.info(f"Final feature count for serve-conditional model: {len(feat_cols)}")

    X = post_serve_samples[feat_cols]
    y_action = post_serve_samples["target_actionId"]
    y_point = post_serve_samples["target_pointId"]

    # Dedicated action model for returns after serves — use higher capacity
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

    logger.info("Training serve-conditional action model (returns after opening serve)...")
    action_model.fit(X, y_action, verbose=30)

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

    logger.info("Training serve-conditional point model...")
    point_model.fit(X, y_point, verbose=30)

    # Rally outcome after serve (critical for short-rally serverGetPoint)
    rally_samples = post_serve_samples[post_serve_samples["is_final_prefix"]].copy()
    rally_model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.035,
        depth=7,
        loss_function="Logloss",
        auto_class_weights="Balanced",
        random_seed=args.seed + 2,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False,
    )

    logger.info(f"Training serve-conditional serverGetPoint model on {len(rally_samples)} post-serve rallies...")
    rally_model.fit(rally_samples[feat_cols], rally_samples["serverGetPoint"], verbose=20)

    # Prepare test features with the same serve-conditional enrichment
    test_samples = make_test_samples(test)
    test_samples = add_serve_conditional_features(test_samples)
    X_test = test_samples[feat_cols]

    logger.info("Predicting with serve-conditional specialist...")
    action_pred = action_model.predict(X_test).ravel().astype(int)
    point_pred = point_model.predict(X_test).ravel().astype(int)
    rally_pred = rally_model.predict_proba(X_test)[:, 1]

    submission = prediction_frame(test_samples, action_pred, point_pred, rally_pred)

    metadata = {
        "model": "serve_conditional",
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "cat_depth": args.cat_depth,
        "post_serve_train_samples": int(len(post_serve_samples)),
        "note": "Specialist for predictions immediately after opening serves (action 15/16)",
    }

    return submission, metadata, feat_cols


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Serve-Conditional Return Specialist — deeper fix for opening serve + short rally regime"
    )
    p.add_argument("--train", default="train.csv")
    p.add_argument("--test", default=default_test_path())
    p.add_argument("--out", default="outputs/submission_serve_conditional_v1.csv")
    p.add_argument("--metrics-out", default=None)
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--learning-rate", type=float, default=0.025)
    p.add_argument("--cat-depth", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    submission, metadata, _ = train_serve_conditional_models(args)

    test = load_test_csv(args.test)
    validate_submission(submission, test)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False, encoding="utf-8")
    logger.success(f"Saved serve-conditional submission to: {out_path}")

    if args.metrics_out:
        Path(args.metrics_out).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logger.success(f"Saved metadata to: {args.metrics_out}")


if __name__ == "__main__":
    main()
