#!/usr/bin/env python3
"""
Rally Opening Predictor — The Real "Deeper Layer" Solution

This is the independent model that does NOT rely on the next-shot formulation.

Core Idea:
- Every rally has a critical "opening moment" (the first stroke, almost always a serve: action 15 or 16).
- In test_new, ~81.6% of first strokes are action 15/16, and 70% of rallies are very short (length <= 3).
- The standard next-shot pipeline almost never trains on serve actions as targets.

This module creates a dedicated predictor that directly models:
1. What the opening action of a rally is likely to be (actionId at strikeNumber=1).
2. What the final serverGetPoint of that rally will be, given the opening characteristics.

At inference time, for any test rally whose observed prefix is short (strikeNumber <= 2 or 3),
we can heavily trust (or completely override with) this model's predictions.

This is the most direct attack on the fundamental limitation that has kept all previous
attempts (including serve-conditional) from improving LB beyond ~0.30.
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
from train_tabular_baseline import prediction_frame


def make_rally_opening_samples(train: pd.DataFrame) -> pd.DataFrame:
    """
    For each rally, create ONE sample that represents its opening.

    Features:
    - Basic context: sex, match, numberGame, gamePlayerId, gamePlayerOtherId
    - The actual first stroke features (strikeNumber == 1): this is what we observe
      in test for short rallies.
    - Simple opening score state (usually 0-0 at the very start).

    Targets:
    - actionId of the first stroke (the opening serve action)
    - serverGetPoint of the whole rally
    """
    # Take only the first stroke of each rally
    first = train[train["strikeNumber"] == 1].copy()

    # Features available at the "opening" of the rally
    feature_cols = [
        "sex",
        "match",
        "numberGame",
        "gamePlayerId",
        "gamePlayerOtherId",
        "strikeId",
        "handId",
        "strengthId",
        "spinId",
        "positionId",
        "scoreSelf",      # usually 0 at start
        "scoreOther",     # usually 0 at start
    ]

    samples = first[feature_cols + ["actionId", "serverGetPoint"]].copy()
    samples = samples.rename(columns={"actionId": "target_actionId"})

    # Add some derived opening features
    samples["is_deuce_start"] = ((samples["scoreSelf"] >= 10) & (samples["scoreOther"] >= 10)).astype("int8")
    samples["score_diff_start"] = samples["scoreSelf"] - samples["scoreOther"]

    logger.info(f"Created {len(samples):,} rally opening samples")
    logger.info(f"  Opening action distribution (top 5):\n{samples['target_actionId'].value_counts(normalize=True).head(5)*100}")

    return samples.reset_index(drop=True)


def train_rally_opening_models(args):
    logger.info("=== Rally Opening Predictor (True Deeper Fix) ===")
    logger.info("Building rally-level opening samples (one per rally)...")

    train = load_train_csv(args.train)
    test = load_test_csv(args.test)

    opening_samples = make_rally_opening_samples(train)

    # Feature columns (everything except targets)
    feat_cols = [c for c in opening_samples.columns if not c.startswith("target_") and c != "serverGetPoint"]

    X = opening_samples[feat_cols]
    y_action = opening_samples["target_actionId"]
    y_rally = opening_samples["serverGetPoint"]

    logger.info(f"Using {len(feat_cols)} features for Rally Opening models: {feat_cols}")

    # Action model: directly predict what the opening serve action is
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

    logger.info("Training Rally Opening Action model (predicting the first stroke action)...")
    action_model.fit(X, y_action, verbose=40)

    # Rally outcome model (serverGetPoint given the opening characteristics)
    rally_model = CatBoostClassifier(
        iterations=400,
        learning_rate=0.03,
        depth=7,
        loss_function="Logloss",
        auto_class_weights="Balanced",
        random_seed=args.seed + 1,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False,
    )

    logger.info("Training Rally Opening serverGetPoint model...")
    rally_model.fit(X, y_rally, verbose=30)

    # For test set, we also create "opening samples" using the first observed stroke(s)
    # In practice, for short rallies in test, the first row is the opening serve.
    test_first = test[test["strikeNumber"] == 1].copy()

    # Build the exact same features as training (including derived ones)
    test_opening = test_first[["sex", "match", "numberGame", "gamePlayerId", "gamePlayerOtherId",
                               "strikeId", "handId", "strengthId", "spinId", "positionId",
                               "scoreSelf", "scoreOther"]].copy()
    test_opening["is_deuce_start"] = ((test_opening["scoreSelf"] >= 10) & (test_opening["scoreOther"] >= 10)).astype("int8")
    test_opening["score_diff_start"] = test_opening["scoreSelf"] - test_opening["scoreOther"]

    X_test = test_opening[feat_cols]

    logger.info("Predicting opening actions and rally outcomes with the Opening Predictor...")
    action_pred = action_model.predict(X_test).ravel().astype(int)
    rally_pred = rally_model.predict_proba(X_test)[:, 1]

    # We need to produce a full submission.
    # Strategy: For rallies whose observed prefix is short, we will later override
    # using this model's predictions in the blender.
    # Here we produce a "pure opening predictor" submission for blending.

    # Build a submission aligned to test rallies (one row per rally)
    # We use the first stroke's rally_uid as anchor
    submission = pd.DataFrame({
        "rally_uid": test_first["rally_uid"].values,
        "actionId": action_pred,
        "pointId": 0,                    # placeholder, will be filled by blender
        "serverGetPoint": rally_pred,
    })

    # Make sure we have all test rallies (some test rallies may start from strike > 1 in theory)
    all_rallies = test[["rally_uid"]].drop_duplicates()
    submission = all_rallies.merge(submission, on="rally_uid", how="left")

    # Fill missing (very rare) with reasonable defaults
    submission["actionId"] = submission["actionId"].fillna(1).astype(int)
    submission["pointId"] = submission["pointId"].fillna(0).astype(int)
    submission["serverGetPoint"] = submission["serverGetPoint"].fillna(0.5)

    submission = submission[SUBMISSION_COLUMNS].sort_values("rally_uid").reset_index(drop=True)

    metadata = {
        "model": "rally_opening_predictor",
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "cat_depth": args.cat_depth,
        "n_rallies_train": len(opening_samples),
        "note": "Directly predicts opening action and rally outcome, bypassing next-shot formulation",
    }

    return submission, metadata


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Rally Opening Predictor — independent model for the critical first stroke of each rally"
    )
    p.add_argument("--train", default="train.csv")
    p.add_argument("--test", default=default_test_path())
    p.add_argument("--out", default="outputs/submission_rally_opening_v1.csv")
    p.add_argument("--metrics-out", default=None)
    p.add_argument("--n-estimators", type=int, default=600)
    p.add_argument("--learning-rate", type=float, default=0.025)
    p.add_argument("--cat-depth", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logger.info("Running Rally Opening Predictor (the real deeper architectural fix)...")

    submission, metadata = train_rally_opening_models(args)

    test = load_test_csv(args.test)
    validate_submission(submission, test)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False, encoding="utf-8")
    logger.success(f"Saved Rally Opening Predictor submission to: {out_path}")

    if args.metrics_out:
        Path(args.metrics_out).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logger.success(f"Saved metadata to: {args.metrics_out}")


if __name__ == "__main__":
    main()
