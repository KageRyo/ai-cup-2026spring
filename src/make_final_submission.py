#!/usr/bin/env python3
"""
Automatic Final Submission Builder for AI CUP 2026 Spring.

Usage after any new specialist training finishes:

    aicup-make-final \
        --first-stroke outputs/submission_first_stroke_v1.csv \
        --early-game outputs/submission_early_game_specialist_v1.csv \
        --base outputs/submission_catboost_gpu_seed1011_v6_prior_corrected.csv \
        --out outputs/submission_final_v2.csv

This script:
1. Ensures serverGetPoint prior correction is applied to the base.
2. Performs tiered blending:
   - strikeNumber <= 2 → heavily use first-stroke specialist
   - strikeNumber 3-4 → use early-game specialist
   - later prefixes → use the strong base model
3. Produces a clean, well-documented final submission ready for upload.

Run this right after any new `aicup-*-specialist` training completes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from data_contract import (
    SUBMISSION_COLUMNS,
    default_test_path,
    load_test_csv,
    validate_submission,
)
from prior_correction import adjust_submission_server_prior
from train_tabular_baseline import add_prefix_features


def tiered_blend(
    base: pd.DataFrame,
    first_stroke: pd.DataFrame | None,
    early_game: pd.DataFrame | None,
    test_samples: pd.DataFrame,
    fs_weight: float = 0.82,
    eg_weight: float = 0.65,
) -> pd.DataFrame:
    """
    Tiered blending strategy based on observed prefix length.
    """
    base = base.sort_values("rally_uid").reset_index(drop=True)
    out = base.copy()

    strike = test_samples["strikeNumber"].values

    # Tier 1: very early (strike <= 2) - first stroke specialist dominates
    if first_stroke is not None:
        fs = first_stroke.sort_values("rally_uid").reset_index(drop=True)
        very_early = strike <= 2
        out.loc[very_early, "actionId"] = fs.loc[very_early, "actionId"].values
        out.loc[very_early, "pointId"] = fs.loc[very_early, "pointId"].values
        out.loc[very_early, "serverGetPoint"] = (
            fs_weight * fs.loc[very_early, "serverGetPoint"].values +
            (1 - fs_weight) * base.loc[very_early, "serverGetPoint"].values
        )
        logger.info(f"Applied first-stroke override to {very_early.sum()} rallies (strike <= 2)")

    # Tier 2: early (strike 3-4) - early-game specialist
    if early_game is not None:
        eg = early_game.sort_values("rally_uid").reset_index(drop=True)
        early = (strike > 2) & (strike <= 4)
        out.loc[early, "actionId"] = eg.loc[early, "actionId"].values
        out.loc[early, "pointId"] = eg.loc[early, "pointId"].values
        out.loc[early, "serverGetPoint"] = (
            eg_weight * eg.loc[early, "serverGetPoint"].values +
            (1 - eg_weight) * out.loc[early, "serverGetPoint"].values   # use already possibly FS-blended
        )
        logger.info(f"Applied early-game override to {early.sum()} rallies (strike 3-4)")

    return out[SUBMISSION_COLUMNS]


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Automatically build the best final blended submission after specialist training"
    )
    p.add_argument("--base", required=True,
                   help="Strong base submission (will auto-apply prior correction if needed)")
    p.add_argument("--first-stroke", default=None,
                   help="First-stroke specialist submission (highest priority for strike <= 2)")
    p.add_argument("--early-game", default=None,
                   help="Early-game specialist submission (for strike 3-4)")
    p.add_argument("--out", required=True, help="Output final submission path")
    p.add_argument("--fs-weight", type=float, default=0.82, help="Weight for first-stroke in tier 1")
    p.add_argument("--eg-weight", type=float, default=0.65, help="Weight for early-game in tier 2")
    p.add_argument("--validate", action="store_true")
    args = p.parse_args(argv)

    logger.info("Building final blended submission...")

    base = pd.read_csv(args.base)[SUBMISSION_COLUMNS]

    # Auto-apply prior correction if the base still has train-like serverGetPoint mean
    if base["serverGetPoint"].mean() > 0.48:
        logger.info("Base has high serverGetPoint mean → applying prior correction automatically")
        base = adjust_submission_server_prior(base, target_prior=0.37615)

    first_stroke = pd.read_csv(args.first_stroke)[SUBMISSION_COLUMNS] if args.first_stroke else None
    early_game = pd.read_csv(args.early_game)[SUBMISSION_COLUMNS] if args.early_game else None

    test = load_test_csv(default_test_path())
    featured = add_prefix_features(test)
    test_samples = featured.groupby("rally_uid", sort=False).tail(1).reset_index(drop=True)

    final = tiered_blend(
        base, first_stroke, early_game, test_samples,
        fs_weight=args.fs_weight,
        eg_weight=args.eg_weight,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(out_path, index=False, encoding="utf-8")

    logger.success(f"Final blended submission saved to: {out_path}")
    logger.info(f"  serverGetPoint mean: {final['serverGetPoint'].mean():.4f}")

    if args.validate:
        validate_submission(final, test)
        logger.success("Validation passed.")


if __name__ == "__main__":
    main()
