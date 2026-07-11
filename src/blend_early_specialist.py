#!/usr/bin/env python3
"""
Blend a strong base submission with an early-game specialist.

For test rallies where the observed prefix is very short (knownPrefixLen <= 3),
we trust the early-game specialist much more for actionId / pointId.
serverGetPoint can be blended or taken from the prior-corrected base.

This is the practical way to apply the specialist without a full retrain.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

from data_contract import (
    SUBMISSION_COLUMNS,
    default_test_path,
    load_test_csv,
    validate_submission,
)


def blend_submissions(
    base: pd.DataFrame,
    specialist: pd.DataFrame,
    test_samples: pd.DataFrame,
    early_threshold: int = 3,
    specialist_weight: float = 0.85,
) -> pd.DataFrame:
    """
    Replace or blend predictions for early-prefix test rows.
    All inputs must be sorted by rally_uid.
    """
    # Align specialist to base order (both should already be sorted by rally_uid)
    base = base.sort_values("rally_uid").reset_index(drop=True)
    specialist = specialist.sort_values("rally_uid").reset_index(drop=True)

    # Build early mask from the featured test_samples (also aligned)
    prefix_info = test_samples[["rally_uid", "knownPrefixLen"]].copy()
    prefix_info = prefix_info.sort_values("rally_uid").reset_index(drop=True)

    is_early = (prefix_info["knownPrefixLen"] <= early_threshold).values

    out = base.copy()

    # Hard override for action/point on early prefixes (specialist is stronger here)
    out.loc[is_early, "actionId"] = specialist.loc[is_early, "actionId"].values
    out.loc[is_early, "pointId"] = specialist.loc[is_early, "pointId"].values

    # Soft blend for serverGetPoint on early prefixes
    base_sp = base["serverGetPoint"].values
    spec_sp = specialist["serverGetPoint"].values
    blended_sp = base_sp.copy()
    blended_sp[is_early] = (
        specialist_weight * spec_sp[is_early] + (1.0 - specialist_weight) * base_sp[is_early]
    )
    out["serverGetPoint"] = blended_sp

    return out[SUBMISSION_COLUMNS]


def main(argv=None):
    p = argparse.ArgumentParser(description="Blend base submission with early-game specialist")
    p.add_argument("--base", required=True, help="Base submission (ideally already prior-corrected)")
    p.add_argument("--specialist", required=True, help="Early-game specialist submission")
    p.add_argument("--out", required=True)
    p.add_argument("--early-threshold", type=int, default=3)
    p.add_argument("--specialist-weight", type=float, default=0.82)
    p.add_argument("--validate", action="store_true")
    args = p.parse_args(argv)

    base = pd.read_csv(args.base)[SUBMISSION_COLUMNS]
    specialist = pd.read_csv(args.specialist)[SUBMISSION_COLUMNS]
    test = load_test_csv(default_test_path())
    # We need the featured columns (especially knownPrefixLen) for deciding early vs late
    from train_tabular_baseline import add_prefix_features
    featured_test = add_prefix_features(test)
    test_samples = featured_test.groupby("rally_uid", sort=False).tail(1).reset_index(drop=True)

    blended = blend_submissions(
        base, specialist, test_samples,
        early_threshold=args.early_threshold,
        specialist_weight=args.specialist_weight,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blended.to_csv(out_path, index=False, encoding="utf-8")
    logger.success(f"Saved blended submission to: {out_path}")

    early_count = (test_samples["knownPrefixLen"] <= args.early_threshold).sum()
    logger.info(f"  Early prefixes overridden: {early_count} / {len(test_samples)} "
                f"({early_count/len(test_samples)*100:.1f}%)")

    if args.validate:
        validate_submission(blended, test)
        logger.success("Validation passed.")


if __name__ == "__main__":
    main()
