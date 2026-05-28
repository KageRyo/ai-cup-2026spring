#!/usr/bin/env python3
"""
Prior correction utilities for AI CUP 2026 Spring.

Addresses the large serverGetPoint distribution shift between train (55%)
and test_new (~37.6% via last-score proxy).

Main method: odds-ratio adjustment (preserves ranking, shifts mean probability
to match target prior). This is a form of probability calibration for known prior shift.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from data_contract import SUBMISSION_COLUMNS, load_test_csv, default_test_path, validate_submission


def adjust_binary_prior(
    probs: np.ndarray | pd.Series,
    target_prior: float = 0.37615,
    max_iter: int = 60,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Adjust predicted probabilities to match a target prior using additive bias
    in logit space.

    This is far more effective than simple odds-ratio when the raw model is
    extremely over-confident (AUC ~0.9995, most probabilities pushed to 0.0 or 1.0).

    We binary-search for a single bias b such that:
        mean( sigmoid( logit(p_raw) + b ) ) ≈ target_prior

    This perfectly preserves ranking/order while shifting the effective
    decision threshold to match the test distribution.
    """
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 1e-9, 1.0 - 1e-9)

    logits = np.log(probs / (1.0 - probs))

    # Binary search for bias that produces the desired mean probability
    lo, hi = -30.0, 30.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        adjusted = 1.0 / (1.0 + np.exp(-(logits + mid)))
        mean_p = adjusted.mean()
        if abs(mean_p - target_prior) < tol:
            break
        if mean_p > target_prior:
            hi = mid
        else:
            lo = mid

    bias = (lo + hi) / 2.0
    adjusted = 1.0 / (1.0 + np.exp(-(logits + bias)))
    return np.clip(adjusted, 0.0, 1.0)


def adjust_submission_server_prior(
    submission: pd.DataFrame,
    target_prior: float = 0.37615,
) -> pd.DataFrame:
    """
    Apply prior correction to the serverGetPoint column of a submission.
    Does NOT change actionId / pointId.
    """
    out = submission.copy()
    out["serverGetPoint"] = adjust_binary_prior(
        out["serverGetPoint"].values, target_prior=target_prior
    )
    return out


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Apply serverGetPoint prior correction to a submission "
                    "(shifts from ~55% train prior to ~37.6% test proxy prior using logit bias)"
    )
    p.add_argument("--in", dest="input", required=True, help="Input submission CSV")
    p.add_argument("--out", required=True, help="Output corrected submission CSV")
    p.add_argument("--target-prior", type=float, default=0.37615, help="Target positive rate (use proxy from audit)")
    p.add_argument("--validate", action="store_true", help="Run submission validation against test_new.csv")
    args = p.parse_args(argv)

    sub = pd.read_csv(args.input)
    corrected = adjust_submission_server_prior(
        sub, target_prior=args.target_prior
    )

    # Keep column order
    corrected = corrected[SUBMISSION_COLUMNS]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    corrected.to_csv(out_path, index=False, encoding="utf-8")
    logger.success(f"Saved prior-corrected submission to: {out_path}")
    logger.info(f"  Original serverGetPoint mean: {sub['serverGetPoint'].mean():.4f}")
    logger.info(f"  Corrected serverGetPoint mean: {corrected['serverGetPoint'].mean():.4f}")
    logger.info(f"  (target prior was {args.target_prior:.5f})")

    if args.validate:
        test = load_test_csv(default_test_path())
        validate_submission(corrected, test)
        logger.success("Validation passed.")


if __name__ == "__main__":
    main()
