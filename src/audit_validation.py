from __future__ import annotations

import argparse
from loguru import logger
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier

from data_contract import default_test_path, load_test_csv, load_train_csv
from train_tabular_baseline import (
    feature_columns,
    make_next_shot_samples,
    make_rally_samples,
    make_test_samples,
)


SEED = 42


def split_by_match(frame: pd.DataFrame, val_size: float, seed: int):
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(frame, groups=frame["match"]))
    return frame.iloc[train_idx].copy(), frame.iloc[val_idx].copy()


def single_feature_auc(train: pd.DataFrame, val: pd.DataFrame, features: list[str]) -> list[dict]:
    rows = []
    for feature in features:
        model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, random_state=SEED)
        model.fit(train[[feature]], train["serverGetPoint"])
        pred = model.predict_proba(val[[feature]])[:, 1]
        auc = roc_auc_score(val["serverGetPoint"], pred) if val["serverGetPoint"].nunique() > 1 else 0.5
        rows.append({"feature": feature, "auc": float(auc)})
    return sorted(rows, key=lambda row: row["auc"], reverse=True)


def exact_next_shot_rules(val: pd.DataFrame) -> list[dict]:
    rows = []
    candidates = [
        "actionId",
        "prev_actionId",
        "prev2_actionId",
        "pointId",
        "prev_pointId",
        "prev2_pointId",
        "strikeId",
        "prev_strikeId",
        "positionId",
        "prev_positionId",
    ]
    for feature in candidates:
        if feature not in val:
            continue
        if feature.endswith("pointId"):
            target = "target_pointId"
            metric = f1_score(val[target], val[feature], average="macro")
        elif feature.endswith("actionId"):
            target = "target_actionId"
            metric = f1_score(val[target], val[feature], average="macro")
        else:
            continue
        rows.append(
            {
                "feature": feature,
                "target": target,
                "accuracy": float(accuracy_score(val[target], val[feature])),
                "macro_f1": float(metric),
            }
        )
    return sorted(rows, key=lambda row: row["macro_f1"], reverse=True)


def audit(args) -> dict:
    train = load_train_csv(args.train)
    test = load_test_csv(args.test)

    next_shot = make_next_shot_samples(train)
    rally = make_rally_samples(train)
    test_rally = make_test_samples(test)

    _, val_next = split_by_match(next_shot, args.val_size, args.seed)
    train_rally, val_rally = split_by_match(rally, args.val_size, args.seed)
    features = feature_columns(rally)

    result = {
        "train_rows": int(len(train)),
        "train_rallies": int(train["rally_uid"].nunique()),
        "test_rows": int(len(test)),
        "test_rallies": int(test["rally_uid"].nunique()),
        "rally_features": int(len(features)),
        "rally_single_feature_auc_top": single_feature_auc(train_rally, val_rally, features)[
            : args.top_k
        ],
        "next_shot_exact_rule_top": exact_next_shot_rules(val_next)[: args.top_k],
        "server_get_point_rate": {
            "train": float(train.groupby("rally_uid")["serverGetPoint"].first().mean()),
            "test_proxy_last_score_self_gt_other": float(
                (test_rally["scoreSelf"] > test_rally["scoreOther"]).mean()
            ),
        },
    }
    return result


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default=default_test_path())
    parser.add_argument("--out", default="outputs/audit_validation.json")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--top-k", type=int, default=15)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = audit(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info(json.dumps(result, indent=2, sort_keys=True))
    logger.info(f"Saved validation audit to: {out_path}")


if __name__ == "__main__":
    main()
