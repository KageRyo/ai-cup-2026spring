from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit


SEED = 42
SUBMISSION_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
SHOT_COLUMNS = ["strikeId", "handId", "strengthId", "spinId", "pointId", "actionId", "positionId"]
BASE_COLUMNS = [
    "sex",
    "match",
    "numberGame",
    "rally_id",
    "strikeNumber",
    "scoreSelf",
    "scoreOther",
    "gamePlayerId",
    "gamePlayerOtherId",
    *SHOT_COLUMNS,
]


def add_prefix_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["rally_uid", "strikeNumber"]).copy()
    group = df.groupby("rally_uid", sort=False)

    df["scoreDiff"] = df["scoreSelf"] - df["scoreOther"]
    df["scoreSum"] = df["scoreSelf"] + df["scoreOther"]
    df["isServerStroke"] = (df["strikeNumber"] % 2 == 1).astype("int8")
    df["knownPrefixLen"] = group.cumcount() + 1

    for col in SHOT_COLUMNS:
        df[f"prev_{col}"] = group[col].shift(1).fillna(-1).astype("int16")
        df[f"prev2_{col}"] = group[col].shift(2).fillna(-1).astype("int16")

    for col in ["actionId", "pointId", "strikeId"]:
        df[f"{col}_prefix_mean"] = group[col].expanding().mean().reset_index(level=0, drop=True)
        df[f"{col}_prefix_std"] = (
            group[col].expanding().std().reset_index(level=0, drop=True).fillna(0.0)
        )

    return df


def make_next_shot_samples(train: pd.DataFrame) -> pd.DataFrame:
    featured = add_prefix_features(train)
    group = featured.groupby("rally_uid", sort=False)
    featured["target_actionId"] = group["actionId"].shift(-1)
    featured["target_pointId"] = group["pointId"].shift(-1)
    featured["is_final_prefix"] = group.cumcount() == (group["rally_uid"].transform("size") - 2)
    samples = featured.dropna(subset=["target_actionId", "target_pointId"]).copy()
    samples["target_actionId"] = samples["target_actionId"].astype("int16")
    samples["target_pointId"] = samples["target_pointId"].astype("int16")
    samples["serverGetPoint"] = samples["serverGetPoint"].astype("int8")
    return samples


def make_test_samples(test: pd.DataFrame) -> pd.DataFrame:
    featured = add_prefix_features(test)
    return featured.groupby("rally_uid", sort=False).tail(1).copy()


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "rally_uid",
        "target_actionId",
        "target_pointId",
        "serverGetPoint",
        "is_final_prefix",
    }
    return [col for col in frame.columns if col not in excluded]


def score(action_true, action_pred, point_true, point_pred, rally_true, rally_pred) -> dict[str, float]:
    f1_action = f1_score(action_true, action_pred, average="macro")
    f1_point = f1_score(point_true, point_pred, average="macro")
    auc = roc_auc_score(rally_true, rally_pred) if len(set(rally_true)) > 1 else 0.5
    final = 0.4 * f1_action + 0.4 * f1_point + 0.2 * auc
    return {"f1_action": f1_action, "f1_point": f1_point, "auc": auc, "score": final}


def build_model(task: str, args):
    if args.model == "rf":
        return RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth if args.max_depth > 0 else None,
            min_samples_leaf=args.min_samples_leaf,
            class_weight="balanced_subsample",
            n_jobs=args.n_jobs,
            random_state=args.seed,
        )

    common = {
        "max_iter": args.max_iter,
        "learning_rate": args.learning_rate,
        "max_leaf_nodes": args.max_leaf_nodes,
        "l2_regularization": args.l2_regularization,
        "early_stopping": False,
        "random_state": args.seed,
    }
    if task in {"action", "point"}:
        common["class_weight"] = "balanced"
    return HistGradientBoostingClassifier(**common)


def train_and_predict(args, train: pd.DataFrame, test: pd.DataFrame):
    samples = make_next_shot_samples(train)
    test_samples = make_test_samples(test)
    features = feature_columns(samples)
    print(
        f"Train prefix samples={len(samples)} | test rallies={len(test_samples)} | "
        f"features={len(features)}",
        flush=True,
    )

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(samples, groups=samples["match"]))
    train_part = samples.iloc[train_idx]
    val_part = samples.iloc[val_idx]
    print(
        f"Train matches={train_part['match'].nunique()} | val matches={val_part['match'].nunique()}",
        flush=True,
    )

    x_train = train_part[features]
    x_val = val_part[features]
    x_test = test_samples[features]

    action_model = build_model("action", args)
    point_model = build_model("point", args)
    rally_model = build_model("rally", args)

    print("Training action model...", flush=True)
    action_model.fit(x_train, train_part["target_actionId"])
    print("Training point model...", flush=True)
    point_model.fit(x_train, train_part["target_pointId"])
    print("Training rally model...", flush=True)
    rally_model.fit(x_train, train_part["serverGetPoint"])

    val_final = val_part[val_part["is_final_prefix"]].copy()
    metrics = score(
        val_part["target_actionId"],
        action_model.predict(x_val),
        val_part["target_pointId"],
        point_model.predict(x_val),
        val_final["serverGetPoint"],
        rally_model.predict_proba(val_final[features])[:, 1],
    )
    print(
        f"Validation F1_action={metrics['f1_action']:.4f} "
        f"F1_point={metrics['f1_point']:.4f} AUC={metrics['auc']:.4f} "
        f"score={metrics['score']:.4f}",
        flush=True,
    )

    if args.refit_full:
        print("Refitting models on all prefix samples...", flush=True)
        action_model = build_model("action", args).fit(samples[features], samples["target_actionId"])
        point_model = build_model("point", args).fit(samples[features], samples["target_pointId"])
        rally_model = build_model("rally", args).fit(samples[features], samples["serverGetPoint"])

    submission = pd.DataFrame(
        {
            "rally_uid": test_samples["rally_uid"].astype(int).to_numpy(),
            "actionId": action_model.predict(x_test).astype(int),
            "pointId": point_model.predict(x_test).astype(int),
            "serverGetPoint": rally_model.predict_proba(x_test)[:, 1],
        },
        columns=SUBMISSION_COLUMNS,
    ).sort_values("rally_uid")
    return metrics, submission


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test.csv")
    parser.add_argument("--out", default="outputs/submission_tabular_baseline.csv")
    parser.add_argument("--model", choices=["hgb", "rf"], default="hgb")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-iter", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.01)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=18)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--no-refit-full", dest="refit_full", action="store_false")
    parser.set_defaults(refit_full=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print(f"Loading train data from {args.train}", flush=True)
    train = pd.read_csv(args.train)
    print(f"Loading test data from {args.test}", flush=True)
    test = pd.read_csv(args.test)
    metrics, submission = train_and_predict(args, train, test)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved submission to: {out_path}", flush=True)
    print(submission.head(), flush=True)


if __name__ == "__main__":
    main()
