from __future__ import annotations

import argparse
import json
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
    df["absScoreDiff"] = df["scoreDiff"].abs()
    df["scoreSum"] = df["scoreSelf"] + df["scoreOther"]
    df["isTied"] = (df["scoreDiff"] == 0).astype("int8")
    df["isLeading"] = (df["scoreDiff"] > 0).astype("int8")
    df["isTrailing"] = (df["scoreDiff"] < 0).astype("int8")
    df["isLatePoint"] = (df[["scoreSelf", "scoreOther"]].max(axis=1) >= 8).astype("int8")
    df["isDeuceOrAd"] = (
        (df["scoreSelf"] >= 10) & (df["scoreOther"] >= 10) & (df["absScoreDiff"] <= 1)
    ).astype("int8")
    df["isServerStroke"] = (df["strikeNumber"] % 2 == 1).astype("int8")
    df["knownPrefixLen"] = group.cumcount() + 1

    for col in SHOT_COLUMNS:
        df[f"prev_{col}"] = group[col].shift(1).fillna(-1).astype("int16")
        df[f"prev2_{col}"] = group[col].shift(2).fillna(-1).astype("int16")
        df[f"prev3_{col}"] = group[col].shift(3).fillna(-1).astype("int16")
        df[f"{col}_prefix_value_count"] = df.groupby(["rally_uid", col], sort=False).cumcount() + 1
        df[f"{col}_prefix_value_rate"] = df[f"{col}_prefix_value_count"] / df["knownPrefixLen"]
        df[f"{col}_same_as_prev"] = (df[col] == df[f"prev_{col}"]).astype("int8")

    for col in ["actionId", "pointId", "strikeId"]:
        df[f"{col}_prefix_mean"] = group[col].expanding().mean().reset_index(level=0, drop=True)
        df[f"{col}_prefix_std"] = (
            group[col].expanding().std().reset_index(level=0, drop=True).fillna(0.0)
        )
        df[f"{col}_prefix_min"] = group[col].expanding().min().reset_index(level=0, drop=True)
        df[f"{col}_prefix_max"] = group[col].expanding().max().reset_index(level=0, drop=True)

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


def make_rally_samples(train: pd.DataFrame) -> pd.DataFrame:
    featured = add_prefix_features(train)
    group = featured.groupby("rally_uid", sort=False)
    featured["is_final_prefix"] = group.cumcount() == (group["rally_uid"].transform("size") - 2)
    samples = featured[featured["is_final_prefix"]].copy()
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


def prediction_frame(
    frame: pd.DataFrame,
    action_pred,
    point_pred,
    rally_pred,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rally_uid": frame["rally_uid"].astype(int).to_numpy(),
            "actionId": np.asarray(action_pred).ravel().astype(int),
            "pointId": np.asarray(point_pred).ravel().astype(int),
            "serverGetPoint": np.asarray(rally_pred).ravel().astype(float),
        },
        columns=SUBMISSION_COLUMNS,
    ).sort_values("rally_uid")


def add_action_hint(x: pd.DataFrame, action_id) -> pd.DataFrame:
    x = x.copy()
    x["next_actionId_hint"] = np.asarray(action_id).ravel().astype("int16")
    return x


def save_validation_predictions(
    out_prefix: str,
    next_shot: pd.DataFrame,
    rally: pd.DataFrame,
) -> None:
    prefix = Path(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    next_shot_path = prefix.with_name(f"{prefix.stem}_next_shot.csv")
    rally_path = prefix.with_name(f"{prefix.stem}_rally.csv")
    next_shot.to_csv(next_shot_path, index=False, encoding="utf-8")
    rally.to_csv(rally_path, index=False, encoding="utf-8")
    print(
        f"Saved validation predictions to: {next_shot_path} and {rally_path}",
        flush=True,
    )


def score(action_true, action_pred, point_true, point_pred, rally_true, rally_pred) -> dict[str, float]:
    f1_action = f1_score(action_true, action_pred, average="macro")
    f1_point = f1_score(point_true, point_pred, average="macro")
    auc = roc_auc_score(rally_true, rally_pred) if len(set(rally_true)) > 1 else 0.5
    final = 0.4 * f1_action + 0.4 * f1_point + 0.2 * auc
    return {"f1_action": f1_action, "f1_point": f1_point, "auc": auc, "score": final}


def build_model(task: str, args):
    if args.model == "lightgbm":
        from lightgbm import LGBMClassifier

        objective = "binary" if task == "rally" else "multiclass"
        params = {
            "objective": objective,
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "num_leaves": args.num_leaves,
            "max_depth": args.max_depth if args.max_depth > 0 else -1,
            "min_child_samples": args.min_samples_leaf,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "reg_lambda": args.l2_regularization,
            "class_weight": "balanced" if task in {"action", "point"} else None,
            "random_state": args.seed,
            "n_jobs": args.n_jobs,
            "verbosity": -1,
        }
        if args.device == "gpu":
            params["device_type"] = "gpu"
        return LGBMClassifier(**params)

    if args.model == "catboost":
        from catboost import CatBoostClassifier

        params = {
            "iterations": args.n_estimators,
            "learning_rate": args.learning_rate,
            "depth": args.cat_depth,
            "l2_leaf_reg": max(args.l2_regularization, 1e-8),
            "loss_function": "Logloss" if task == "rally" else "MultiClass",
            "auto_class_weights": "Balanced" if task in {"action", "point"} else None,
            "random_seed": args.seed,
            "thread_count": args.n_jobs,
            "verbose": args.verbose_eval,
            "allow_writing_files": False,
        }
        if args.device == "gpu":
            params["task_type"] = "GPU"
            params["devices"] = args.gpu_devices
        return CatBoostClassifier(**params)

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
    rally_samples = make_rally_samples(train)
    test_samples = make_test_samples(test)
    features = feature_columns(samples)
    rally_features = feature_columns(rally_samples)
    print(
        f"Train prefix samples={len(samples)} | test rallies={len(test_samples)} | "
        f"features={len(features)} | rally samples={len(rally_samples)}",
        flush=True,
    )

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed)
    train_rally_idx, val_rally_idx = next(splitter.split(rally_samples, groups=rally_samples["match"]))
    train_matches = set(rally_samples.iloc[train_rally_idx]["match"].unique())
    val_matches = set(rally_samples.iloc[val_rally_idx]["match"].unique())

    train_part = samples[samples["match"].isin(train_matches)].copy()
    val_part = samples[samples["match"].isin(val_matches)].copy()
    train_rally = rally_samples.iloc[train_rally_idx].copy()
    val_rally = rally_samples.iloc[val_rally_idx].copy()
    print(
        f"Train matches={len(train_matches)} | val matches={len(val_matches)} | "
        f"val rallies={len(val_rally)}",
        flush=True,
    )

    x_train = train_part[features]
    x_val = val_part[features]
    x_test = test_samples[features]
    x_rally_train = train_rally[rally_features]
    x_rally_val = val_rally[rally_features]
    x_rally_test = test_samples[rally_features]

    action_model = build_model("action", args)
    point_model = build_model("point", args)
    rally_model = build_model("rally", args)

    print("Training action model...", flush=True)
    action_model.fit(x_train, train_part["target_actionId"])
    val_action_pred = action_model.predict(x_val)
    test_action_pred = action_model.predict(x_test)

    if args.stack_point_action:
        point_x_train = add_action_hint(x_train, train_part["target_actionId"])
        point_x_val = add_action_hint(x_val, val_action_pred)
        point_x_test = add_action_hint(x_test, test_action_pred)
    else:
        point_x_train = x_train
        point_x_val = x_val
        point_x_test = x_test

    print("Training point model...", flush=True)
    point_model.fit(point_x_train, train_part["target_pointId"])
    print("Training rally model on last known prefix per rally...", flush=True)
    rally_model.fit(x_rally_train, train_rally["serverGetPoint"])

    val_point_pred = point_model.predict(point_x_val)
    val_rally_pred = rally_model.predict_proba(x_rally_val)[:, 1]
    metrics = score(
        val_part["target_actionId"],
        val_action_pred,
        val_part["target_pointId"],
        val_point_pred,
        val_rally["serverGetPoint"],
        val_rally_pred,
    )
    metrics.update(
        {
            "model": args.model,
            "seed": args.seed,
            "val_size": args.val_size,
            "train_prefix_samples": int(len(train_part)),
            "val_prefix_samples": int(len(val_part)),
            "train_rallies": int(len(train_rally)),
            "val_rallies": int(len(val_rally)),
            "features": int(len(features)),
            "stack_point_action": bool(args.stack_point_action),
        }
    )
    print(
        f"Validation F1_action={metrics['f1_action']:.4f} "
        f"F1_point={metrics['f1_point']:.4f} AUC={metrics['auc']:.4f} "
        f"score={metrics['score']:.4f}",
        flush=True,
    )

    if args.val_predictions_out:
        val_action_frame = val_part[["rally_uid"]].copy()
        val_action_frame["actionId_true"] = val_part["target_actionId"].astype(int).to_numpy()
        val_action_frame["pointId_true"] = val_part["target_pointId"].astype(int).to_numpy()
        val_action_frame["actionId_pred"] = np.asarray(val_action_pred).ravel().astype(int)
        val_action_frame["pointId_pred"] = np.asarray(val_point_pred).ravel().astype(int)

        val_rally_frame = val_rally[["rally_uid", "serverGetPoint"]].rename(
            columns={"serverGetPoint": "serverGetPoint_true"}
        )
        val_rally_frame["serverGetPoint_pred"] = val_rally_pred

        save_validation_predictions(args.val_predictions_out, val_action_frame, val_rally_frame)

    if args.refit_full:
        print("Refitting models on all prefix samples...", flush=True)
        action_model = build_model("action", args).fit(samples[features], samples["target_actionId"])
        full_x = samples[features]
        full_test_x = test_samples[features]
        test_action_pred = action_model.predict(full_test_x)
        if args.stack_point_action:
            point_full_x = add_action_hint(full_x, samples["target_actionId"])
            point_test_x = add_action_hint(full_test_x, test_action_pred)
        else:
            point_full_x = full_x
            point_test_x = full_test_x
        point_model = build_model("point", args).fit(point_full_x, samples["target_pointId"])
        rally_model = build_model("rally", args).fit(
            rally_samples[rally_features],
            rally_samples["serverGetPoint"],
        )
    else:
        point_test_x = point_x_test

    submission = prediction_frame(
        test_samples,
        test_action_pred,
        point_model.predict(point_test_x),
        rally_model.predict_proba(x_rally_test)[:, 1],
    )
    return metrics, submission


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test.csv")
    parser.add_argument("--out", default="outputs/submission_tabular_baseline.csv")
    parser.add_argument("--model", choices=["hgb", "rf", "lightgbm", "catboost"], default="lightgbm")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--gpu-devices", default="0:1")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-iter", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--l2-regularization", type=float, default=0.01)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=18)
    parser.add_argument("--cat-depth", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--verbose-eval", type=int, default=False)
    parser.add_argument("--metrics-out")
    parser.add_argument("--val-predictions-out")
    parser.add_argument("--no-stack-point-action", dest="stack_point_action", action="store_false")
    parser.add_argument("--no-refit-full", dest="refit_full", action="store_false")
    parser.set_defaults(refit_full=True, stack_point_action=True)
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
    if args.metrics_out:
        metrics_path = Path(args.metrics_out)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Saved metrics to: {metrics_path}", flush=True)
    print(f"Saved submission to: {out_path}", flush=True)
    print(submission.head(), flush=True)


if __name__ == "__main__":
    main()
