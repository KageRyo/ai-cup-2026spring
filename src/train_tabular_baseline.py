from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from data_contract import (
    SHOT_COLUMNS,
    SUBMISSION_COLUMNS,
    default_test_path,
    load_test_csv,
    load_train_csv,
    validate_submission,
)

SEED = 42
ACTION_TRANSITION_CONTEXTS = [
    ("actionId",),
    ("strikeId", "actionId"),
    ("strikeId", "actionId", "spinId"),
    ("prev_actionId", "actionId"),
    ("handId", "spinId", "positionId"),
]
POINT_TRANSITION_CONTEXTS = [
    ("pointId",),
    ("actionId", "pointId"),
    ("strikeId", "pointId"),
    ("actionId", "spinId", "pointId"),
    ("prev_pointId", "pointId"),
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

    df["action_transition"] = df["prev_actionId"] * 32 + df["actionId"]
    df["point_transition"] = df["prev_pointId"] * 16 + df["pointId"]
    df["spin_strength"] = df["spinId"] * 8 + df["strengthId"]
    df["hand_position"] = df["handId"] * 8 + df["positionId"]
    df["shot_context"] = df["strikeId"] * 128 + df["spinId"] * 16 + df["positionId"]

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
    return samples.reset_index(drop=True)


def make_rally_samples(train: pd.DataFrame) -> pd.DataFrame:
    featured = add_prefix_features(train)
    group = featured.groupby("rally_uid", sort=False)
    featured["is_final_prefix"] = group.cumcount() == (group["rally_uid"].transform("size") - 2)
    samples = featured[featured["is_final_prefix"]].copy()
    samples["serverGetPoint"] = samples["serverGetPoint"].astype("int8")
    return samples.reset_index(drop=True)


def make_test_samples(test: pd.DataFrame) -> pd.DataFrame:
    featured = add_prefix_features(test)
    return featured.groupby("rally_uid", sort=False).tail(1).copy().reset_index(drop=True)


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


def empirical_feature_names() -> list[str]:
    names = []
    for target_prefix, contexts in [
        ("emp_action", ACTION_TRANSITION_CONTEXTS),
        ("emp_point", POINT_TRANSITION_CONTEXTS),
    ]:
        for idx, _ in enumerate(contexts):
            base = f"{target_prefix}_ctx{idx}"
            names.extend([f"{base}_pred", f"{base}_conf", f"{base}_count", f"{base}_entropy"])
    return names


def context_key(row: pd.Series, columns: tuple[str, ...]) -> tuple:
    return tuple(row[col] for col in columns)


def distribution_summary(counts: dict[int, int]) -> tuple[int, float, int, float]:
    total = int(sum(counts.values()))
    if total <= 0:
        return -1, 0.0, 0, 0.0
    pred, best_count = max(counts.items(), key=lambda item: (item[1], -item[0]))
    probs = np.asarray(list(counts.values()), dtype=np.float64) / total
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    return int(pred), float(best_count / total), total, entropy


def build_context_counts(
    stats_frame: pd.DataFrame,
    target: str,
    contexts: list[tuple[str, ...]],
) -> list[dict[tuple, dict[int, int]]]:
    context_counts = []
    for context in contexts:
        counts: dict[tuple, dict[int, int]] = {}
        grouped = stats_frame.groupby(list(context) + [target], sort=False).size()
        for key, value in grouped.items():
            if not isinstance(key, tuple):
                key = (key,)
            context_key_value = key[: len(context)]
            label = int(key[-1])
            counts.setdefault(context_key_value, {})[label] = int(value)
        context_counts.append(counts)
    return context_counts


def add_empirical_target_features(
    frame: pd.DataFrame,
    stats_frame: pd.DataFrame,
    target: str,
    contexts: list[tuple[str, ...]],
    prefix: str,
    leave_one_out: bool,
) -> pd.DataFrame:
    out = frame.copy()
    context_counts = build_context_counts(stats_frame, target, contexts)
    global_counts = stats_frame[target].value_counts().astype(int).to_dict()

    for idx, context in enumerate(contexts):
        pred_values, conf_values, count_values, entropy_values = [], [], [], []
        counts_by_key = context_counts[idx]

        for row_idx, row in frame.iterrows():
            key = context_key(row, context)
            counts = dict(counts_by_key.get(key, global_counts))
            if leave_one_out and target in frame.columns:
                label = int(frame.at[row_idx, target])
                if label in counts:
                    counts[label] -= 1
                    if counts[label] <= 0:
                        del counts[label]
            pred, conf, count, entropy = distribution_summary(counts)
            pred_values.append(pred)
            conf_values.append(conf)
            count_values.append(count)
            entropy_values.append(entropy)

        base = f"{prefix}_ctx{idx}"
        out[f"{base}_pred"] = np.asarray(pred_values, dtype=np.int16)
        out[f"{base}_conf"] = np.asarray(conf_values, dtype=np.float32)
        out[f"{base}_count"] = np.log1p(np.asarray(count_values, dtype=np.float32))
        out[f"{base}_entropy"] = np.asarray(entropy_values, dtype=np.float32)

    return out


def add_empirical_next_shot_features(
    frame: pd.DataFrame,
    stats_frame: pd.DataFrame,
    leave_one_out: bool = False,
) -> pd.DataFrame:
    out = add_empirical_target_features(
        frame,
        stats_frame,
        target="target_actionId",
        contexts=ACTION_TRANSITION_CONTEXTS,
        prefix="emp_action",
        leave_one_out=leave_one_out,
    )
    return add_empirical_target_features(
        out,
        stats_frame,
        target="target_pointId",
        contexts=POINT_TRANSITION_CONTEXTS,
        prefix="emp_point",
        leave_one_out=leave_one_out,
    )


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


def save_model_artifact(model, path: Path) -> None:
    if hasattr(model, "save_model"):
        model.save_model(path)
    else:
        joblib.dump(model, path)


def save_final_models(
    out_dir: str,
    models: dict[str, object],
    metadata: dict,
) -> None:
    model_dir = Path(out_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        suffix = ".cbm" if hasattr(model, "save_model") else ".joblib"
        save_model_artifact(model, model_dir / f"{name}{suffix}")
    (model_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved final models to: {model_dir}", flush=True)


def aligned_predict_proba(model, frame: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    proba = np.asarray(model.predict_proba(frame))
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])
    aligned = np.zeros((len(frame), len(labels)), dtype=np.float64)
    model_classes = np.asarray(model.classes_)
    for idx, label in enumerate(labels):
        matches = np.where(model_classes == label)[0]
        if matches.size:
            aligned[:, idx] = proba[:, matches[0]]
    return aligned


def predict_from_proba(labels: np.ndarray, proba: np.ndarray) -> np.ndarray:
    return labels[np.argmax(proba, axis=1)]


def build_group_splits(args, rally_samples: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = rally_samples["match"].to_numpy()
    unique_groups = np.unique(groups)
    if args.cv_folds > 1:
        if len(unique_groups) < args.cv_folds:
            raise ValueError(
                f"Requested cv_folds={args.cv_folds}, but only {len(unique_groups)} unique matches exist."
            )
        splitter = GroupKFold(n_splits=args.cv_folds)
        return list(splitter.split(rally_samples, groups=groups))

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed)
    return list(splitter.split(rally_samples, groups=groups))


def summarize_fold_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for key in ["f1_action", "f1_point", "auc", "score"]:
        values = np.asarray([fold[key] for fold in fold_metrics], dtype=np.float64)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=0))
    return summary


def score(action_true, action_pred, point_true, point_pred, rally_true, rally_pred) -> dict[str, float]:
    f1_action = f1_score(action_true, action_pred, average="macro")
    f1_point = f1_score(point_true, point_pred, average="macro")
    auc = roc_auc_score(rally_true, rally_pred) if len(set(rally_true)) > 1 else 0.5
    final = 0.4 * f1_action + 0.4 * f1_point + 0.2 * auc
    return {"f1_action": f1_action, "f1_point": f1_point, "auc": auc, "score": final}


def use_class_weights(task: str, args) -> bool:
    if task == "action":
        return args.action_class_weights == "balanced"
    if task == "point":
        return args.point_class_weights == "balanced"
    return False


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
            "class_weight": "balanced" if use_class_weights(task, args) else None,
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
            "auto_class_weights": "Balanced" if use_class_weights(task, args) else None,
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
            class_weight="balanced_subsample" if use_class_weights(task, args) else None,
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
    if use_class_weights(task, args):
        common["class_weight"] = "balanced"
    return HistGradientBoostingClassifier(**common)


def train_and_predict(args, train: pd.DataFrame, test: pd.DataFrame):
    samples = make_next_shot_samples(train)
    rally_samples = make_rally_samples(train)
    test_samples = make_test_samples(test)
    base_features = feature_columns(samples)
    base_rally_features = feature_columns(rally_samples)
    transition_features = empirical_feature_names() if args.transition_features else []
    features = base_features + transition_features
    rally_features = base_rally_features + transition_features
    print(
        f"Train prefix samples={len(samples)} | test rallies={len(test_samples)} | "
        f"features={len(features)} | rally samples={len(rally_samples)}",
        flush=True,
    )

    splits = build_group_splits(args, rally_samples)
    action_labels = np.sort(samples["target_actionId"].unique())
    point_labels = np.sort(samples["target_pointId"].unique())
    rally_labels = np.array([0, 1], dtype=np.int8)

    oof_action_pred = np.full(len(samples), -1, dtype=np.int32)
    oof_point_pred = np.full(len(samples), -1, dtype=np.int32)
    oof_rally_pred = np.full(len(rally_samples), np.nan, dtype=np.float64)
    oof_fold_id = np.full(len(samples), -1, dtype=np.int16)
    oof_rally_fold_id = np.full(len(rally_samples), -1, dtype=np.int16)
    fold_metrics: list[dict[str, float]] = []

    test_action_proba_sum = np.zeros((len(test_samples), len(action_labels)), dtype=np.float64)
    test_point_proba_sum = np.zeros((len(test_samples), len(point_labels)), dtype=np.float64)
    test_rally_proba_sum = np.zeros((len(test_samples), len(rally_labels)), dtype=np.float64)

    for fold_id, (train_rally_idx, val_rally_idx) in enumerate(splits):
        train_matches = set(rally_samples.iloc[train_rally_idx]["match"].unique())
        val_matches = set(rally_samples.iloc[val_rally_idx]["match"].unique())

        train_part = samples[samples["match"].isin(train_matches)].copy()
        val_part = samples[samples["match"].isin(val_matches)].copy()
        train_rally = rally_samples.iloc[train_rally_idx].copy()
        val_rally = rally_samples.iloc[val_rally_idx].copy()
        fold_test_samples = test_samples
        if args.transition_features:
            train_part = add_empirical_next_shot_features(
                train_part,
                train_part,
                leave_one_out=True,
            )
            val_part = add_empirical_next_shot_features(val_part, train_part)
            train_rally = add_empirical_next_shot_features(train_rally, train_part)
            val_rally = add_empirical_next_shot_features(val_rally, train_part)
            fold_test_samples = add_empirical_next_shot_features(test_samples, train_part)
        print(
            f"Fold {fold_id}: train matches={len(train_matches)} | val matches={len(val_matches)} | "
            f"train prefixes={len(train_part)} | val prefixes={len(val_part)} | "
            f"val rallies={len(val_rally)}",
            flush=True,
        )

        x_train = train_part[features]
        x_val = val_part[features]
        x_rally_train = train_rally[rally_features]
        x_rally_val = val_rally[rally_features]
        x_test = fold_test_samples[features]
        x_rally_test = fold_test_samples[rally_features]

        action_model = build_model("action", args)
        point_model = build_model("point", args)
        rally_model = build_model("rally", args)

        print(f"Fold {fold_id}: training action model...", flush=True)
        action_model.fit(x_train, train_part["target_actionId"])
        val_action_pred = np.asarray(action_model.predict(x_val)).ravel().astype(np.int32)
        test_action_proba = aligned_predict_proba(action_model, x_test, action_labels)
        test_action_proba_sum += test_action_proba
        test_action_pred = predict_from_proba(action_labels, test_action_proba)

        if args.stack_point_action:
            point_x_train = add_action_hint(x_train, train_part["target_actionId"])
            point_x_val = add_action_hint(x_val, val_action_pred)
            point_x_test = add_action_hint(x_test, test_action_pred)
        else:
            point_x_train = x_train
            point_x_val = x_val
            point_x_test = x_test

        print(f"Fold {fold_id}: training point model...", flush=True)
        point_model.fit(point_x_train, train_part["target_pointId"])
        val_point_pred = np.asarray(point_model.predict(point_x_val)).ravel().astype(np.int32)
        test_point_proba_sum += aligned_predict_proba(point_model, point_x_test, point_labels)

        print(f"Fold {fold_id}: training rally model on last known prefix per rally...", flush=True)
        rally_model.fit(x_rally_train, train_rally["serverGetPoint"])
        val_rally_proba = aligned_predict_proba(rally_model, x_rally_val, rally_labels)[:, 1]
        test_rally_proba_sum += aligned_predict_proba(rally_model, x_rally_test, rally_labels)

        fold_metric = score(
            val_part["target_actionId"],
            val_action_pred,
            val_part["target_pointId"],
            val_point_pred,
            val_rally["serverGetPoint"],
            val_rally_proba,
        )
        fold_metric.update(
            {
                "fold": fold_id,
                "train_matches": int(len(train_matches)),
                "val_matches": int(len(val_matches)),
                "train_prefix_samples": int(len(train_part)),
                "val_prefix_samples": int(len(val_part)),
                "train_rallies": int(len(train_rally)),
                "val_rallies": int(len(val_rally)),
            }
        )
        fold_metrics.append(fold_metric)
        print(
            f"Fold {fold_id}: F1_action={fold_metric['f1_action']:.4f} "
            f"F1_point={fold_metric['f1_point']:.4f} AUC={fold_metric['auc']:.4f} "
            f"score={fold_metric['score']:.4f}",
            flush=True,
        )

        oof_action_pred[val_part.index.to_numpy()] = val_action_pred
        oof_point_pred[val_part.index.to_numpy()] = val_point_pred
        oof_rally_pred[val_rally.index.to_numpy()] = val_rally_proba
        oof_fold_id[val_part.index.to_numpy()] = fold_id
        oof_rally_fold_id[val_rally.index.to_numpy()] = fold_id

    next_mask = oof_fold_id >= 0
    rally_mask = oof_rally_fold_id >= 0
    if not next_mask.any() or not rally_mask.any():
        raise ValueError("No validation predictions were generated for OOF scoring.")

    metrics = score(
        samples.loc[next_mask, "target_actionId"],
        oof_action_pred[next_mask],
        samples.loc[next_mask, "target_pointId"],
        oof_point_pred[next_mask],
        rally_samples.loc[rally_mask, "serverGetPoint"],
        oof_rally_pred[rally_mask],
    )
    metrics.update(
        {
            "model": args.model,
            "seed": args.seed,
            "val_size": args.val_size,
            "cv_folds": int(args.cv_folds),
            "cv_strategy": "GroupKFold" if args.cv_folds > 1 else "GroupShuffleSplit",
            "train_prefix_samples": int(len(samples)),
            "train_rallies": int(len(rally_samples)),
            "features": int(len(features)),
            "stack_point_action": bool(args.stack_point_action),
            "action_class_weights": args.action_class_weights,
            "point_class_weights": args.point_class_weights,
            "fold_metrics": fold_metrics,
            "oof_prefix_samples_scored": int(next_mask.sum()),
            "oof_rallies_scored": int(rally_mask.sum()),
        }
    )
    metrics.update(summarize_fold_metrics(fold_metrics))
    print(
        f"OOF F1_action={metrics['f1_action']:.4f} "
        f"F1_point={metrics['f1_point']:.4f} AUC={metrics['auc']:.4f} "
        f"score={metrics['score']:.4f}",
        flush=True,
    )
    if args.cv_folds > 1:
        print(
            f"CV mean score={metrics['score_mean']:.4f} std={metrics['score_std']:.4f}",
            flush=True,
        )

    if args.val_predictions_out or args.save_oof:
        oof_prefix = args.save_oof or args.val_predictions_out
        diagnostic_cols = [
            "rally_uid",
            "match",
            "strikeNumber",
            "knownPrefixLen",
            "scoreDiff",
            "isLatePoint",
            "is_final_prefix",
        ]
        val_action_frame = samples[diagnostic_cols].copy()
        val_action_frame["fold"] = oof_fold_id
        val_action_frame["actionId_true"] = samples["target_actionId"].astype(int).to_numpy()
        val_action_frame["pointId_true"] = samples["target_pointId"].astype(int).to_numpy()
        val_action_frame["actionId_pred"] = oof_action_pred.astype(int)
        val_action_frame["pointId_pred"] = oof_point_pred.astype(int)

        val_rally_frame = rally_samples[["rally_uid", "serverGetPoint"]].rename(
            columns={"serverGetPoint": "serverGetPoint_true"}
        )
        val_rally_frame["fold"] = oof_rally_fold_id
        val_rally_frame["serverGetPoint_pred"] = oof_rally_pred

        save_validation_predictions(oof_prefix, val_action_frame, val_rally_frame)

    if args.refit_full:
        print("Refitting models on all prefix samples...", flush=True)
        full_samples = samples
        full_rally_samples = rally_samples
        full_test_samples = test_samples
        if args.transition_features:
            full_samples = add_empirical_next_shot_features(samples, samples, leave_one_out=True)
            full_rally_samples = add_empirical_next_shot_features(rally_samples, samples)
            full_test_samples = add_empirical_next_shot_features(test_samples, samples)

        action_model = build_model("action", args).fit(
            full_samples[features],
            full_samples["target_actionId"],
        )
        full_x = full_samples[features]
        full_test_x = full_test_samples[features]
        test_action_pred = action_model.predict(full_test_x)
        if args.stack_point_action:
            point_full_x = add_action_hint(full_x, full_samples["target_actionId"])
            point_test_x = add_action_hint(full_test_x, test_action_pred)
        else:
            point_full_x = full_x
            point_test_x = full_test_x
        point_model = build_model("point", args).fit(point_full_x, full_samples["target_pointId"])
        rally_model = build_model("rally", args).fit(
            full_rally_samples[rally_features],
            full_rally_samples["serverGetPoint"],
        )
        test_point_pred = point_model.predict(point_test_x)
        test_rally_pred = rally_model.predict_proba(full_test_samples[rally_features])[:, 1]
        if args.models_out_dir:
            save_final_models(
                args.models_out_dir,
                {
                    "action": action_model,
                    "point": point_model,
                    "rally": rally_model,
                },
                {
                    "model": args.model,
                    "seed": args.seed,
                    "features": features,
                    "rally_features": rally_features,
                    "action_labels": action_labels.astype(int).tolist(),
                    "point_labels": point_labels.astype(int).tolist(),
                    "stack_point_action": bool(args.stack_point_action),
                    "transition_features": bool(args.transition_features),
                    "refit_full": True,
                },
            )
    else:
        if args.models_out_dir:
            print("Skipping model save because --no-refit-full does not produce final models.", flush=True)
        test_action_pred = predict_from_proba(action_labels, test_action_proba_sum)
        test_point_pred = predict_from_proba(point_labels, test_point_proba_sum)
        test_rally_pred = test_rally_proba_sum[:, 1] / max(len(splits), 1)

    submission = prediction_frame(
        test_samples,
        test_action_pred,
        test_point_pred,
        test_rally_pred,
    )
    return metrics, submission


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default=default_test_path())
    parser.add_argument("--out", default="outputs/submission_tabular_baseline.csv")
    parser.add_argument("--model", choices=["hgb", "rf", "lightgbm", "catboost"], default="lightgbm")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--gpu-devices", default="0:1")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--cv-folds", type=int, default=1)
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
    parser.add_argument("--save-oof")
    parser.add_argument("--models-out-dir")
    parser.add_argument("--transition-features", action="store_true")
    parser.add_argument("--action-class-weights", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--point-class-weights", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--no-stack-point-action", dest="stack_point_action", action="store_false")
    parser.add_argument("--no-refit-full", dest="refit_full", action="store_false")
    parser.set_defaults(refit_full=True, stack_point_action=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print(f"Loading train data from {args.train}", flush=True)
    train = load_train_csv(args.train)
    print(f"Loading test data from {args.test}", flush=True)
    test = load_test_csv(args.test)
    metrics, submission = train_and_predict(args, train, test)
    validate_submission(submission, test)

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
