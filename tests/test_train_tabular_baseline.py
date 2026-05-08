from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from data_contract import (
    SUBMISSION_COLUMNS,
    default_test_path,
    load_test_csv,
    load_train_csv,
    validate_submission,
)
from train_tabular_baseline import (
    build_group_splits,
    feature_columns,
    make_next_shot_samples,
    make_rally_samples,
    make_test_samples,
    train_and_predict,
)


def sample_match_frame() -> pd.DataFrame:
    rows = []
    rally_specs = [
        (101, 11, 1, 0),
        (102, 11, 0, 1),
        (201, 22, 1, 2),
        (202, 22, 0, 3),
        (301, 33, 1, 4),
        (302, 33, 0, 5),
    ]
    for rally_uid, match, server_get_point, offset in rally_specs:
        for strike_number in range(1, 4):
            rows.append(
                {
                    "rally_uid": rally_uid,
                    "sex": offset % 2,
                    "match": match,
                    "numberGame": 1 + offset % 3,
                    "rally_id": rally_uid * 10,
                    "strikeNumber": strike_number,
                    "scoreSelf": strike_number - 1 + int(server_get_point and strike_number == 3),
                    "scoreOther": int((not server_get_point) and strike_number == 3),
                    "gamePlayerId": 1000 + match,
                    "gamePlayerOtherId": 2000 + match,
                    "strikeId": strike_number + offset,
                    "handId": (strike_number + offset) % 2,
                    "strengthId": (strike_number + offset) % 3,
                    "spinId": (strike_number + 1 + offset) % 4,
                    "pointId": strike_number + offset,
                    "actionId": strike_number + offset + 1,
                    "positionId": (strike_number + offset) % 5,
                    "serverGetPoint": server_get_point,
                }
            )
    return pd.DataFrame(rows)


def base_args(**overrides):
    defaults = dict(
        model="rf",
        device="cpu",
        gpu_devices="0:1",
        val_size=0.34,
        cv_folds=1,
        seed=42,
        max_iter=20,
        learning_rate=0.1,
        max_leaf_nodes=16,
        num_leaves=16,
        l2_regularization=0.0,
        n_estimators=10,
        max_depth=4,
        cat_depth=4,
        min_samples_leaf=1,
        subsample=1.0,
        colsample_bytree=1.0,
        n_jobs=1,
        verbose_eval=False,
        metrics_out=None,
        val_predictions_out=None,
        save_oof=None,
        action_class_weights="none",
        point_class_weights="none",
        stack_point_action=False,
        refit_full=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_sample_builders_generate_expected_rows_and_features():
    train = sample_match_frame()

    next_shot = make_next_shot_samples(train)
    rally = make_rally_samples(train)
    test = make_test_samples(train)
    features = feature_columns(next_shot)

    assert len(next_shot) == train["rally_uid"].nunique() * 2
    assert len(rally) == train["rally_uid"].nunique()
    assert len(test) == train["rally_uid"].nunique()
    assert {"scoreDiff", "prev_actionId", "action_transition", "shot_context"} <= set(features)
    assert "target_actionId" not in features
    assert next_shot["target_actionId"].dtype.kind in {"i", "u"}
    assert rally["serverGetPoint"].dtype.kind in {"i", "u"}


def test_group_splits_keep_matches_disjoint_for_cv_and_single_split():
    rally = make_rally_samples(sample_match_frame())

    cv_splits = build_group_splits(base_args(cv_folds=3), rally)
    assert len(cv_splits) == 3
    for train_idx, val_idx in cv_splits:
        train_matches = set(rally.iloc[train_idx]["match"])
        val_matches = set(rally.iloc[val_idx]["match"])
        assert train_matches.isdisjoint(val_matches)

    single_split = build_group_splits(base_args(cv_folds=1, val_size=0.34), rally)
    assert len(single_split) == 1
    train_idx, val_idx = single_split[0]
    train_matches = set(rally.iloc[train_idx]["match"])
    val_matches = set(rally.iloc[val_idx]["match"])
    assert train_matches.isdisjoint(val_matches)


def test_train_and_predict_masks_single_split_oof_metrics():
    train = sample_match_frame()
    test = train.groupby("rally_uid", sort=False).tail(1).copy().reset_index(drop=True)

    metrics, submission = train_and_predict(base_args(cv_folds=1), train, test)

    assert metrics["oof_prefix_samples_scored"] < metrics["train_prefix_samples"]
    assert metrics["oof_rallies_scored"] < metrics["train_rallies"]
    assert metrics["oof_prefix_samples_scored"] > 0
    assert metrics["oof_rallies_scored"] > 0
    assert len(metrics["fold_metrics"]) == 1
    assert list(submission.columns) == ["rally_uid", "actionId", "pointId", "serverGetPoint"]
    assert len(submission) == test["rally_uid"].nunique()


def test_train_and_predict_oof_exports_diagnostic_columns(tmp_path):
    train = sample_match_frame()
    test = train.groupby("rally_uid", sort=False).tail(1).copy().reset_index(drop=True)
    prefix = tmp_path / "oof_smoke"

    train_and_predict(base_args(cv_folds=1, save_oof=str(prefix)), train, test)

    next_shot = pd.read_csv(tmp_path / "oof_smoke_next_shot.csv")
    assert {
        "match",
        "strikeNumber",
        "knownPrefixLen",
        "scoreDiff",
        "isLatePoint",
        "is_final_prefix",
    } <= set(next_shot.columns)


def test_data_contract_loads_sorts_and_validates_submission(tmp_path, monkeypatch):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test_new.csv"
    frame = sample_match_frame().sample(frac=1.0, random_state=7)
    frame.to_csv(train_path, index=False)
    frame.drop(columns=["serverGetPoint"]).to_csv(test_path, index=False)

    monkeypatch.chdir(tmp_path)

    assert default_test_path() == "test_new.csv"

    train = load_train_csv(train_path)
    test = load_test_csv(test_path)
    assert train[["rally_uid", "strikeNumber"]].equals(
        train[["rally_uid", "strikeNumber"]].sort_values(
            ["rally_uid", "strikeNumber"],
            kind="mergesort",
        ).reset_index(drop=True)
    )

    submission = pd.DataFrame(
        {
            "rally_uid": sorted(test["rally_uid"].unique()),
            "actionId": 1,
            "pointId": 1,
            "serverGetPoint": 0.5,
        },
        columns=SUBMISSION_COLUMNS,
    )
    validate_submission(submission, test)
