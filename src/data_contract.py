from __future__ import annotations

from pathlib import Path

import pandas as pd


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
TRAIN_COLUMNS = ["rally_uid", *BASE_COLUMNS, "serverGetPoint"]
TEST_COLUMNS = ["rally_uid", *BASE_COLUMNS]


def default_test_path() -> str:
    return "test_new.csv" if Path("test_new.csv").exists() else "test.csv"


def validate_columns(frame: pd.DataFrame, required: list[str], path: str | Path) -> None:
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def sort_rally_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["rally_uid", "strikeNumber"], kind="mergesort").reset_index(drop=True)


def load_train_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_columns(frame, TRAIN_COLUMNS, path)
    return sort_rally_rows(frame)


def load_test_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_columns(frame, TEST_COLUMNS, path)
    return sort_rally_rows(frame)


def validate_submission(submission: pd.DataFrame, test: pd.DataFrame) -> None:
    validate_columns(submission, SUBMISSION_COLUMNS, "submission")
    expected = int(test["rally_uid"].nunique())
    if len(submission) != expected:
        raise ValueError(f"Submission rows={len(submission)} does not match test rallies={expected}.")
    if submission["rally_uid"].nunique() != expected:
        raise ValueError("Submission rally_uid values must be unique per test rally.")
