from __future__ import annotations

import argparse
from loguru import logger
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score


def sorted_labels(true: pd.Series, pred: pd.Series) -> list[int]:
    labels = pd.concat([true, pred], ignore_index=True).dropna().astype(int).unique().tolist()
    return sorted(labels)


def per_class_f1(frame: pd.DataFrame, true_col: str, pred_col: str) -> pd.DataFrame:
    labels = sorted_labels(frame[true_col], frame[pred_col])
    precision, recall, f1, support = precision_recall_fscore_support(
        frame[true_col],
        frame[pred_col],
        labels=labels,
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "class_id": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "pred_count": [(frame[pred_col] == label).sum() for label in labels],
        }
    ).sort_values(["f1", "support", "class_id"], ascending=[True, False, True])


def confusion(frame: pd.DataFrame, true_col: str, pred_col: str) -> pd.DataFrame:
    labels = sorted_labels(frame[true_col], frame[pred_col])
    matrix = confusion_matrix(frame[true_col], frame[pred_col], labels=labels)
    return pd.DataFrame(
        matrix,
        index=pd.Index(labels, name="true"),
        columns=pd.Index(labels, name="pred"),
    )


def error_by_group(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for value, group in frame.groupby(group_col, dropna=False, sort=True):
        rows.append(
            {
                group_col: value,
                "rows": int(len(group)),
                "action_accuracy": float(group["action_correct"].mean()),
                "point_accuracy": float(group["point_correct"].mean()),
                "both_accuracy": float((group["action_correct"] & group["point_correct"]).mean()),
            }
        )
    return pd.DataFrame(rows)


def rally_summary(frame: pd.DataFrame) -> dict[str, float]:
    result = {
        "rows": int(len(frame)),
        "server_get_point_mean_true": float(frame["serverGetPoint_true"].mean()),
        "server_get_point_mean_pred": float(frame["serverGetPoint_pred"].mean()),
    }
    if frame["serverGetPoint_true"].nunique() > 1:
        result["auc"] = float(roc_auc_score(frame["serverGetPoint_true"], frame["serverGetPoint_pred"]))
    else:
        result["auc"] = 0.5
    return result


def analyze(next_shot_oof: Path, rally_oof: Path | None, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    next_shot = pd.read_csv(next_shot_oof)
    next_shot = next_shot[(next_shot["actionId_pred"] >= 0) & (next_shot["pointId_pred"] >= 0)].copy()
    next_shot["action_correct"] = next_shot["actionId_true"] == next_shot["actionId_pred"]
    next_shot["point_correct"] = next_shot["pointId_true"] == next_shot["pointId_pred"]

    action_f1 = per_class_f1(next_shot, "actionId_true", "actionId_pred")
    point_f1 = per_class_f1(next_shot, "pointId_true", "pointId_pred")
    action_confusion = confusion(next_shot, "actionId_true", "actionId_pred")
    point_confusion = confusion(next_shot, "pointId_true", "pointId_pred")

    action_f1.to_csv(out_dir / "action_per_class_f1.csv", index=False)
    point_f1.to_csv(out_dir / "point_per_class_f1.csv", index=False)
    action_confusion.to_csv(out_dir / "action_confusion_matrix.csv")
    point_confusion.to_csv(out_dir / "point_confusion_matrix.csv")

    group_outputs = {}
    for column, filename in [
        ("fold", "error_by_fold.csv"),
        ("strikeNumber", "error_by_strike_number.csv"),
        ("knownPrefixLen", "error_by_prefix_len.csv"),
        ("scoreDiff", "error_by_score_diff.csv"),
        ("isLatePoint", "error_by_late_point.csv"),
    ]:
        if column in next_shot:
            grouped = error_by_group(next_shot, column)
            grouped.to_csv(out_dir / filename, index=False)
            group_outputs[column] = filename

    summary = {
        "next_shot_rows": int(len(next_shot)),
        "action_accuracy": float(next_shot["action_correct"].mean()),
        "point_accuracy": float(next_shot["point_correct"].mean()),
        "action_zero_f1_classes": action_f1.loc[action_f1["f1"] == 0, "class_id"].astype(int).tolist(),
        "point_zero_f1_classes": point_f1.loc[point_f1["f1"] == 0, "class_id"].astype(int).tolist(),
        "group_outputs": group_outputs,
    }

    if rally_oof:
        rally = pd.read_csv(rally_oof)
        rally = rally[rally["serverGetPoint_pred"].notna()].copy()
        summary["rally"] = rally_summary(rally)

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--next-shot-oof", required=True)
    parser.add_argument("--rally-oof")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = analyze(
        Path(args.next_shot_oof),
        Path(args.rally_oof) if args.rally_oof else None,
        Path(args.out_dir),
    )
    logger.info(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
