from __future__ import annotations

import json

import pandas as pd

from error_analysis import analyze


def test_error_analysis_writes_reports(tmp_path):
    next_shot = pd.DataFrame(
        {
            "rally_uid": [1, 1, 2, 2],
            "match": [10, 10, 20, 20],
            "strikeNumber": [1, 2, 1, 2],
            "knownPrefixLen": [1, 2, 1, 2],
            "scoreDiff": [0, 0, 1, 1],
            "isLatePoint": [0, 0, 1, 1],
            "fold": [0, 0, 1, 1],
            "actionId_true": [1, 2, 1, 3],
            "pointId_true": [0, 1, 0, 2],
            "actionId_pred": [1, 1, 1, 2],
            "pointId_pred": [0, 1, 1, 2],
        }
    )
    rally = pd.DataFrame(
        {
            "rally_uid": [1, 2],
            "serverGetPoint_true": [0, 1],
            "fold": [0, 1],
            "serverGetPoint_pred": [0.2, 0.8],
        }
    )
    next_path = tmp_path / "oof_next_shot.csv"
    rally_path = tmp_path / "oof_rally.csv"
    out_dir = tmp_path / "analysis"
    next_shot.to_csv(next_path, index=False)
    rally.to_csv(rally_path, index=False)

    summary = analyze(next_path, rally_path, out_dir)

    assert summary["next_shot_rows"] == 4
    assert summary["rally"]["auc"] == 1.0
    assert (out_dir / "action_per_class_f1.csv").exists()
    assert (out_dir / "point_per_class_f1.csv").exists()
    assert (out_dir / "action_confusion_matrix.csv").exists()
    assert (out_dir / "point_confusion_matrix.csv").exists()
    assert (out_dir / "error_by_prefix_len.csv").exists()

    saved_summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert saved_summary["group_outputs"]["knownPrefixLen"] == "error_by_prefix_len.csv"
