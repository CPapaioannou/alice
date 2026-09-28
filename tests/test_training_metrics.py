#!/usr/bin/env python3
"""Regression tests for Ultralytics training metric reporting."""

from pathlib import Path

from src.trainer import _read_training_metrics


def test_read_training_metrics_uses_best_map50_95_and_strips_headers(tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "epoch, metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B)\n"
        "1,0.80,0.90,0.95,0.70\n"
        "2,0.85,0.88,0.96,0.82\n"
        "3,0.90,0.86,0.97,0.79\n"
    )

    metrics = _read_training_metrics(str(csv_path))

    assert metrics == {
        "epoch": 2,
        "precision": 0.85,
        "recall": 0.88,
        "mAP50": 0.96,
        "mAP50_95": 0.82,
    }


def test_read_training_metrics_missing_file_returns_none(tmp_path):
    assert _read_training_metrics(str(tmp_path / "missing.csv")) is None
