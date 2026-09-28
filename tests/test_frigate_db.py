#!/usr/bin/env python3
"""Tests for local and SSH-backed Frigate database access."""

import sqlite3
import subprocess

import pytest

from src import header
from src import trainer


@pytest.fixture(autouse=True)
def restore_conf():
    old = dict(header.CONF)
    yield
    header.CONF.clear()
    header.CONF.update(old)


def test_frigate_event_rows_local_read_only(tmp_path):
    db = tmp_path / "frigate.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE event (id TEXT PRIMARY KEY, camera TEXT, has_snapshot INTEGER, start_time REAL)"
    )
    conn.executemany(
        "INSERT INTO event (id, camera, has_snapshot, start_time) VALUES (?, ?, ?, ?)",
        [
            ("older", "front", 1, 10.0),
            ("ignored", "front", 0, 20.0),
            ("newer", "back", 1, 30.0),
        ],
    )
    conn.commit()
    conn.close()

    header.CONF.clear()
    header.CONF.update({
        "FRIGATE_DB": str(db),
        "FRIGATE_DB_SSH_HOST": "",
    })

    assert trainer._frigate_event_rows() == [
        ("newer", "back"),
        ("older", "front"),
    ]


def test_frigate_event_rows_over_ssh(monkeypatch):
    header.CONF.clear()
    header.CONF.update({
        "FRIGATE_DB": "/config/frigate.db",
        "FRIGATE_DB_SSH_HOST": "frigate-host",
        "FRIGATE_DB_SSH_PORT": 2222,
        "FRIGATE_DB_SSH_IDENTITY": "~/.ssh/frigate",
    })

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout='[{"id":"event-2","camera":"back"},{"id":"event-1","camera":"front"}]\n',
            stderr="",
        )

    monkeypatch.setattr(trainer.subprocess, "run", fake_run)

    assert trainer._frigate_event_rows() == [
        ("event-2", "back"),
        ("event-1", "front"),
    ]

    cmd = captured["cmd"]
    assert cmd[0] == "ssh"
    assert "BatchMode=yes" in cmd
    assert "ClearAllForwardings=yes" in cmd
    assert "-p" in cmd and "2222" in cmd
    assert "-i" in cmd
    assert cmd[-2] == "frigate-host"
    assert "sqlite3 -readonly -json" in cmd[-1]
    assert "/config/frigate.db" in cmd[-1]
    assert captured["kwargs"]["timeout"] == 30
