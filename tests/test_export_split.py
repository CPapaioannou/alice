#!/usr/bin/env python3
"""
Regression tests for the export split-integrity fixes (fix/export-split-integrity).

Covers:
  1. assign_split() is deterministic and assigns ~10% of events to val
  2. A second no-limit export of the same Frigate rows exports 0 new files
     and never re-rolls train/val membership (the "re-export grows dataset" bug)
  3. Cross-split duplicate images are cleaned up, with boxes merged
  4. Orphan label files (label without its image) are removed
  5. camera_map.json is merged across runs, not clobbered by limited exports
  6. pHash dedup split-settling is idempotent (second run moves 0 files)

Run: python3 -m pytest tests/test_export_split.py -v

Like tests/test_core.py, this execs the real src/ modules in a synthetic
namespace so it tests the shipped implementation, never a copy.
"""

import os
import sys
import json
import zlib
import types
import tempfile
import shutil
import pytest

_src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")

CONF_DEFAULTS = {
    "DEFAULT_PORT": 8080,
    "DEFAULT_DATASET": "/tmp/test-dataset",
    "DATASETS_ROOT": "/tmp/test-datasets",
    "MODELS_DIR": "/tmp/test-models",
    "LIVE_DIR": "/tmp/test-clips",
    "EXPORTS_DIR": "/tmp/test-exports",
    "FRIGATE_DB": "/tmp/test-frigate.db",
    "FRIGATE_DB_SSH_HOST": "",
    "VIDEO_EXTENSIONS": [".mp4", ".avi", ".mkv", ".mov"],
    "DEFAULT_MODEL": "",
    "TEACHER_MODEL": "",
    "STUDENT_MODEL": "",
    "DEFAULT_CONFIDENCE": 0.7,
    "DEFAULT_CLASSES": [0, 2, 15, 16],
    "EPOCHS": 10,
    "BATCH_SIZE": 8,
    "EARLY_STOPPING_PATIENCE": 15,
    "LEARNING_RATE": 0.0001,
    "LR_FINAL": 0.01,
    "IMAGE_SIZE": 640,
    "FREEZE_LAYERS": 10,
    "AUGMENTATION": False,
    "DEVICE": "cpu",
    "HELPERS_ENABLED": True,
    "SORT_ORDER": "modified",
    "WELCOME_DISMISSED": True,
    "DEDUP_BOXES": False,
    "DEDUP_BOX_SIM": 10,
    "DEDUP_PHASH": True,
    "DEDUP_PHASH_SIM": 85,
    "DEDUP_NMS": False,
    "DEDUP_NMS_SIM": 85,
}


# ---------------------------------------------------------------------------
# Namespace bootstrap (same pattern as tests/test_core.py)
# ---------------------------------------------------------------------------

def _make_step_status():
    empty = {"running": False, "progress": 0, "current": 0, "total": 0, "message": "", "epochs": []}
    return {k: dict(empty) for k in ("export", "dedup", "annotate", "train", "onnx")}


def _load_trainer_ns(tmp):
    """Exec core.py + trainer.py in a namespace with all needed globals."""
    import threading
    from datetime import datetime
    from pathlib import Path
    from collections import defaultdict

    dataset_dir = os.path.join(tmp, "dataset")
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(dataset_dir, sub), exist_ok=True)

    conf = dict(CONF_DEFAULTS)
    conf["LIVE_DIR"] = os.path.join(tmp, "clips")
    os.makedirs(conf["LIVE_DIR"], exist_ok=True)

    state = {"DATASET_DIR": dataset_dir, "TRAINER_DATASET": dataset_dir}

    def _conf(key):
        return conf.get(key, CONF_DEFAULTS.get(key))

    class _LogCapture:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    # core.py namespace
    core_ns = {
        "__name__": "src._test_exec_core",
        "__package__": "src",
        "CONF": conf,
        "CONF_DEFAULTS": CONF_DEFAULTS,
        "STATE": state,
        "IMAGE_LIST": [],
        "LIVE_LIST": [],
        "LIVE_ALL": [],
        "LIVE_CAMERAS": [],
        "VIDEO_LIST": [],
        "_state_lock": threading.Lock(),
        "conf": _conf,
        "os": os, "sys": sys, "glob": __import__("glob"), "re": __import__("re"),
        "time": __import__("time"), "threading": threading,
        "datetime": datetime, "Path": Path, "defaultdict": defaultdict,
    }
    with open(os.path.join(_src_dir, "core.py")) as f:
        code = f.read()
    if code.startswith("#!"):
        code = "\n".join(code.split("\n")[1:])
    exec(compile(code, os.path.join(_src_dir, "core.py"), "exec"), core_ns)

    # trainer.py namespace (built on top of core's so it sees assign_split etc.)
    trainer_ns = dict(core_ns)
    trainer_ns.update({
        "CLASS_NAMES": {0: "person", 2: "car"},
        "REVERSE_CLASS_NAMES": {v: k for k, v in {0: "person", 2: "car"}.items()},
        "STEP_STATUS": _make_step_status(),
        "TRAINER_LOG": [],
        "LogCapture": _LogCapture,
        "resolve_device": lambda: "cpu",
        "compute_phash": None,  # replaced per-test
        "csv": __import__("csv"),
        "shlex": __import__("shlex"),
        "shutil": shutil,
        "sqlite3": __import__("sqlite3"),
        "subprocess": __import__("subprocess"),
        "zlib": zlib,
    })
    with open(os.path.join(_src_dir, "trainer.py")) as f:
        code = f.read()
    if code.startswith("#!"):
        code = "\n".join(code.split("\n")[1:])
    exec(compile(code, os.path.join(_src_dir, "trainer.py"), "exec"), trainer_ns)

    # trainer.py's `from .header import ...` re-bound several names to the real
    # module globals during exec; re-inject the test-owned objects AFTER exec so
    # the exec'd functions operate on THIS dataset, not the process-wide one.
    trainer_ns.update({
        "CONF": conf,
        "CONF_DEFAULTS": CONF_DEFAULTS,
        "STATE": state,
        "conf": _conf,
        "STEP_STATUS": _make_step_status(),
        "TRAINER_LOG": [],
        "LogCapture": _LogCapture,
        "resolve_device": lambda: "cpu",
        "compute_phash": None,
    })
    return trainer_ns


# ---------------------------------------------------------------------------
# Fake cv2 — keeps the export path hermetic (no opencv / real webp needed)
# ---------------------------------------------------------------------------

class _FakeCv2:
    IMWRITE_JPEG_QUALITY = 1

    @staticmethod
    def imread(path):
        return b"FAKE-IMG" if os.path.exists(path) else None

    @staticmethod
    def imwrite(path, data, params=None):
        # Unique content per destination so pHash treats each event as distinct
        with open(path, "wb") as f:
            f.write(b"JPEG-" + os.path.basename(path).encode())
        return True


@pytest.fixture
def trainer_env(tmp_path, monkeypatch):
    """Return (trainer_ns, dataset_dir, clips_dir) with fake cv2 + fake Frigate rows."""
    clips_dir = os.path.join(tmp_path, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    saved_cv2 = sys.modules.get("cv2")
    sys.modules["cv2"] = _FakeCv2()
    try:
        ns = _load_trainer_ns(tmp_path)
        dataset_dir = ns["STATE"]["DATASET_DIR"]

        rows = []
        rows_holder = {}

        def _set_rows(new_rows):
            rows_holder["rows"] = new_rows

        ns["_frigate_event_rows"] = lambda: rows_holder.get("rows", [])
        ns["_set_rows_for_test"] = _set_rows
        yield ns, dataset_dir, clips_dir
    finally:
        if saved_cv2 is None:
            sys.modules.pop("cv2", None)
        else:
            sys.modules["cv2"] = saved_cv2


def _make_rows(clips_dir, event_ids, camera="cam1"):
    """Create fake snapshot files and return the Frigate row tuples."""
    rows = []
    for eid in event_ids:
        src = os.path.join(clips_dir, f"{camera}-{eid}-clean.webp")
        with open(src, "wb") as f:
            f.write(b"WEBP-FAKE")
        rows.append((eid, camera))
    return rows


def _all_images(dataset_dir):
    found = {}
    for split in ("train", "val"):
        d = os.path.join(dataset_dir, "images", split)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                found.setdefault(f, []).append(split)
    return found


# ============================================================
# TESTS
# ============================================================

class TestAssignSplit:
    def test_deterministic(self):
        # exec core alone for this lightweight check
        ns = {}
        with open(os.path.join(_src_dir, "core.py")) as f:
            code = f.read()
        if code.startswith("#!"):
            code = "\n".join(code.split("\n")[1:])
        # core.py needs a few globals at module scope
        ns.update({
            "__name__": "src._t", "__package__": "src",
            "CONF": {}, "CONF_DEFAULTS": {}, "STATE": {},
            "IMAGE_LIST": [], "LIVE_LIST": [], "LIVE_ALL": [], "LIVE_CAMERAS": [],
            "VIDEO_LIST": [], "_state_lock": __import__("threading").Lock(),
            "conf": lambda k: None,
            "os": os, "glob": __import__("glob"), "re": __import__("re"),
            "time": __import__("time"), "threading": __import__("threading"),
            "datetime": __import__("datetime").datetime,
            "Path": __import__("pathlib").Path,
        })
        exec(compile(code, os.path.join(_src_dir, "core.py"), "exec"), ns)
        assign_split = ns["assign_split"]
        assert assign_split("1790589611.837512-k5vr9z") == assign_split("1790589611.837512-k5vr9z")
        assert assign_split("a") == assign_split("a")

    def test_distribution(self):
        ns = {}
        with open(os.path.join(_src_dir, "core.py")) as f:
            code = f.read()
        ns.update({
            "__name__": "src._t", "__package__": "src",
            "CONF": {}, "CONF_DEFAULTS": {}, "STATE": {},
            "IMAGE_LIST": [], "LIVE_LIST": [], "LIVE_ALL": [], "LIVE_CAMERAS": [],
            "VIDEO_LIST": [], "_state_lock": __import__("threading").Lock(),
            "conf": lambda k: None,
            "os": os, "glob": __import__("glob"), "re": __import__("re"),
            "time": __import__("time"), "threading": __import__("threading"),
            "datetime": __import__("datetime").datetime,
            "Path": __import__("pathlib").Path,
        })
        exec(compile(code, os.path.join(_src_dir, "core.py"), "exec"), ns)
        assign_split = ns["assign_split"]
        val = sum(1 for i in range(10000) if assign_split(f"1790000000.1-{i:04d}") == "val")
        assert 700 < val < 1300  # ~10%


class TestExportIdempotency:
    def test_second_no_limit_export_exports_zero(self, trainer_env):
        ns, dataset_dir, clips_dir = trainer_env
        rows = _make_rows(clips_dir, [f"1790{ i:04d}.5-ab{i}" for i in range(12)])
        ns["_set_rows_for_test"](rows)

        r1 = ns["trainer_export_dataset"](max_images=0)
        assert r1["ok"] is True
        assert r1["exported"] == 12
        layout_1 = _all_images(dataset_dir)

        r2 = ns["trainer_export_dataset"](max_images=0)
        assert r2["ok"] is True
        assert r2["exported"] == 0
        assert r2["existing"] == 12
        layout_2 = _all_images(dataset_dir)

        # Dataset must not grow on re-export
        assert layout_2 == layout_1
        # No event may exist in both splits
        for splits in layout_2.values():
            assert len(splits) == 1, f"event in both splits: {splits}"

    def test_split_assignment_is_stable_across_runs(self, trainer_env):
        ns, dataset_dir, clips_dir = trainer_env
        rows = _make_rows(clips_dir, [f"1790{i:04d}.5-ab{i}" for i in range(12)])
        ns["_set_rows_for_test"](rows)
        ns["trainer_export_dataset"](max_images=0)
        layout_1 = {f: s[0] for f, s in _all_images(dataset_dir).items()}

        ns["trainer_export_dataset"](max_images=0)
        layout_2 = {f: s[0] for f, s in _all_images(dataset_dir).items()}
        assert layout_1 == layout_2


class TestIntegrityCleanup:
    def test_cross_split_duplicate_cleaned_with_box_merge(self, trainer_env):
        ns, dataset_dir, clips_dir = trainer_env
        rows = _make_rows(clips_dir, ["e1", "e2", "e3"])
        ns["_set_rows_for_test"](rows)
        ns["trainer_export_dataset"](max_images=0)

        # Simulate the historical corruption: event e1 present in BOTH splits,
        # with boxes only in the stale (val) copy. Cleanup keeps the train-side
        # copy (train is scanned first) and merges the stale label's boxes in.
        e1_img = os.path.join(dataset_dir, "images")
        stale_img = os.path.join(e1_img, "val", "e1.jpg")
        if not os.path.exists(stale_img):
            # e1's original copy is in whichever split it was exported to
            original = os.path.join(e1_img, "train", "e1.jpg")
            if os.path.exists(original):
                shutil.copy2(original, stale_img)
            else:
                shutil.copy2(os.path.join(e1_img, "val", "e1.jpg"), stale_img)
        stale_lbl = os.path.join(dataset_dir, "labels", "val", "e1.txt")
        with open(stale_lbl, "w") as f:
            f.write("0 0.5 0.5 0.4 0.4\n")

        r = ns["trainer_export_dataset"](max_images=0)
        assert r["ok"] is True
        assert r["exported"] == 0
        assert r["dupes_removed"] == 1

        # Only the train copy survives, and it carries the merged boxes
        assert os.path.exists(os.path.join(e1_img, "train", "e1.jpg"))
        assert not os.path.exists(stale_img)
        boxes = ns["read_boxes"](os.path.join(dataset_dir, "labels", "train", "e1.txt"))
        assert len(boxes) == 1 and boxes[0]["cls"] == 0

    def test_orphan_label_removed(self, trainer_env):
        ns, dataset_dir, clips_dir = trainer_env
        rows = _make_rows(clips_dir, ["e1"])
        ns["_set_rows_for_test"](rows)
        ns["trainer_export_dataset"](max_images=0)

        orphan = os.path.join(dataset_dir, "labels", "train", "ghost-event.txt")
        with open(orphan, "w") as f:
            f.write("2 0.5 0.5 0.1 0.1\n")

        r = ns["trainer_export_dataset"](max_images=0)
        assert r["orphans_removed"] == 1
        assert not os.path.exists(orphan)


class TestCameraMapMerge:
    def test_limited_export_does_not_clobber_camera_map(self, trainer_env):
        ns, dataset_dir, clips_dir = trainer_env

        # Run 1: limited export of the first 3 of 6 events
        rows_a = _make_rows(clips_dir, [f"a{i}" for i in range(3)], camera="camA")
        _make_rows(clips_dir, [f"b{i}" for i in range(3)], camera="camB")  # sources exist but not exported
        ns["_set_rows_for_test"](rows_a)
        r1 = ns["trainer_export_dataset"](max_images=3)
        assert r1["ok"] is True

        # Run 2: limited export of 3 different events
        rows_b = [(f"b{i}", "camB") for i in range(3)]
        ns["_set_rows_for_test"](rows_b)
        r2 = ns["trainer_export_dataset"](max_images=3)
        assert r2["ok"] is True

        with open(os.path.join(dataset_dir, "camera_map.json")) as f:
            cam_map = json.load(f)
        # Entries from BOTH runs must be present (merge, not clobber)
        assert all(k in cam_map for k in ("a0", "a1", "a2", "b0", "b1", "b2"))
        assert cam_map["a0"] == "camA"
        assert cam_map["b0"] == "camB"


class TestDedupSettling:
    def test_phash_dedup_settling_is_idempotent(self, trainer_env):
        ns, dataset_dir, clips_dir = trainer_env
        rows = _make_rows(clips_dir, [f"d{i}" for i in range(20)])
        ns["_set_rows_for_test"](rows)
        ns["trainer_export_dataset"](max_images=0)

        # 64-bit deterministic phash (two salted crc32s): unique content per
        # event means nothing is a duplicate; only split-settling can move files.
        def _fake_phash(p):
            data = open(p, "rb").read()
            return zlib.crc32(data) | (zlib.crc32(data + b"salt") << 32)
        ns["compute_phash"] = _fake_phash
        cam_map = {f"d{i}": "cam1" for i in range(20)}
        with open(os.path.join(dataset_dir, "camera_map.json"), "w") as f:
            json.dump(cam_map, f)

        # Run 1: settles any mis-assigned images to their stable split
        r1 = ns["trainer_dedup_run"](phash=True, hamming=10, dry_run=False)
        assert r1["ok"] is True
        layout_1 = {f: s[0] for f, s in _all_images(dataset_dir).items()}

        # Run 2: must move nothing
        r2 = ns["trainer_dedup_run"](phash=True, hamming=10, dry_run=False)
        assert r2["ok"] is True
        assert r2["steps"][0]["removed"] == 0
        layout_2 = {f: s[0] for f, s in _all_images(dataset_dir).items()}
        assert layout_1 == layout_2
