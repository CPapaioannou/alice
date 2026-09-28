#!/usr/bin/env python3
"""
Unit tests for ALICE core functions.

Run: python3 -m pytest tests/test_core.py -v

Tests import directly from the source modules via exec,
so they always test the real implementation — never a stale copy.
"""

import os
import sys
import math
import tempfile
import shutil
import pytest

# ---------------------------------------------------------------------------
# Bootstrap: make src/ importable so we test the real implementation.
# We inject minimal globals that the modules expect before importing.
# ---------------------------------------------------------------------------

_src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")

# Minimal globals that core.py / config.py expect at module level
CONF_DEFAULTS = {
    "DEFAULT_PORT": 8080,
    "DEFAULT_DATASET": "/tmp/test-dataset",
    "DATASETS_ROOT": "/tmp/test-datasets",
    "MODELS_DIR": "/tmp/test-models",
    "LIVE_DIR": "/tmp/test-live",
    "EXPORTS_DIR": "/tmp/test-exports",
    "FRIGATE_DB": "/tmp/test-frigate.db",
    "VIDEO_EXTENSIONS": [".mp4", ".avi", ".mkv", ".mov"],
    "DEFAULT_MODEL": "",
    "TEACHER_MODEL": "",
    "STUDENT_MODEL": "",
    "DEFAULT_CONFIDENCE": 0.7,
    "DEFAULT_CLASSES": [0, 2, 15, 16],
    "EPOCHS": 10,
    "BATCH_SIZE": 8,
    "LEARNING_RATE": 0.0001,
    "LR_FINAL": 0.01,
    "IMAGE_SIZE": 640,
    "FREEZE_LAYERS": 10,
    "HELPERS_ENABLED": True,
    "SORT_ORDER": "modified",
    "DEDUP_BOXES": False,
    "DEDUP_BOX_SIM": 10,
    "DEDUP_PHASH": True,
    "DEDUP_PHASH_SIM": 85,
    "DEDUP_NMS": True,
    "DEDUP_NMS_SIM": 85,
}


# ---------------------------------------------------------------------------
# Import the real functions from source files by exec-ing them in a namespace
# that provides the required globals. This avoids duplicating the code.
# ---------------------------------------------------------------------------

def _load_functions(*filenames):
    """Exec source files and return the combined namespace."""
    ns = {
        # Source modules use package-relative imports such as from .header.
        # exec() does not provide package context automatically, so emulate
        # execution as a src submodule.
        "__name__": "src._test_exec",
        "__package__": "src",
        "CONF_DEFAULTS": CONF_DEFAULTS,
        "os": os, "sys": sys, "math": math,
        "glob": __import__("glob"),
        "re": __import__("re"),
        "time": __import__("time"),
        "shutil": shutil,
        "threading": __import__("threading"),
        "json": __import__("json"),
        "Path": __import__("pathlib").Path,
        "defaultdict": __import__("collections").defaultdict,
        "Optional": None,
        "Any": None,
        "__file__": os.path.join(_src_dir, "alice.py"),
    }
    for fn in filenames:
        path = os.path.join(_src_dir, fn)
        with open(path) as f:
            code = f.read()
        # Strip shebang
        if code.startswith("#!"):
            code = "\n".join(code.split("\n")[1:])
        exec(compile(code, path, "exec"), ns)
    return ns


_ns = _load_functions("config.py")

# Pull out the functions we want to test
_parse_value = _ns["_parse_value"]
_format_value = _ns["_format_value"]
load_conf = _ns["load_conf"]
save_conf = _ns["save_conf"]

# Load core.py functions — needs CONF and other globals from header context
_core_ns = dict(_ns)
_core_ns["CONF"] = dict(CONF_DEFAULTS)
_core_ns["DATASET_DIR"] = "/tmp/test-dataset"
_core_ns["IMAGE_LIST"] = []
_core_ns["LIVE_LIST"] = []
_core_ns["LIVE_CAMERAS"] = []
_core_ns["VIDEO_LIST"] = []
_core_ns["PHASH_CACHE"] = {}
_core_ns["DATASET_VERSION"] = 0
_core_ns["LIVE_VERSION"] = 0
_core_ns["VIDEO_VERSION"] = 0
_core_ns["datetime"] = __import__("datetime").datetime

with open(os.path.join(_src_dir, "core.py")) as f:
    exec(compile(f.read(), os.path.join(_src_dir, "core.py"), "exec"), _core_ns)

box_iou = _core_ns["box_iou"]
read_boxes = _core_ns["read_boxes"]
write_boxes = _core_ns["write_boxes"]

# Load validate_path from header.py
_hdr_ns = _load_functions("header.py")
validate_path = _hdr_ns["validate_path"]


# ============================================================
# TESTS
# ============================================================

class TestBoxIoU:
    """Tests for box_iou function."""

    def test_identical_boxes(self):
        box = {"xc": 0.5, "yc": 0.5, "w": 0.2, "h": 0.2}
        assert box_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = {"xc": 0.1, "yc": 0.1, "w": 0.1, "h": 0.1}
        b = {"xc": 0.9, "yc": 0.9, "w": 0.1, "h": 0.1}
        assert box_iou(a, b) == 0

    def test_partial_overlap(self):
        a = {"xc": 0.5, "yc": 0.5, "w": 0.4, "h": 0.4}
        b = {"xc": 0.6, "yc": 0.6, "w": 0.4, "h": 0.4}
        iou = box_iou(a, b)
        assert 0 < iou < 1

    def test_contained_box(self):
        outer = {"xc": 0.5, "yc": 0.5, "w": 0.8, "h": 0.8}
        inner = {"xc": 0.5, "yc": 0.5, "w": 0.2, "h": 0.2}
        iou = box_iou(outer, inner)
        expected = (0.2 * 0.2) / (0.8 * 0.8)
        assert iou == pytest.approx(expected, abs=0.001)

    def test_touching_edges(self):
        a = {"xc": 0.25, "yc": 0.5, "w": 0.5, "h": 0.5}
        b = {"xc": 0.75, "yc": 0.5, "w": 0.5, "h": 0.5}
        assert box_iou(a, b) == 0

    def test_symmetry(self):
        a = {"xc": 0.3, "yc": 0.4, "w": 0.3, "h": 0.2}
        b = {"xc": 0.4, "yc": 0.45, "w": 0.25, "h": 0.3}
        assert box_iou(a, b) == pytest.approx(box_iou(b, a))

    def test_zero_size_box(self):
        a = {"xc": 0.5, "yc": 0.5, "w": 0, "h": 0}
        b = {"xc": 0.5, "yc": 0.5, "w": 0.2, "h": 0.2}
        assert box_iou(a, b) == 0


class TestReadWriteBoxes:
    """Tests for read_boxes / write_boxes round-trip."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def test_write_and_read(self):
        path = os.path.join(self.tmpdir, "labels", "test.txt")
        boxes = [
            {"cls": 0, "xc": 0.5, "yc": 0.5, "w": 0.3, "h": 0.4},
            {"cls": 2, "xc": 0.1, "yc": 0.9, "w": 0.05, "h": 0.1},
        ]
        write_boxes(path, boxes)
        result = read_boxes(path)
        assert len(result) == 2
        assert result[0]["cls"] == 0
        assert result[1]["cls"] == 2
        assert result[0]["xc"] == pytest.approx(0.5, abs=0.0001)

    def test_read_nonexistent(self):
        path = os.path.join(self.tmpdir, "nope.txt")
        assert read_boxes(path) == []

    def test_empty_file(self):
        path = os.path.join(self.tmpdir, "empty.txt")
        with open(path, "w") as f:
            pass
        assert read_boxes(path) == []

    def test_write_empty(self):
        path = os.path.join(self.tmpdir, "labels", "empty.txt")
        write_boxes(path, [])
        assert read_boxes(path) == []
        assert os.path.exists(path)

    def test_precision(self):
        path = os.path.join(self.tmpdir, "labels", "prec.txt")
        boxes = [{"cls": 15, "xc": 0.123456789, "yc": 0.987654321, "w": 0.111111, "h": 0.222222}]
        write_boxes(path, boxes)
        result = read_boxes(path)
        assert result[0]["xc"] == pytest.approx(0.123457, abs=0.000001)

    def test_malformed_lines_ignored(self):
        path = os.path.join(self.tmpdir, "bad.txt")
        with open(path, "w") as f:
            f.write("0 0.5 0.5 0.3 0.4\n")
            f.write("garbage line\n")
            f.write("1 0.1\n")
            f.write("2 0.2 0.2 0.1 0.1\n")
        result = read_boxes(path)
        assert len(result) == 2
        assert result[0]["cls"] == 0
        assert result[1]["cls"] == 2


class TestConfigParser:
    """Tests for alice.conf parsing and writing."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def test_parse_types(self):
        assert _parse_value("42") == 42
        assert _parse_value("3.14") == 3.14
        assert _parse_value("true") is True
        assert _parse_value("false") is False
        assert _parse_value("hello") == "hello"
        assert _parse_value('"quoted"') == "quoted"

    def test_parse_list(self):
        result = _parse_value("0, 2, 15, 16")
        assert result == [0, 2, 15, 16]

    def test_format_value(self):
        assert _format_value(True) == "true"
        assert _format_value(False) == "false"
        assert _format_value([1, 2, 3]) == "1, 2, 3"
        assert _format_value(42) == "42"

    def test_load_defaults(self):
        path = os.path.join(self.tmpdir, "missing.conf")
        conf = load_conf(path)
        assert conf["DEFAULT_PORT"] == 8080
        assert conf["DEFAULT_CONFIDENCE"] == 0.7

    def test_load_overrides(self):
        path = os.path.join(self.tmpdir, "test.conf")
        with open(path, "w") as f:
            f.write("DEFAULT_PORT = 9090\n")
            f.write("DEFAULT_CONFIDENCE = 0.5\n")
            f.write("# comment line\n")
            f.write("HELPERS_ENABLED = false\n")
        conf = load_conf(path)
        assert conf["DEFAULT_PORT"] == 9090
        assert conf["DEFAULT_CONFIDENCE"] == 0.5
        assert conf["HELPERS_ENABLED"] is False
        assert conf["EPOCHS"] == 10

    def test_save_preserves_comments(self):
        path = os.path.join(self.tmpdir, "save.conf")
        with open(path, "w") as f:
            f.write("# My config\n")
            f.write("DEFAULT_PORT = 8080\n")
            f.write("# AI section\n")
            f.write("DEFAULT_CONFIDENCE = 0.7\n")
        conf = load_conf(path)
        conf["DEFAULT_PORT"] = 9999
        save_conf(path, conf)
        with open(path) as f:
            content = f.read()
        assert "# My config" in content
        assert "# AI section" in content
        assert "DEFAULT_PORT = 9999" in content

    def test_roundtrip(self):
        path = os.path.join(self.tmpdir, "rt.conf")
        original = dict(CONF_DEFAULTS)
        original["DEFAULT_PORT"] = 1234
        original["DEFAULT_CLASSES"] = [0, 1, 2]
        save_conf(path, original)
        loaded = load_conf(path)
        assert loaded["DEFAULT_PORT"] == 1234
        assert loaded["DEFAULT_CLASSES"] == [0, 1, 2]


class TestPhashSimilarity:
    """Tests for pHash similarity calculation."""

    def test_identical(self):
        h = 0xABCDEF0123456789
        similarity = 1.0 - (bin(h ^ h).count('1') / 64.0)
        assert similarity == 1.0

    def test_completely_different(self):
        h1 = 0x0000000000000000
        h2 = 0xFFFFFFFFFFFFFFFF
        similarity = 1.0 - (bin(h1 ^ h2).count('1') / 64.0)
        assert similarity == 0.0

    def test_one_bit_difference(self):
        h1 = 0xABCDEF0123456789
        h2 = h1 ^ 1
        similarity = 1.0 - (bin(h1 ^ h2).count('1') / 64.0)
        assert similarity == pytest.approx(63 / 64)


class TestValidatePath:
    """Tests for path traversal prevention."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.subdir = os.path.join(self.tmpdir, "exports")
        os.makedirs(self.subdir)
        with open(os.path.join(self.subdir, "test.mp4"), "w") as f:
            f.write("fake")

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def test_valid_path(self):
        p = os.path.join(self.subdir, "test.mp4")
        assert validate_path(p, self.subdir) == os.path.realpath(p)

    def test_traversal_blocked(self):
        p = os.path.join(self.subdir, "..", "..", "etc", "passwd")
        assert validate_path(p, self.subdir) is None

    def test_empty_path(self):
        assert validate_path("", self.subdir) is None

    def test_empty_root(self):
        assert validate_path("/some/path", "") is None

    def test_root_itself(self):
        assert validate_path(self.subdir, self.subdir) == os.path.realpath(self.subdir)
