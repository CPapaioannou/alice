#!/usr/bin/env python3
"""
Regression tests for capture-time sorting and watcher debouncing
(fix/sort-by-capture-time).

Covers:
  1. event_capture_ts() parses the Unix timestamp embedded in Frigate-style
     image names ('1790589611.837512-k5vr9z.jpg') and returns 0.0 otherwise
  2. build_image_list() carries a capture_ts field
  3. sort_image_list() with SORT_ORDER='event' orders chronologically by
     capture time, with unknown-timestamp images last
  4. _drain_watch_events() coalesces event bursts (N files -> ~2 callbacks)

Run: python3 -m pytest tests/test_capture_sort.py -v
"""

import os
import sys
import tempfile
import shutil
import pytest

_src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


# ---------------------------------------------------------------------------
# Namespace bootstrap (same pattern as tests/test_core.py / test_export_split.py)
# ---------------------------------------------------------------------------

def _load_core_ns(tmp_dataset_dir):
    import threading
    from datetime import datetime
    from pathlib import Path
    from collections import defaultdict

    conf = {}
    state = {"DATASET_DIR": tmp_dataset_dir}

    def _conf(key):
        return conf.get(key)

    core_ns = {
        "__name__": "src._test_exec_core",
        "__package__": "src",
        "CONF": conf,
        "CONF_DEFAULTS": {},
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

    # core.py's `from .header import ...` re-bound several names to the real
    # module globals during exec; re-inject the test-owned objects AFTER exec.
    core_ns.update({"CONF": conf, "STATE": state, "conf": _conf})
    return core_ns


def _mkimg(ds_dir, split, name, mtime=None):
    """Create an image file in the tmp dataset with a controllable mtime."""
    d = os.path.join(ds_dir, "images", split)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name)
    with open(p, "wb") as f:
        f.write(b"x")
    if mtime is not None:
        os.utime(p, (mtime, mtime))
    return p


# ============================================================
# TESTS
# ============================================================

class TestEventCaptureTs:
    def setup_method(self):
        tmp = tempfile.mkdtemp()
        self.ns = _load_core_ns(tmp)
        self.teardown_tmp = tmp

    def teardown_method(self):
        shutil.rmtree(self.teardown_tmp, ignore_errors=True)
        self.teardown_tmp = None

    def test_frigate_event_id(self):
        assert self.ns["event_capture_ts"]("1790589611.837512-k5vr9z.jpg") == pytest.approx(1790589611.837512)

    def test_no_fraction(self):
        assert self.ns["event_capture_ts"]("1790590301-v09x6z.jpg") == pytest.approx(1790590301)

    def test_plain_name(self):
        assert self.ns["event_capture_ts"]("sunset-photo.jpg") == 0.0

    def test_video_frame_name(self):
        assert self.ns["event_capture_ts"]("driveway_clip_f000120.jpg") == 0.0

    def test_short_numeric_not_a_timestamp(self):
        # 8 digits is a date, not a unix ts — must not match
        assert self.ns["event_capture_ts"]("20260929-abc.jpg") == 0.0

    def test_milliseconds(self):
        assert self.ns["event_capture_ts"]("1790589611837-x.jpg") == pytest.approx(1790589611837)


class TestBuildImageListCaptureTs:
    def test_capture_ts_field_present(self):
        tmp = tempfile.mkdtemp()
        ds = os.path.join(tmp, "dataset")
        try:
            ns = _load_core_ns(ds)
            _mkimg(ds, "train", "1790589611.837512-k5vr9z.jpg")
            _mkimg(ds, "train", "plain.jpg")
            items = ns["build_image_list"]()
            by_name = {i["name"]: i for i in items}
            assert "capture_ts" in by_name["plain.jpg"]
            assert by_name["1790589611.837512-k5vr9z.jpg"]["capture_ts"] == pytest.approx(1790589611.837512)
            assert by_name["plain.jpg"]["capture_ts"] == 0.0
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestSortEventMode:
    def test_event_order_chronological(self):
        tmp = tempfile.mkdtemp()
        ds = os.path.join(tmp, "dataset")
        try:
            ns = _load_core_ns(ds)
            # Capture times: a=100, b=300, c=200; mtimes REVERSED (b newest)
            _mkimg(ds, "train", "100.0-a.jpg", mtime=1000)
            _mkimg(ds, "train", "300.0-b.jpg", mtime=3000)
            _mkimg(ds, "train", "200.0-c.jpg", mtime=2000)
            _mkimg(ds, "train", "plain-x.jpg", mtime=1500)

            ns["CONF"]["SORT_ORDER"] = "event"
            ns["IMAGE_LIST"].clear()
            ns["IMAGE_LIST"].extend(ns["build_image_list"]())
            ns["sort_image_list"]()
            names = [i["name"] for i in ns["IMAGE_LIST"]]
            # chronological, unknown-ts last
            assert names == ["100.0-a.jpg", "200.0-c.jpg", "300.0-b.jpg", "plain-x.jpg"]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_modified_order_unchanged(self):
        tmp = tempfile.mkdtemp()
        ds = os.path.join(tmp, "dataset")
        try:
            ns = _load_core_ns(ds)
            _mkimg(ds, "train", "100.0-a.jpg", mtime=1000)
            _mkimg(ds, "train", "300.0-b.jpg", mtime=3000)
            _mkimg(ds, "train", "200.0-c.jpg", mtime=2000)

            ns["CONF"]["SORT_ORDER"] = "modified"
            ns["IMAGE_LIST"].clear()
            ns["IMAGE_LIST"].extend(ns["build_image_list"]())
            ns["sort_image_list"]()
            names = [i["name"] for i in ns["IMAGE_LIST"]]
            assert names == ["100.0-a.jpg", "200.0-c.jpg", "300.0-b.jpg"]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_filename_order_unchanged(self):
        tmp = tempfile.mkdtemp()
        ds = os.path.join(tmp, "dataset")
        try:
            ns = _load_core_ns(ds)
            _mkimg(ds, "val", "100.0-a.jpg", mtime=1000)
            _mkimg(ds, "train", "300.0-b.jpg", mtime=3000)

            ns["CONF"]["SORT_ORDER"] = "filename"
            ns["IMAGE_LIST"].clear()
            ns["IMAGE_LIST"].extend(ns["build_image_list"]())
            ns["sort_image_list"]()
            names = [i["name"] for i in ns["IMAGE_LIST"]]
            # (split, name): train before val
            assert names == ["300.0-b.jpg", "100.0-a.jpg"]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestDrainWatchEvents:
    def _ev(self, filename):
        return (0, ["IN_CREATE"], "/x", filename)

    def test_single_event_one_call(self):
        tmp = tempfile.mkdtemp()
        try:
            ns = _load_core_ns(tmp)
            calls = []
            ns["_drain_watch_events"]([self._ev("a.jpg")], lambda: calls.append(1),
                                      (".jpg", ".jpeg", ".png", ".webp", ".txt"))
            assert calls == [1]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_burst_coalesced_to_two_calls(self):
        tmp = tempfile.mkdtemp()
        try:
            ns = _load_core_ns(tmp)
            calls = []
            stream = [self._ev(f"f{i}.jpg") for i in range(500)]  # no quiet tick
            ns["_drain_watch_events"](stream, lambda: calls.append(1),
                                      (".jpg", ".jpeg", ".png", ".webp", ".txt"))
            # first event immediate + one trailing flush
            assert calls == [1, 1]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_quiet_tick_flushes_burst(self):
        tmp = tempfile.mkdtemp()
        try:
            ns = _load_core_ns(tmp)
            calls = []
            stream = [self._ev("a.jpg"), self._ev("b.jpg"), None,
                      self._ev("c.jpg"), None]
            ns["_drain_watch_events"](stream, lambda: calls.append(1),
                                      (".jpg", ".jpeg", ".png", ".webp", ".txt"))
            # a: immediate; b: coalesced -> flushed on None; c: immediate; None with no burst
            assert calls == [1, 1, 1]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_non_matching_extensions_ignored(self):
        tmp = tempfile.mkdtemp()
        try:
            ns = _load_core_ns(tmp)
            calls = []
            ns["_drain_watch_events"]([self._ev("vid.mp4"), None], lambda: calls.append(1),
                                      (".jpg", ".jpeg", ".png", ".webp", ".txt"))
            assert calls == []
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
