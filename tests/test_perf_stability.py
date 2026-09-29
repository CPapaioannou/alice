#!/usr/bin/env python3
"""
Regression tests for frontend performance/stability (B4: perf/frontend-stability).

Covers:
  1. get_filtered — reads a consistent snapshot (torn-read safe), and the
     pin-insertion is O(n) (no IMAGE_LIST.index inside a loop) while still
     placing the pinned image at its natural position
  2. pHash — ensure_hashes_computed no longer spawns a per-request
     multiprocessing.Pool; a shared thread pool + background warmup
     (precompute_hashes_async) keeps the cache warm

Run: python3 -m pytest tests/test_perf_stability.py -v
"""

import os
import sys
import types
import shutil
import tempfile
import threading
import pytest

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


# ---------------------------------------------------------------------------
# Bootstrap: exec the real src modules in shared namespaces so `from .x import`
# resolves between them, then restore sys.modules so other test modules that do
# a real `from src.trainer import ...` see the real package.
# ---------------------------------------------------------------------------

def _exec_into(ns, fname):
    path = os.path.join(_SRC, fname)
    with open(path) as f:
        code = f.read()
    if code.startswith("#!"):
        code = "\n".join(code.split("\n")[1:])
    ns["__name__"] = "src." + fname[:-3]
    ns["__package__"] = "src"
    ns["__file__"] = path
    exec(compile(code, path, "exec"), ns)
    return ns


def _install_mod(name, ns):
    m = types.ModuleType(name)
    for k, v in list(ns.items()):
        setattr(m, k, v)
    sys.modules[name] = m


def _build_modules():
    _saved = {k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")}
    try:
        header = _exec_into({}, "header.py")
        _install_mod("src.header", header)
        config = _exec_into({}, "config.py")
        _install_mod("src.config", config)
        core = _exec_into({}, "core.py")
        _install_mod("src.core", core)
        ai = _exec_into({}, "ai_phash_video.py")
        _install_mod("src.ai_phash_video", ai)
        return header, core, ai
    finally:
        for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
            del sys.modules[k]
        sys.modules.update(_saved)


_HEADER, _CORE, _AIV = _build_modules()
IMAGE_LIST = _HEADER["IMAGE_LIST"]
PHASH_CACHE = _HEADER["PHASH_CACHE"]
STATE = _HEADER["STATE"]


def _mk_ds(tmp):
    ds = os.path.join(tmp, "ds")
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(ds, sub), exist_ok=True)
    return ds


def _img(ds, split, name):
    p = os.path.join(ds, "images", split, name)
    with open(p, "wb") as f:
        f.write(b"x")
    return p


def _item(split, name, boxes=0, classes=None):
    return {"split": split, "name": name, "boxes": boxes,
            "classes": classes or [], "mtime": 0, "capture_ts": 0.0}


# ============================================================
# 1. get_filtered — snapshot + O(n) pin-insertion
# ============================================================

class TestGetFiltered:
    def setup_method(self):
        IMAGE_LIST.clear()

    def teardown_method(self):
        IMAGE_LIST.clear()

    def test_filter_train_returns_subsequence(self):
        IMAGE_LIST.extend([
            _item("train", "a"), _item("val", "b"), _item("train", "c"), _item("val", "d"),
        ])
        out = _CORE["get_filtered"]("train")
        assert [x["name"] for x in out] == ["a", "c"]

    def test_pin_inserted_at_natural_position(self):
        # Full order: a(train), b(val), c(train), d(val), e(train)
        IMAGE_LIST.extend([
            _item("train", "a"), _item("val", "b"), _item("train", "c"),
            _item("val", "d"), _item("train", "e"),
        ])
        # Filter to train, but the user is currently viewing 'd' (a val item).
        # 'd' must be re-inserted at its natural position (index 2 of the full
        # list), i.e. between 'c' and 'e'.
        out = _CORE["get_filtered"]("train", pin_split="val", pin_name="d")
        assert [x["name"] for x in out] == ["a", "c", "d", "e"]

    def test_pin_equals_natural_order_invariant(self):
        # Strong invariant: filtering with a pin == full list order with the
        # non-matching items dropped but the pinned item kept in place.
        full = [
            _item("train", "a"), _item("val", "b"), _item("train", "c"),
            _item("val", "d"), _item("train", "e"), _item("val", "f"),
        ]
        IMAGE_LIST.extend(full)
        expected = [x for x in full if x["split"] == "train" or x["name"] == "d"]
        out = _CORE["get_filtered"]("train", pin_split="val", pin_name="d")
        assert [x["name"] for x in out] == [x["name"] for x in expected]

    def test_pin_already_included_not_duplicated(self):
        IMAGE_LIST.extend([_item("train", "a"), _item("train", "b")])
        out = _CORE["get_filtered"]("train", pin_split="train", pin_name="a")
        assert [x["name"] for x in out] == ["a", "b"]  # 'a' appears once

    def test_returned_list_is_a_copy(self):
        IMAGE_LIST.extend([_item("train", "a"), _item("train", "b")])
        out = _CORE["get_filtered"]("train")
        out.append(_item("train", "zzz"))
        # Mutating the returned list must not affect IMAGE_LIST
        assert all(x["name"] != "zzz" for x in IMAGE_LIST)

    def test_no_index_in_loop(self):
        # Regression guard: the old implementation called IMAGE_LIST.index()
        # once per list item (O(n^2)). It must not reappear.
        src = open(os.path.join(_SRC, "core.py")).read()
        fn_start = src.index("def get_filtered")
        fn_end = src.index("\ndef ", fn_start + 1)
        body = src[fn_start:fn_end]
        # Match an actual call (with an open paren); the explanatory comment
        # mentions the name without the call, so this only fires on real code.
        assert "IMAGE_LIST.index(" not in body, "O(n^2) IMAGE_LIST.index call reappeared"


# ============================================================
# 2. pHash — no per-request multiprocessing.Pool
# ============================================================

class TestPHashNoMultiprocessing:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.ds = _mk_ds(self.tmp, )
        # 4 images in the dataset
        for i, (split, name) in enumerate([
            ("train", "a.jpg"), ("train", "b.jpg"), ("val", "c.jpg"), ("val", "d.jpg")]):
            _img(self.ds, split, name)
        STATE["DATASET_DIR"] = self.ds
        PHASH_CACHE.clear()
        # Reset the warmup state and stub the worker (no PIL needed).
        _AIV["_PHASH_WARM"]["sig"] = None
        _AIV["_PHASH_WARM"]["fut"] = None
        self._orig_worker = _AIV["_phash_worker"]
        _AIV["_phash_worker"] = lambda p: 0x1234

    def teardown_method(self):
        _AIV["_phash_worker"] = self._orig_worker
        PHASH_CACHE.clear()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_no_multiprocessing_in_source(self):
        # Regression guard: ensure_hashes_computed must not fork a process pool.
        src = open(os.path.join(_SRC, "ai_phash_video.py")).read()
        fn_start = src.index("def ensure_hashes_computed")
        fn_end = src.index("\ndef ", fn_start + 1)
        body = src[fn_start:fn_end]
        assert "multiprocessing" not in body
        assert "Pool(" not in body

    def test_ensure_hashes_fills_cache(self):
        _AIV["ensure_hashes_computed"]()
        # All 4 dataset images must now be cached with the stub hash.
        cached = [p for p in PHASH_CACHE if p.startswith(self.ds)]
        assert len(cached) == 4
        assert all(PHASH_CACHE[p] == 0x1234 for p in cached)

    def test_ensure_hashes_uses_thread_pool(self):
        # The executor is created lazily and reused across calls.
        _AIV["ensure_hashes_computed"]()
        ex1 = _AIV["_PHASH_EXECUTOR"]
        assert ex1 is not None
        _AIV["ensure_hashes_computed"]()
        assert _AIV["_PHASH_EXECUTOR"] is ex1  # same shared executor

    def test_precompute_async_dedups_same_dataset(self):
        fut1 = _AIV["precompute_hashes_async"]()
        fut2 = _AIV["precompute_hashes_async"]()
        assert fut1 is fut2, "same-dataset precompute should be deduped"
        fut1.join(timeout=10)
        cached = [p for p in PHASH_CACHE if p.startswith(self.ds)]
        assert len(cached) == 4

    def test_precompute_async_new_dataset_starts_new_warmup(self):
        fut1 = _AIV["precompute_hashes_async"]()
        fut1.join(timeout=10)
        # Switch to a different (empty) dataset -> different signature.
        ds2 = _mk_ds(os.path.join(self.tmp, "b"))
        STATE["DATASET_DIR"] = ds2
        _AIV["_PHASH_WARM"]["sig"] = None  # force a fresh decision
        fut2 = _AIV["precompute_hashes_async"]()
        assert fut2 is not fut1, "different dataset should start a new warmup"
        fut2.join(timeout=10)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
