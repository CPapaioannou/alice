#!/usr/bin/env python3
"""
Regression tests for path-validation and safety guards (B3:
fix/path-validation-and-guards).

Covers:
  1. _guard_rel() — the path-traversal primitive used by the API handlers
  2. handler-level rejection of client-supplied names/splits/paths that
     escape the dataset (arbitrary read/write/delete)
  3. _post_settings_save() — unknown keys are dropped, all values normalized
     through the same _parse_value the conf loader uses (no raw JSON type leak)
  4. trainer single-run lock — a second step is rejected while one runs
  5. LogCapture — the pipe write-end is closed and the reader thread joined on
     exit (no fd/thread leak)

Run: python3 -m pytest tests/test_path_guards.py -v
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
# resolves between them. Heavy third-party deps (cv2, ultralytics) are lazy
# imports inside function bodies, so nothing heavy loads at import time.
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
    return m


def _build_modules():
    # Snapshot any pre-existing src.* entries so we can restore them: leaving
    # our fakes in sys.modules would break other test modules that do a real
    # `from src.trainer import ...`.
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

        # Stub the trainer module with no-op step functions so handler_trainer's
        # launched threads do nothing (we only test the lock / pre-flight here).
        trainer_stub = {
            "trainer_dedup_run": lambda **kw: {"ok": True, "dry_run": kw.get("dry_run", False)},
            "trainer_export_dataset": lambda **kw: {"ok": True, "exported": 0},
            "trainer_export_onnx": lambda *a, **kw: {"ok": True},
            "trainer_reannotate": lambda *a, **kw: {"ok": True},
            "trainer_train": lambda *a, **kw: {"ok": True},
        }
        _install_mod("src.trainer", trainer_stub)

        hapi = _exec_into({}, "handler_api.py")
        _install_mod("src.handler_api", hapi)

        htrain = _exec_into({}, "handler_trainer.py")
        _install_mod("src.handler_trainer", htrain)

        return header, config, hapi, htrain
    finally:
        for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
            del sys.modules[k]
        sys.modules.update(_saved)


_MODS = _build_modules()
_HEADER, _CONFIG, _HAPI, _HTRAIN = _MODS

# Shared mutable state objects (same dicts/lists the handler code mutates).
STATE = _HEADER["STATE"]
CONF = _HEADER["CONF"]
_step_lock = threading.Lock()


def _mk_env():
    """Create an isolated dataset + datasets-root + victim file layout."""
    tmp = tempfile.mkdtemp()
    ds_root = os.path.join(tmp, "datasets")
    ds = os.path.join(ds_root, "default")
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(ds, sub), exist_ok=True)
    # A normal dataset image (so the "valid" path cases have something to hit)
    with open(os.path.join(ds, "images", "train", "good.jpg"), "wb") as f:
        f.write(b"x")
    # A victim file OUTSIDE the datasets root that traversal must not touch
    victim = os.path.join(tmp, "victim.txt")
    with open(victim, "w") as f:
        f.write("do-not-touch\n")
    # A dataset dir OUTSIDE the configured root (for switch/copymove tests)
    outside_ds = os.path.join(tmp, "elsewhere_ds")
    os.makedirs(os.path.join(outside_ds, "images"), exist_ok=True)

    CONF["DATASETS_ROOT"] = ds_root
    CONF["LIVE_DIR"] = os.path.join(tmp, "live")
    os.makedirs(CONF["LIVE_DIR"], exist_ok=True)
    STATE["DATASET_DIR"] = ds
    STATE["CONF_PATH"] = os.path.join(tmp, "alice.conf")
    STATE["TRAINER_DATASET"] = ds
    return {"tmp": tmp, "ds": ds, "ds_root": ds_root, "victim": victim, "outside_ds": outside_ds}


def _status(resp):
    """Unpack a (status, ctype, body) handler response -> (status, json)."""
    import json as _json
    return resp[0], _json.loads(resp[2])


# ============================================================
# 1. _guard_rel — the traversal primitive
# ============================================================

class TestGuardRel:
    def setup_method(self):
        self.env = _mk_env()
        self.root = self.env["ds"]

    def teardown_method(self):
        shutil.rmtree(self.env["tmp"], ignore_errors=True)

    def test_normal_name_allowed(self):
        assert _HAPI["_guard_rel"](self.root, "images", "train", "good.jpg") is True

    def test_dotdot_component_rejected(self):
        assert _HAPI["_guard_rel"](self.root, "images", "..", "good.jpg") is False

    def test_embedded_separator_rejected(self):
        assert _HAPI["_guard_rel"](self.root, "images", "train", "../../../etc/passwd") is False

    def test_absolute_path_rejected(self):
        assert _HAPI["_guard_rel"](self.root, "images", "/etc/passwd") is False

    def test_empty_component_rejected(self):
        assert _HAPI["_guard_rel"](self.root, "images", "", "good.jpg") is False

    def test_backslash_separator_rejected(self):
        assert _HAPI["_guard_rel"](self.root, "images", "train", "..\\..\\x.jpg") is False


# ============================================================
# 2. Handler-level path rejection
# ============================================================

class TestHandlerPathRejection:
    def setup_method(self):
        self.env = _mk_env()

    def teardown_method(self):
        shutil.rmtree(self.env["tmp"], ignore_errors=True)

    def _del(self, payload):
        return _status(_HAPI["_post_del"](payload))

    def test_del_traversal_rejected_and_victim_intact(self):
        st, body = self._del({"split": "train", "name": "../../../../victim.txt"})
        assert st == 400
        assert body["ok"] is False
        assert os.path.exists(self.env["victim"]), "traversal must not delete outside files"

    def test_del_dotdot_split_rejected(self):
        st, body = self._del({"split": "..", "name": "good.jpg"})
        assert st == 400

    def test_del_valid_image_ok(self):
        st, body = self._del({"split": "train", "name": "good.jpg"})
        assert st == 200 and body["ok"] is True

    def test_save_traversal_rejected(self):
        st, body = _status(_HAPI["_post_save"](
            {"split": "train", "name": "../../victim.txt", "boxes": []}))
        assert st == 400

    def test_ai_traversal_rejected(self):
        st, body = _status(_HAPI["_post_ai"](
            {"split": "train", "name": "../../../etc/passwd",
             "model": "m.pt", "conf": 0.5, "classes": [0]}))
        assert st == 400

    def test_preview_ai_traversal_rejected(self):
        st, body = _status(_HAPI["_post_preview_ai"](
            {"split": "train", "name": "../../../../victim.txt", "model": "m.pt"}))
        assert st == 400

    def test_switch_outside_root_rejected(self):
        st, body = _status(_HAPI["_post_switch"]({"path": self.env["outside_ds"]}))
        assert st == 400
        # DATASET_DIR must be unchanged
        assert STATE["DATASET_DIR"] == self.env["ds"]

    def test_copymove_dst_outside_root_rejected(self):
        st, body = _status(_HAPI["_post_copymove"]({
            "src_split": "train", "src_name": "good.jpg",
            "dst_dataset": self.env["outside_ds"],
            "dst_split": "train", "action": "copy",
        }))
        assert st == 400
        # No image must have been written to the outside dataset
        assert not os.path.exists(
            os.path.join(self.env["outside_ds"], "images", "train", "good.jpg"))

    def test_copymove_src_traversal_rejected(self):
        st, body = _status(_HAPI["_post_copymove"]({
            "src_split": "train", "src_name": "../../victim.txt",
            "dst_dataset": self.env["ds"], "dst_split": "val", "action": "copy",
        }))
        assert st == 400


# ============================================================
# 3. _post_settings_save — type/key guard
# ============================================================

class TestSettingsSaveGuard:
    def setup_method(self):
        self.env = _mk_env()
        self._saved_conf = dict(CONF)

    def teardown_method(self):
        CONF.clear()
        CONF.update(self._saved_conf)
        shutil.rmtree(self.env["tmp"], ignore_errors=True)

    def test_unknown_key_dropped(self):
        st, body = _status(_HAPI["_post_settings_save"]({"TOTALLY_UNKNOWN_KEY": "1"}))
        assert st == 200
        assert "TOTALLY_UNKNOWN_KEY" not in CONF

    def test_string_value_parsed_like_loader(self):
        _HAPI["_post_settings_save"]({"EPOCHS": "7"})
        assert CONF["EPOCHS"] == 7 and isinstance(CONF["EPOCHS"], int)

    def test_non_string_normalized_not_leaked(self):
        # A raw JSON value that is not a string must be stored exactly as the
        # conf-file loader (_parse_value) would produce from its string form —
        # so the in-memory CONF and the on-disk alice.conf always agree. The
        # old code stored the raw object (e.g. a float/int/list) verbatim,
        # which drifted from what a later load_conf() would read back.
        _parse_value = _CONFIG["_parse_value"]
        sent = 12  # a non-string (int) sent for a numeric conf key
        _HAPI["_post_settings_save"]({"DEDUP_BOX_SIM": sent})
        assert CONF["DEDUP_BOX_SIM"] == _parse_value(str(sent)), "value not normalized through the loader"
        # Round-trip: in-memory value equals a fresh load of the saved file.
        loaded = _CONFIG["load_conf"](STATE["CONF_PATH"])
        assert loaded["DEDUP_BOX_SIM"] == CONF["DEDUP_BOX_SIM"]

    def test_unknown_key_not_written_to_file(self):
        # A client-supplied unknown key must not be injected into alice.conf.
        _HAPI["_post_settings_save"]({"EVIL_INJECTED_KEY": "pwned"})
        assert "EVIL_INJECTED_KEY" not in CONF
        on_disk = open(STATE["CONF_PATH"]).read()
        assert "EVIL_INJECTED_KEY" not in on_disk

    def test_bool_normalized(self):
        _HAPI["_post_settings_save"]({"DEDUP_BOXES": True})
        assert CONF["DEDUP_BOXES"] is True


# ============================================================
# 4. Trainer single-run lock
# ============================================================

class TestTrainerRunLock:
    def setup_method(self):
        self.env = _mk_env()

    def teardown_method(self):
        # Release in case a test left the lock held
        if _HTRAIN["TRAINER_RUN_LOCK"].locked():
            _HTRAIN["TRAINER_RUN_LOCK"].release()
        shutil.rmtree(self.env["tmp"], ignore_errors=True)

    def test_second_step_rejected_while_busy(self):
        _HTRAIN["TRAINER_RUN_LOCK"].acquire()
        try:
            st, body = _status(_HTRAIN["_post_trainer_export"]({"max_images": 0}))
            assert st == 400
            assert "already running" in body["error"]
        finally:
            _HTRAIN["TRAINER_RUN_LOCK"].release()

    def test_step_allowed_when_idle(self):
        st, body = _status(_HTRAIN["_post_trainer_export"]({"max_images": 0}))
        assert st == 200 and body.get("async") is True
        # The launched thread acquires+releases the run lock; wait for it to
        # finish so the lock is free for teardown.
        _HTRAIN["TRAINER_RUN_LOCK"].acquire()  # blocks until the worker releases
        _HTRAIN["TRAINER_RUN_LOCK"].release()

    def test_set_dataset_outside_root_rejected(self):
        st, body = _status(_HTRAIN["_post_trainer_set_dataset"](
            {"path": self.env["outside_ds"]}))
        assert st == 400


# ============================================================
# 5. LogCapture — fd/thread leak fix
# ============================================================

class TestLogCaptureLeak:
    def _open_fds(self):
        return set(os.listdir("/proc/self/fd"))

    def test_no_fd_leak_after_capture(self):
        cap = _HEADER["LogCapture"]()
        before = self._open_fds()
        with cap:
            pass
        after = self._open_fds()
        leaked = after - before
        assert not leaked, f"LogCapture leaked file descriptors: {sorted(leaked)}"

    def test_reader_thread_joined_and_cleaned_up(self):
        cap = _HEADER["LogCapture"]()
        with cap:
            reader = cap._reader_thread
            assert reader is not None
        # After exit the reader must have been joined and the refs cleared.
        assert cap._reader_thread is None
        assert cap._out_file is None
        assert not reader.is_alive()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
