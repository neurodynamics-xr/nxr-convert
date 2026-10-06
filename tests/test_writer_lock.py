"""THE WRITER LOCK (OPEN-DEFECTS #26) — one writer a dataset across processes, the converter's half (``writer_lock.py``; the
app's twin is ``backend/src/service/writer-lock.ts``, and the two read each other's files).

A writing ``open_dataset`` / ``create_dataset`` takes ``<dataset>/.writer.lock`` and ``close`` releases it; a lock a LIVE
process holds (the app, here a real child process) refuses the open WITHOUT recovering — the holder's pending rows are its
running composition; a dead holder's lock is stale and taken over; the app's handshake (``NXR_WRITER_LOCK_HELD``) is accepted.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time

import pytest

from nxr_convert.cli import main
from nxr_convert.crud import create_dataset, delete_dataset, open_dataset
from nxr_convert.writer_lock import (WRITER_LOCK_ENV, WRITER_LOCK_FILE, WriterLockHeld, acquire_writer_lock, pid_alive,
                                     read_writer_lock)


@pytest.fixture
def sleeper():
    """A real process that stays alive until the test ends."""
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    yield p
    p.kill()
    p.wait()


def dead_pid() -> int:
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def lock_as(folder, pid: int, program: str = "cortical-flow", host: str | None = None, started_utc: str | None = None) -> None:
    """A lock file as the APP writes it (TypeScript's ``acquireWriterLock``) — started NOW unless told (a start before this
    machine's boot is stale, F3a)."""
    from datetime import datetime, timezone
    (folder / WRITER_LOCK_FILE).write_text(json.dumps({"pid": pid, "host": host or socket.gethostname(), "program": program,
                                                       "started_utc": started_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}))


def with_pending(tmp_path):
    """A dataset holding one PENDING subject row (a crashed composition) — released."""
    ds = create_dataset(tmp_path / "store", "d")
    with ds.db.tx():
        ds.db.insert("subject", {"id": "sub-1", "dataset_id": ds.id, "name": "sub-1", "path": "sub-1.nxr.zarr", "kind": "subject",
                                 "status": "pending"})
    folder = ds.folder
    ds.close()
    return folder


def status_of(folder) -> str | None:
    ds = open_dataset(folder, lock=False)
    try:
        r = ds.db.one("SELECT status FROM subject WHERE id = 'sub-1'")
        return r["status"] if r else None
    finally:
        ds.close()


def test_acquire_and_release(tmp_path):
    folder = tmp_path / "x"
    folder.mkdir()
    lock = acquire_writer_lock(folder, program="test", env={})
    h = read_writer_lock(folder)
    assert (h["pid"], h["host"], h["program"]) == (os.getpid(), socket.gethostname(), "test") and h["started_utc"].startswith("20")
    with pytest.raises(WriterLockHeld):                                     # a second take in this process, while held
        acquire_writer_lock(folder, program="again", env={})
    lock.release()
    lock.release()
    assert not (folder / WRITER_LOCK_FILE).exists()


def test_open_and_create_hold_the_lock_until_close(tmp_path):
    ds = create_dataset(tmp_path / "store", "d")
    assert read_writer_lock(ds.folder)["pid"] == os.getpid() and read_writer_lock(ds.folder)["program"] == "nxr-convert"
    folder = ds.folder
    ds.close()
    assert not (folder / WRITER_LOCK_FILE).exists()
    with open_dataset(folder) as again:
        assert read_writer_lock(folder)["pid"] == os.getpid()
        assert again.lock is not None
    assert not (folder / WRITER_LOCK_FILE).exists()
    with open_dataset(folder, lock=False, recover_now=False) as reader:      # a reader takes no lock
        assert reader.lock is None and not (folder / WRITER_LOCK_FILE).exists()


def test_a_live_app_lock_refuses_and_nothing_is_rolled_back(tmp_path, sleeper):
    folder = with_pending(tmp_path)
    assert pid_alive(sleeper.pid)
    lock_as(folder, sleeper.pid)
    with pytest.raises(WriterLockHeld, match=rf"cortical-flow \(pid {sleeper.pid} on .*since 20"):
        open_dataset(folder)
    assert status_of(folder) == "pending"                                    # the app's composition, untouched
    assert read_writer_lock(folder)["pid"] == sleeper.pid
    with pytest.raises(WriterLockHeld):
        delete_dataset(folder.parent, folder.name)
    assert folder.exists()
    # the CLI refuses with a stage:error naming the holder, exit 1
    assert main(["dataset", "recover", str(folder)]) == 1
    assert status_of(folder) == "pending"


def test_another_hosts_lock_is_never_stale(tmp_path):
    folder = with_pending(tmp_path)
    lock_as(folder, dead_pid(), host="elsewhere.invalid")
    with pytest.raises(WriterLockHeld, match="elsewhere.invalid"):
        open_dataset(folder)


def test_a_dead_holders_lock_is_stale_and_taken_over(tmp_path):
    folder = with_pending(tmp_path)
    pid = dead_pid()
    assert not pid_alive(pid)
    lock_as(folder, pid)
    with open_dataset(folder) as ds:
        assert read_writer_lock(folder)["pid"] == os.getpid()
        assert ds.db.one("SELECT 1 FROM subject WHERE id = 'sub-1'") is None  # rolled back (D141): it holds the lock
    assert not (folder / WRITER_LOCK_FILE).exists()


def test_the_handshake_accepts_the_apps_lock(tmp_path, sleeper, monkeypatch):
    folder = with_pending(tmp_path)
    lock_as(folder, sleeper.pid)
    monkeypatch.setenv(WRITER_LOCK_ENV, f"{folder / WRITER_LOCK_FILE}:{sleeper.pid}")
    with open_dataset(folder) as ds:
        assert ds.lock.borrowed
        # RECOVERY ONLY BY THE TAKER (fix round 1): the app recovered when it took the lock; a row pending since is ITS live write
        # (a save, D141's createNode step 1) — the converter it runs never rolls it back
        assert ds.db.one("SELECT status FROM subject WHERE id = 'sub-1'")["status"] == "pending"
    assert read_writer_lock(folder)["pid"] == sleeper.pid                    # the parent's file, left in place
    with pytest.raises(ValueError, match="took the writer lock"):                  # nor when asked to
        open_dataset(folder, recover_now=True)
    assert status_of(folder) == "pending"
    # a handshake naming another pid, or another dataset's lock, is not accepted
    monkeypatch.setenv(WRITER_LOCK_ENV, f"{folder / WRITER_LOCK_FILE}:{sleeper.pid + 1}")
    with pytest.raises(WriterLockHeld):
        open_dataset(folder)
    monkeypatch.setenv(WRITER_LOCK_ENV, f"{tmp_path / 'other' / WRITER_LOCK_FILE}:{sleeper.pid}")
    with pytest.raises(WriterLockHeld):
        open_dataset(folder)


def test_a_takeover_in_progress_is_not_raced(tmp_path):
    folder = with_pending(tmp_path)
    pid = dead_pid()
    lock_as(folder, pid)
    (folder / f"{WRITER_LOCK_FILE}.break").write_text("")
    with pytest.raises(WriterLockHeld):
        open_dataset(folder)
    assert read_writer_lock(folder)["pid"] == pid and status_of(folder) == "pending"


def test_an_abandoned_break_is_cleared(tmp_path):
    from nxr_convert.writer_lock import BREAK_GRACE_S
    folder = with_pending(tmp_path)
    lock_as(folder, dead_pid())
    brk = folder / f"{WRITER_LOCK_FILE}.break"
    brk.write_text("")
    old = time.time() - BREAK_GRACE_S - 5
    os.utime(brk, (old, old))
    with open_dataset(folder):
        assert read_writer_lock(folder)["pid"] == os.getpid() and not brk.exists()


def test_a_taker_that_loses_the_race_is_refused(tmp_path, monkeypatch):
    """The race, made deterministic: taker A judges the lock stale, and BEFORE A acts, taker B takes it over whole. A must lose
    — a plain unlink + create (the first version) removed B's fresh lock and both won. (A multi-process race in Python never
    reproduced that double win, so it did not discriminate; this does.)"""
    import nxr_convert.writer_lock as wl
    folder = tmp_path / "x"
    folder.mkdir()
    lock_as(folder, dead_pid())
    won, entered = [], []

    def judged_dead(pid):                       # A's liveness probe — B runs its whole takeover inside it
        if not entered:
            entered.append(1)
            won.append(wl.acquire_writer_lock(folder, program="B", env={}))
        return False
    monkeypatch.setattr(wl, "pid_alive", judged_dead)
    with pytest.raises(WriterLockHeld, match="B "):
        wl.acquire_writer_lock(folder, program="A", env={})
    assert read_writer_lock(folder)["program"] == "B"
    won[0].release()


def test_a_failed_write_leaves_no_lock(tmp_path, monkeypatch):
    import nxr_convert.writer_lock as wl
    folder = tmp_path / "x"
    folder.mkdir()

    def boom(*a, **k):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(wl.json, "dumps", boom)
    with pytest.raises(OSError):
        wl.acquire_writer_lock(folder, program="test", env={})
    assert not (folder / WRITER_LOCK_FILE).exists()


def test_an_unreadable_lock_is_held_while_fresh_and_stale_after_30_s(tmp_path):
    from nxr_convert.writer_lock import UNREADABLE_STALE_S
    folder = with_pending(tmp_path)
    f = folder / WRITER_LOCK_FILE
    f.write_text("")
    with pytest.raises(WriterLockHeld, match="unreadable"):
        open_dataset(folder)
    old = time.time() - UNREADABLE_STALE_S - 5
    os.utime(f, (old, old))
    with open_dataset(folder) as ds:
        assert read_writer_lock(folder)["pid"] == os.getpid()
        assert ds.db.one("SELECT 1 FROM subject WHERE id = 'sub-1'") is None


def test_a_lock_started_before_boot_is_stale_even_with_a_live_pid(tmp_path, sleeper):
    from datetime import datetime, timezone
    from nxr_convert.writer_lock import boot_time
    boot = boot_time()
    assert boot is not None and boot < time.time()
    folder = with_pending(tmp_path)
    before = datetime.fromtimestamp(boot - 3600, timezone.utc).isoformat().replace("+00:00", "Z")
    lock_as(folder, sleeper.pid, started_utc=before)
    with open_dataset(folder):
        assert read_writer_lock(folder)["pid"] == os.getpid()


def test_open_dataset_releases_on_a_failed_recovery(tmp_path, monkeypatch):
    import nxr_convert.crud as crud
    folder = with_pending(tmp_path)

    def boom(*a, **k):
        raise RuntimeError("recovery failed")
    monkeypatch.setattr(crud, "recover", boom)
    with pytest.raises(RuntimeError, match="recovery failed"):
        open_dataset(folder)
    assert not (folder / WRITER_LOCK_FILE).exists()
    monkeypatch.undo()
    with open_dataset(folder):                   # the lock was released: the next writer takes it
        pass


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root writes anywhere")
def test_a_dataset_folder_not_writable_is_refused_clearly_and_still_readable(tmp_path):
    from nxr_convert.writer_lock import DatasetNotWritable
    folder = with_pending(tmp_path)
    folder.chmod(0o555)
    try:
        with pytest.raises(DatasetNotWritable, match=r"not writable \(E(ACCES|PERM|ROFS)\)"):
            open_dataset(folder)
        assert main(["dataset", "recover", str(folder)]) == 1                 # the CLI: a stage:error, no traceback
        assert status_of(folder) == "pending"                                 # a reader still opens it
    finally:
        folder.chmod(0o755)


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root writes anywhere")
def test_a_writable_folder_with_a_read_only_database_is_refused_clearly(tmp_path):
    """The FOLDER is writable (the lock can be taken) but ``dataset.sqlite`` is not (0444, or a -wal/-shm beside it): SQLite would
    open it and fail at the first write, inside recovery. It is refused as ``DatasetNotWritable`` naming the file, the lock let go."""
    from nxr_convert.writer_lock import DatasetNotWritable
    folder = with_pending(tmp_path)
    db = folder / "dataset.sqlite"
    db.chmod(0o444)
    try:
        with pytest.raises(DatasetNotWritable, match=r"dataset\.sqlite: not writable \(EACCES\)"):
            open_dataset(folder)
        assert not (folder / WRITER_LOCK_FILE).exists()                      # the lock it took is let go
        assert main(["dataset", "recover", str(folder)]) == 1                 # the CLI: a stage:error, no traceback
        assert status_of(folder) == "pending"                                 # nothing rolled back; a reader still opens it
    finally:
        db.chmod(0o644)
    shm = folder / "dataset.sqlite-shm"
    shm.write_bytes(b"")
    shm.chmod(0o444)
    try:
        with pytest.raises(DatasetNotWritable, match=r"dataset\.sqlite-shm: not writable \(EACCES\)"):
            open_dataset(folder)
    finally:
        shm.chmod(0o644)
        shm.unlink()
    with open_dataset(folder):                   # writable again: it opens, and recovers
        pass
    assert status_of(folder) is None
