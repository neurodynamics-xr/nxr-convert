"""THE WRITER LOCK (OPEN-DEFECTS #26, 2026-10-06) — ONE WRITER A DATASET, across processes.

D141's recover-at-open rolls back EVERY ``pending`` row of a dataset's ``dataset.sqlite``: two writers on one dataset (a
terminal ``nxr-convert`` while the app imports, …) would each roll back the other's running composition. So a writer takes an
ADVISORY lock first: ``<dataset>/.writer.lock``, JSON ``{pid, host, program, started_utc}``, created atomically
(``O_CREAT|O_EXCL``). The app's twin is ``backend/src/service/writer-lock.ts``; the two read each other's files.

    ACQUIRE   create the file and write its JSON at once (a failed write removes it); if it exists, read it. STALE (same host
              only): a dead pid; a start before this machine's boot (a reused pid); an unreadable file older than 30 s — taken
              over through ``<lock>.break``. A live pid, a fresh unreadable file, or another host is REFUSED (``WriterLockHeld``)
    RELEASE   the file removed (only while it still names this process); on close, and at interpreter exit for any still held
    CRASH     the file stays and names a dead pid — the next writer on this host takes it over

THE HANDSHAKE — the app runs the converter INSIDE its own import, holding the lock: it sets
``NXR_WRITER_LOCK_HELD=<lock file>:<app pid>``. The converter accepts that lock as its own when the variable names THIS
dataset's lock file AND the file names that pid as its holder on this host; it neither creates nor removes it (``borrowed``).

PID LIVENESS: POSIX ``os.kill(pid, 0)`` (``ProcessLookupError`` → dead; ``PermissionError`` → alive, not ours). On WINDOWS
``os.kill(pid, 0)`` is not a probe (signal 0 is CTRL_C_EVENT's value there and ``os.kill`` TERMINATES a process for any other
signal), so liveness is asked of the kernel through ``ctypes``: ``OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION)`` then
``GetExitCodeProcess`` — ``STILL_ACTIVE`` is alive; an ``OpenProcess`` failing with ERROR_INVALID_PARAMETER (no such pid)
is dead; any other failure (access denied) counts as alive. Not run on Windows in CI here.

RECOVERY ONLY BY THE TAKER: D141's roll-back runs only in the process that TOOK the lock, at that moment. A BORROWED lock
recovers nothing (``open_dataset``): the holder recovered when it took it, and a row pending since is its live write.

Limits: the lock is ADVISORY; another host's lock is never judged stale (remove the file by hand if that host is gone). A stale
takeover is SERIALISED by ``<lock>.break`` (``_take_over``); only an ABANDONED break met by two takers at once can let both win.
"""
from __future__ import annotations

import atexit
import errno
import json
import os
import socket
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

WRITER_LOCK_FILE = ".writer.lock"
WRITER_LOCK_ENV = "NXR_WRITER_LOCK_HELD"
PROGRAM = "nxr-convert"


class WriterLockHeld(RuntimeError):
    """A dataset's lock is held by another live writer (or one on another host)."""

    def __init__(self, file: Path, holder: dict | None):
        self.file, self.holder = file, holder
        super().__init__(f"{file.parent}: another writer holds this dataset — {describe_holder(holder)} ({file}); "
                         "wait for it to finish, or stop it")


class DatasetNotWritable(RuntimeError):
    """The dataset's folder cannot be written (read-only media, no write permission): no lock can be taken, so nothing may
    write or recover it here. The app opens such a dataset read-only (round 3)."""

    def __init__(self, folder: Path, code: str):
        self.folder, self.code = folder, code
        super().__init__(f"{folder}: not writable ({code}) — nxr-convert cannot write this dataset (read-only media, or no write "
                         "permission); copy it somewhere writable")


_NOT_WRITABLE = {errno.EROFS, errno.EACCES, errno.EPERM}


def database_not_writable(file: Path) -> tuple[str, Path] | None:
    """Can this process WRITE a dataset's database? A writable FOLDER (the lock was taken) does not say: ``dataset.sqlite`` may be
    read-only (0444), or a ``-wal`` / ``-shm`` / ``-journal`` beside it, and SQLite then opens it and fails only at the first write
    — inside recovery. So the file and its companions that exist, and the folder, must be writable. ``None`` when it can be
    written; else the first refusal's code (EACCES · EPERM · EROFS) and path. By permission (``os.access``), as the app's
    ``databaseNotWritable``: a rolled-back trial write in WAL mode never reaches the file."""
    file = Path(file)
    for p in (file, Path(f"{file}-wal"), Path(f"{file}-shm"), Path(f"{file}-journal"), file.parent):
        if p.exists() and not os.access(p, os.W_OK):
            # os.access answers a boolean; the code is the one a write would meet — EROFS on read-only media, else EACCES
            try:
                st = os.statvfs(p)
                ro = bool(st.f_flag & os.ST_RDONLY)
            except (OSError, AttributeError):
                ro = False
            return ("EROFS" if ro else "EACCES"), p
    return None


def _create(path: Path, folder: Path) -> int:
    """The O_EXCL create; a folder that cannot be written is ``DatasetNotWritable``, never a bare OSError."""
    try:
        return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise
    except OSError as e:
        if e.errno in _NOT_WRITABLE:
            raise DatasetNotWritable(folder, errno.errorcode.get(e.errno, str(e.errno))) from None
        raise


def describe_holder(h: dict | None) -> str:
    return f"{h.get('program', '?')} (pid {h['pid']} on {h['host']}, since {h.get('started_utc', '?')})" if h else "an unreadable lock"


def _windows_alive(pid: int) -> bool:
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    h = k32.OpenProcess(0x1000, False, pid)                                 # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return ctypes.get_last_error() != 87                                 # ERROR_INVALID_PARAMETER: no such process
    try:
        code = wintypes.DWORD()
        if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
            return True
        return code.value == 259                                            # STILL_ACTIVE
    finally:
        k32.CloseHandle(h)


def pid_alive(pid: int) -> bool:
    """Is ``pid`` a live process on this host?"""
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        return _windows_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def lock_file(dataset_dir: str | Path) -> Path:
    return Path(dataset_dir).resolve() / WRITER_LOCK_FILE


def read_writer_lock(dataset_dir: str | Path) -> dict | None:
    """The lock's holder as its file states it — None when there is no file or it cannot be read."""
    try:
        h = json.loads(lock_file(dataset_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return h if isinstance(h, dict) and isinstance(h.get("pid"), int) and isinstance(h.get("host"), str) else None


_held: set[Path] = set()


def _release_all() -> None:
    for f in list(_held):
        try:
            if (read_writer_lock(f.parent) or {}).get("pid") == os.getpid():
                f.unlink()
        except OSError:
            pass
    _held.clear()


atexit.register(_release_all)


@dataclass
class WriterLock:
    file: Path
    holder: dict[str, Any]
    borrowed: bool = False
    _released: bool = field(default=False, repr=False)

    def release(self) -> None:
        if self._released or self.borrowed:
            self._released = True
            return
        self._released = True
        _held.discard(self.file)
        try:
            if (read_writer_lock(self.file.parent) or {}).get("pid") == os.getpid():
                self.file.unlink()
        except OSError:
            pass                                                              # gone with its folder


def _inherited(file: Path, env: Mapping[str, str]) -> WriterLock | None:
    v = env.get(WRITER_LOCK_ENV)
    if not v or ":" not in v:
        return None
    named, _, pid = v.rpartition(":")
    try:
        pid_n = int(pid)
    except ValueError:
        return None
    if Path(named).resolve() != file.resolve():
        return None
    h = read_writer_lock(file.parent)
    if not h or h["pid"] != pid_n or h["host"] != socket.gethostname():
        return None
    return WriterLock(file, h, borrowed=True)


def acquire_writer_lock(dataset_dir: str | Path, *, program: str = PROGRAM, env: Mapping[str, str] | None = None) -> WriterLock:
    """TAKE the writer lock of the dataset at ``dataset_dir`` (the folder must exist). Raises ``WriterLockHeld`` while another
    live writer holds it. ``env`` is read for the handshake (default: this process's)."""
    file = lock_file(dataset_dir)
    got = _inherited(file, os.environ if env is None else env)
    if got:
        return got
    holder = {"pid": os.getpid(), "host": socket.gethostname(), "program": program,
              "started_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}
    try:
        fd = _create(file, file.parent)
    except FileExistsError:
        h = read_writer_lock(file.parent)
        was = _identity(file)
        if not _stale(file, h):
            raise WriterLockHeld(file, h) from None
        fd = _take_over(file, h, was)
    # the JSON in the same call as the create; if it cannot be written, the empty file is REMOVED — never left to block (F2)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(holder))
    except BaseException:
        try:
            file.unlink()
        except FileNotFoundError:
            pass
        raise
    _held.add(file)
    return WriterLock(file, holder)


UNREADABLE_STALE_S = 30.0
"""An UNREADABLE lock (empty, torn) older than this is STALE: its creator writes it in the same call, microseconds after the
create, so one still unreadable after 30 s is a crash or a failed write (F2)."""
BOOT_SLACK_S = 60.0


def boot_time() -> float | None:
    """This machine's boot as a Unix time, or None if it cannot be read: Linux ``/proc/uptime``, macOS ``sysctl -n
    kern.boottime``, Windows ``GetTickCount64``. A lock STARTED BEFORE it cannot be a live process's (F3a)."""
    import sys
    try:
        if sys.platform.startswith("linux"):
            return time.time() - float(Path("/proc/uptime").read_text().split()[0])
        if sys.platform == "darwin":
            import re
            import subprocess
            out = subprocess.run(["sysctl", "-n", "kern.boottime"], capture_output=True, text=True, timeout=5).stdout
            m = re.search(r"sec\s*=\s*(\d+)", out)
            return float(m.group(1)) if m else None
        if os.name == "nt":
            import ctypes
            k32 = ctypes.WinDLL("kernel32")
            k32.GetTickCount64.restype = ctypes.c_ulonglong
            return time.time() - k32.GetTickCount64() / 1000.0
    except Exception:  # noqa: BLE001 — unknown boot: no lock is stale by it
        return None
    return None


def _identity(file: Path) -> tuple | None:
    """What a lock file IS on disk — compared under the break, so only the very file judged stale is removed."""
    try:
        st = file.stat()
    except FileNotFoundError:
        return None
    return (st.st_ino, st.st_mtime_ns, st.st_size)


def _started(h: dict) -> float | None:
    try:
        return datetime.fromisoformat(str(h.get("started_utc", "")).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _stale(file: Path, h: dict | None) -> bool:
    """STALE, same host only: a dead pid; this process's pid with no lock object here (a leftover); a start before this
    machine's boot (a reused pid); or an unreadable file older than ``UNREADABLE_STALE_S``."""
    if h is None:
        try:
            return time.time() - file.stat().st_mtime > UNREADABLE_STALE_S
        except FileNotFoundError:
            return False
    if h["host"] != socket.gethostname():
        return False
    if h["pid"] == os.getpid():
        return file not in _held
    if not pid_alive(h["pid"]):
        return True
    boot, started = boot_time(), _started(h)
    return boot is not None and started is not None and started < boot - BOOT_SLACK_S


BREAK_GRACE_S = 10.0
"""How long a ``.writer.lock.break`` may stand before it is judged abandoned (a taker that died mid-takeover)."""


def _take_over(file: Path, stale: dict | None, was: tuple | None) -> int:
    """TAKE OVER the stale lock, SERIALISED — the twin of ``writer-lock.ts``' ``takeOver``. A plain unlink + create lets two
    takers both win (the second's unlink removes the first's fresh lock; measured in Node: two of eight racers won). So a
    takeover first holds ``<lock>.break`` (O_EXCL); a taker meeting another's break is refused. Under the break the lock must
    still be the SAME file judged stale (its holder — pid, started_utc, host — and its inode, mtime and size), else a fresh
    holder's and we are refused; then it is removed and the
    lock created by the same O_EXCL create — a newcomer that created it in between wins and we lose with EEXIST. One winner. A
    break older than ``BREAK_GRACE_S`` is removed and retried once; two takers clearing one abandoned break at the same instant
    can still both win."""
    brk = file.with_name(file.name + ".break")
    b = None
    for attempt in range(2):
        try:
            b = _create(brk, file.parent)
            break
        except FileExistsError:
            try:
                age = time.time() - brk.stat().st_mtime
            except FileNotFoundError:
                continue
            if attempt == 0 and age > BREAK_GRACE_S:
                try:
                    brk.unlink()
                except FileNotFoundError:
                    pass
                continue
            raise WriterLockHeld(file, stale) from None
    if b is None:
        raise WriterLockHeld(file, read_writer_lock(file.parent))
    try:
        now, at = read_writer_lock(file.parent), _identity(file)
        key = (lambda h: (h["pid"], h.get("started_utc"), h["host"]) if h else None)
        same = at is None or (at == was and key(now) == key(stale))
        if not same:                                                         # a fresh holder's (or one being written this instant)
            raise WriterLockHeld(file, now)
        try:
            file.unlink()
        except FileNotFoundError:
            pass
        try:
            return _create(file, file.parent)
        except FileExistsError:
            raise WriterLockHeld(file, read_writer_lock(file.parent)) from None
    finally:
        os.close(b)
        try:
            brk.unlink()
        except FileNotFoundError:
            pass
