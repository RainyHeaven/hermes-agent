"""Idle stop for browser_exec harness daemons.

A ``browser_harness.daemon`` outlives every browser_exec call and was only stopped at process exit
(``stop_harness_daemons``), so a long-lived gateway kept one per ``BU_NAME`` forever and a crash orphaned
them. Each use is recorded as ``<HERMES_HOME>/cache/browser-use/harness-activity/<name>.json``; the
browser cleanup thread of any live Hermes process stops names unused for ``idle_timeout_s()``, including
those whose owner process has died. Fork customization (FORK_CUSTOMIZATIONS.md C6), kept in its own
module to stay clear of upstream merges.
"""
import contextlib
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# browser_use_cli._SESSION_RE, plus a leading ``_`` for the hybrid sidecar daemons (``_local_<name>``).
_NAME_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_-]{0,63}")
_TIMEOUT_ENV = "HERMES_BROWSER_HARNESS_IDLE_TIMEOUT"
_DEFAULT_TIMEOUT_S = 1800


def activity_dir() -> Path:
    return get_hermes_home() / "cache" / "browser-use" / "harness-activity"


def touch(name: str) -> None:
    """Record that the daemon ``name`` was just used. Best-effort: a failed write only delays its stop."""
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        return
    directory = activity_dir()
    tmp = directory / f".{name}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps({"name": name, "owner_pid": os.getpid(), "last_used": time.time()}),
                       encoding="utf-8")
        os.replace(tmp, directory / f"{name}.json")
    except OSError as e:
        logger.debug("Could not record harness daemon use for %s: %s", name, e)
        with contextlib.suppress(OSError):
            tmp.unlink()


def _as_int(value) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(str(value).strip())
    except ValueError:
        return None


def idle_timeout_s() -> int:
    """Seconds a daemon may sit unused: env, then ``browser.harness_idle_timeout``, then 1800. ``<= 0`` disables."""
    from_env = _as_int(os.environ.get(_TIMEOUT_ENV))
    if from_env is not None:
        return from_env
    from tools.browser_use_cli import _read_browser_cfg
    from_cfg = _as_int(_read_browser_cfg().get("harness_idle_timeout"))
    return _DEFAULT_TIMEOUT_S if from_cfg is None else from_cfg


def _last_used(path: Path) -> Optional[float]:
    """The record's ``last_used``; raises ValueError for a corrupt record, None if it vanished."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(e) from e
    last_used = data.get("last_used") if isinstance(data, dict) else None
    if isinstance(last_used, bool) or not isinstance(last_used, (int, float)) or not _NAME_RE.fullmatch(path.stem):
        raise ValueError(f"malformed record {path.name}")
    return float(last_used)


def reap_idle(stop: Callable[[str], None]) -> list[str]:
    """Stop every recorded daemon unused for longer than the timeout and drop its record.

    Safe to race with another process reaping the same name: ``stop`` is the harness's identity-checked
    ``--reload`` (a second call is a no-op) and a record already removed is skipped."""
    timeout = idle_timeout_s()
    if timeout <= 0:
        return []
    try:
        records = sorted(activity_dir().glob("*.json"))
    except OSError:
        return []
    now = time.time()
    stopped = []
    for path in records:
        try:
            last_used = _last_used(path)
        except ValueError as e:
            logger.debug("Dropping corrupt harness activity record: %s", e)
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
            continue
        if last_used is None or now - last_used <= timeout:
            continue
        name = path.stem
        try:
            stop(name)
            stopped.append(name)
        except Exception:  # one bad stop must not keep the other idle daemons alive
            logger.warning("Idle stop of harness daemon %s failed", name, exc_info=True)
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
    if stopped:
        logger.info("Stopped %d idle browser harness daemon(s): %s", len(stopped), ", ".join(stopped))
    return stopped
