"""Durable context-rollover transitions: one compare-and-set record per source session.

``agent/context_rollover_runtime.py`` drives a rollover; this module owns its storage. The SOURCE session's
``model_config`` carries the claim (``ROLLOVER_CLAIM_KEY``), the fresh TARGET root session the origin
(``ROLLOVER_ORIGIN_KEY``). Every other model_config writer keeps both keys from the stored row
(``hermes_state_compression._keep_full_compaction_count``), so only the transactions here write them.

States advance ``prepared -> claimed -> installed -> started -> completed``. ``claimed -> prepared`` releases
a claim that provably invoked nothing; any other failure after the claim ends in ``blocked``. Only a
``prepared`` record may be recovered or superseded: everything later needs a person, because a crashed
``started`` resume may already have run side effects and an orphaned ``claimed`` may still have a live owner.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Optional

from hermes_state_compression import (
    FULL_COMPACTION_COUNT_KEY, ROLLOVER_CLAIM_KEY, ROLLOVER_ORIGIN_KEY, strict_model_config,
)
from hermes_state_errors import (
    CompressionSessionBusyError, CompressionSessionClosedError, StateDbReplacedError,
)

RECORD_SCHEMA = 1
# What an ordinary store failure looks like from these calls (besides RolloverRecordError): a transaction that
# raised one of these was rolled back, so a compare-and-set retried later sees the state from before it.
STORAGE_ERRORS = (sqlite3.Error, OSError, StateDbReplacedError, CompressionSessionBusyError,
                  CompressionSessionClosedError)
SOURCE_END_REASON = "context_rollover"
STATES = ("prepared", "claimed", "installed", "started", "completed", "blocked")
_NEXT_STATES = {
    "prepared": frozenset({"claimed"}),
    "claimed": frozenset({"prepared", "installed", "blocked"}),
    "installed": frozenset({"started", "blocked"}),
    "started": frozenset({"completed"}),
}
_NONE = type(None)
_CLAIM_FIELDS: dict[str, tuple[type, ...]] = {
    "schema": (int,), "rollover_id": (str,), "source_session_id": (str,), "target_session_id": (str,),
    "profile": (str,), "generation": (int,), "handoff_digest": (str,), "state": (str,),
    "zero_progress_streak": (int,), "budget_remaining": (int,), "run_budget_seconds": (float, _NONE),
    "run_budget_started_at": (float, _NONE), "source_api_calls": (int,), "resume_api_calls": (int, _NONE),
    "claim_token": (str, _NONE), "reason": (str, _NONE), "created_at": (float,), "updated_at": (float,),
}
# Fixed at reservation: a transition may change state and bookkeeping, never what the claim is about.
_IDENTITY_FIELDS = frozenset({
    "schema", "rollover_id", "source_session_id", "target_session_id", "profile", "generation",
    "handoff_digest", "zero_progress_streak", "budget_remaining", "run_budget_seconds", "run_budget_started_at",
    "created_at",
})
_ORIGIN_FIELDS = ("schema", "rollover_id", "source_session_id", "profile", "generation", "handoff_digest",
                  "zero_progress_streak")


class RolloverRecordError(RuntimeError):
    """The session row, its model_config or a stored transition record is missing, unreadable or corrupt."""


class RolloverConflict(RuntimeError):
    """The stored state is not what the caller expected; ``record`` is the stored claim (or None)."""

    def __init__(self, message: str, record: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.record = record


@dataclass(frozen=True)
class RolloverSource:
    session_id: str
    profile_name: Optional[str]
    source: str
    cwd: Optional[str]
    ended: bool
    full_compaction_count: int
    claim: Optional[dict[str, Any]]
    origin: Optional[dict[str, Any]]


def _is_count(value: Any) -> bool:
    return type(value) is int and value >= 0


def _is_time(value: Any) -> bool:
    return value is None or math.isfinite(value)


def validate_claim(record: Any) -> dict[str, Any]:
    """A record missing a field (a pre-``run_budget_seconds`` prototype claim included) is corrupt."""
    if not (isinstance(record, dict) and set(record) == set(_CLAIM_FIELDS)
            and all(type(record[k]) in kinds for k, kinds in _CLAIM_FIELDS.items())
            and record["schema"] == RECORD_SCHEMA and record["state"] in STATES
            and all(_is_count(record[k]) for k in ("generation", "zero_progress_streak", "budget_remaining",
                                                    "source_api_calls"))
            and all(_is_time(record[k]) for k in ("run_budget_seconds", "run_budget_started_at", "created_at",
                                                   "updated_at"))
            and (record["run_budget_seconds"] is None or record["run_budget_seconds"] > 0)
            and all(record[k] for k in ("rollover_id", "source_session_id", "target_session_id", "profile"))
            and record["source_session_id"] != record["target_session_id"]):
        raise RolloverRecordError("corrupt context-rollover claim record")
    return record


def origin_for(claim: dict[str, Any]) -> dict[str, Any]:
    return {key: claim[key] for key in _ORIGIN_FIELDS}


def _validate_origin(origin: Any) -> dict[str, Any]:
    if not (isinstance(origin, dict) and set(origin) == set(_ORIGIN_FIELDS)
            and type(origin["generation"]) is int and type(origin["zero_progress_streak"]) is int
            and all(isinstance(origin[k], str) and origin[k] for k in
                    ("rollover_id", "source_session_id", "profile", "handoff_digest"))):
        raise RolloverRecordError("corrupt context-rollover origin record")
    return origin


def _config(raw: Any, session_id: str) -> dict[str, Any]:
    try:
        return strict_model_config(raw)
    except ValueError as exc:
        raise RolloverRecordError(f"unreadable model_config for {session_id}: {exc}") from None


def _count(config: dict[str, Any], session_id: str) -> int:
    value = config.get(FULL_COMPACTION_COUNT_KEY, 0)
    if not _is_count(value):
        raise RolloverRecordError(f"corrupt full-compaction count for {session_id}")
    return value


def read_source(db: Any, session_id: str) -> RolloverSource:
    """Everything a rollover decision reads about *session_id*, from one row; raises
    :class:`RolloverRecordError` whenever any of it cannot be trusted."""
    try:
        row = db.get_session(session_id) if db is not None and session_id else None
    except Exception as exc:
        raise RolloverRecordError(f"session lookup failed: {type(exc).__name__}") from exc
    if row is None:
        raise RolloverRecordError(f"session not found: {session_id}")
    config = _config(row["model_config"], session_id)
    claim, origin = config.get(ROLLOVER_CLAIM_KEY), config.get(ROLLOVER_ORIGIN_KEY)
    return RolloverSource(
        session_id=session_id, profile_name=row["profile_name"], source=row["source"], cwd=row["cwd"],
        ended=row["ended_at"] is not None, full_compaction_count=_count(config, session_id),
        claim=None if claim is None else validate_claim(claim),
        origin=None if origin is None else _validate_origin(origin),
    )


# Bound on the ancestor walk of :func:`compression_descends_from`; ``seen`` guards cycles.
_LINEAGE_CAP = 1000
# A child carrying one of these is a branch, delegate run or reset, never a continuation of its parent.
_NON_CONTINUATION_KEYS = ("_branched_from", "_delegate_from", "_reset_from")


def compression_descends_from(db: Any, session_id: str, ancestor_id: str, *, profile: str, key: str,
                              record: dict[str, Any]) -> bool:
    """True when *session_id* is *ancestor_id*, or continues it only through legacy full-compaction
    rotations of one conversation: every hop's parent ended with ``compression``, no hop is a branch,
    delegate, reset or tool child, every session on the way belongs to *profile* and carries exactly *record*
    under *key* (the copy rotation made). False for anything else; :class:`RolloverRecordError` when the store
    cannot answer."""
    current, seen = session_id, set()
    for _ in range(_LINEAGE_CAP):
        if current in seen:
            return False
        seen.add(current)
        try:
            row = db.get_session(current)
        except Exception as exc:  # health: allow BLE001 -- any store failure means "cannot answer", fail closed
            raise RolloverRecordError(f"session lookup failed: {type(exc).__name__}") from None
        if row is None or row["profile_name"] != profile:
            return False
        config = _config(row["model_config"], current)
        if config.get(key) != record:
            return False
        if current == ancestor_id:
            return True
        parent = row["parent_session_id"]
        if (not isinstance(parent, str) or not parent or row["source"] == "tool"
                or any(config.get(k) is not None for k in _NON_CONTINUATION_KEYS)):
            return False
        try:
            parent_row = db.get_session(parent)
        except Exception as exc:  # health: allow BLE001 -- as above
            raise RolloverRecordError(f"session lookup failed: {type(exc).__name__}") from None
        if parent_row is None or parent_row["ended_at"] is None or parent_row["end_reason"] != "compression":
            return False
        current = parent
    return False


def _load(conn: Any, session_id: str) -> tuple[Any, dict[str, Any]]:
    row = conn.execute(
        "SELECT model_config, profile_name, ended_at FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if row is None:
        raise RolloverRecordError(f"session not found: {session_id}")
    return row, _config(row[0], session_id)


def _store(conn: Any, session_id: str, config: dict[str, Any]) -> None:
    conn.execute("UPDATE sessions SET model_config = ? WHERE id = ?", (json.dumps(config), session_id))


def reserve(db: Any, record: dict[str, Any], *, supersede: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Write *record* (state ``prepared``) on its live source, iff the source belongs to the record's profile,
    its committed count is exactly the record's generation, and it holds no claim, or holds exactly
    *supersede* still ``prepared`` (nothing was ever invoked for it)."""
    record = validate_claim(record)
    if record["state"] != "prepared":
        raise ValueError("a reservation starts prepared")
    source_id = record["source_session_id"]

    def _do(conn):
        row, config = _load(conn, source_id)
        if row[1] != record["profile"]:
            raise RolloverConflict(f"session {source_id} belongs to another profile")
        if row[2] is not None:
            raise RolloverConflict(f"session {source_id} has ended")
        if _count(config, source_id) != record["generation"]:
            raise RolloverConflict(f"full-compaction count of {source_id} moved")
        existing = config.get(ROLLOVER_CLAIM_KEY)
        if existing is not None:
            validate_claim(existing)
            if not (supersede is not None and existing == supersede and existing["state"] == "prepared"):
                raise RolloverConflict(f"session {source_id} already holds a rollover record", existing)
        config[ROLLOVER_CLAIM_KEY] = record
        _store(conn, source_id, config)
        return record
    return db._execute_write(_do)


_MESSAGES_MARK_SQL = "SELECT COALESCE(MAX(id), 0) FROM messages WHERE session_id = ?"


def messages_mark(db: Any, session_id: str) -> int:
    """The newest transcript row id of *session_id* (0 when empty). Read it BEFORE the evidence a handoff
    is compared against: a row appended in between then fails the ``installed`` fence instead of slipping by."""
    try:
        row = db._read_one(_MESSAGES_MARK_SQL, (session_id,))
    except STORAGE_ERRORS as exc:
        raise RolloverRecordError(f"transcript lookup failed: {type(exc).__name__}") from None
    return int(row[0]) if row is not None else 0


def _fence_live_source(conn: Any, row: Any, config: dict[str, Any], claim: dict[str, Any],
                       mark: Optional[int]) -> None:
    """The source a claim is about to leave must still be exactly what the gate saw: live, same profile, same
    full-compaction generation, no compression lease and (with *mark*) no newer transcript row."""
    sid = claim["source_session_id"]
    if row[2] is not None:
        raise RolloverConflict(f"session {sid} has ended", claim)
    if row[1] != claim["profile"]:
        raise RolloverConflict(f"session {sid} belongs to another profile", claim)
    if _count(config, sid) != claim["generation"]:
        raise RolloverConflict(f"full-compaction count of {sid} moved", claim)
    if conn.execute("SELECT 1 FROM compression_locks WHERE session_id = ? AND expires_at >= ?",
                    (sid, time.time())).fetchone() is not None:
        raise RolloverConflict(f"session {sid} is being compressed", claim)
    if mark is not None and conn.execute(_MESSAGES_MARK_SQL, (sid,)).fetchone()[0] != mark:
        raise RolloverConflict(f"session {sid} has newer transcript rows", claim)


def advance(
    db: Any, source_session_id: str, rollover_id: str, from_state: str, to_state: str, *,
    updates: Optional[dict[str, Any]] = None, end_source: bool = False, messages_mark: Optional[int] = None,
) -> dict[str, Any]:
    """Compare-and-set the claim from *from_state* to *to_state* (one legal edge), merging *updates*
    (bookkeeping only). *end_source* ends the source session in the same transaction, and only if that
    source is still the one the claim was made for (see :func:`_fence_live_source`); otherwise
    :class:`RolloverConflict` and nothing is written."""
    if to_state not in _NEXT_STATES.get(from_state, ()):
        raise ValueError(f"illegal rollover transition {from_state} -> {to_state}")
    if set(updates or ()) & (_IDENTITY_FIELDS | {"state", "updated_at"}):
        raise ValueError("a transition cannot rewrite the claim's identity")

    def _do(conn):
        row, config = _load(conn, source_session_id)
        current = config.get(ROLLOVER_CLAIM_KEY)
        if current is not None:
            validate_claim(current)
        if current is None or current["rollover_id"] != rollover_id or current["state"] != from_state:
            raise RolloverConflict(f"rollover {rollover_id} is not {from_state}", current)
        if end_source:
            _fence_live_source(conn, row, config, current, messages_mark)
        new = validate_claim({**current, **(updates or {}), "state": to_state,
                              "updated_at": max(time.time(), current["updated_at"])})
        config[ROLLOVER_CLAIM_KEY] = new
        _store(conn, source_session_id, config)
        if end_source and db._end_and_bump(
                conn, "UPDATE sessions SET ended_at = ?, end_reason = ? WHERE id = ? AND ended_at IS NULL",
                (time.time(), SOURCE_END_REASON, source_session_id), source_session_id, SOURCE_END_REASON) != 1:
            raise RolloverConflict(f"session {source_session_id} could not be ended", current)
        return new
    return db._execute_write(_do)


def _check_target(conn: Any, claim: dict[str, Any], origin: dict[str, Any]) -> bool:
    """False when the target row does not exist yet; True when it is exactly this claim's untouched fresh
    root; otherwise :class:`RolloverConflict` (an id collision, or a target something already wrote into)."""
    target = claim["target_session_id"]
    row = conn.execute(
        "SELECT parent_session_id, profile_name, model_config, ended_at FROM sessions WHERE id = ?",
        (target,)).fetchone()
    if row is None:
        return False
    config = _config(row[2], target)
    used = conn.execute("SELECT 1 FROM messages WHERE session_id = ? LIMIT 1", (target,)).fetchone()
    if (row[0] is not None or row[1] != claim["profile"] or row[3] is not None or used is not None
            or config.get(ROLLOVER_ORIGIN_KEY) != origin
            or FULL_COMPACTION_COUNT_KEY in config or ROLLOVER_CLAIM_KEY in config):
        raise RolloverConflict(f"target session {target} is not this rollover's fresh root", claim)
    return True


def bind_target(
    db: Any, claim: dict[str, Any], *, source: str, model: Optional[str], cwd: Optional[str],
    model_config: dict[str, Any],
) -> dict[str, Any]:
    """Create the claim's target as a fresh ROOT session (no parent, no ledger, no claim) carrying the
    origin, or accept the one this claim created on an earlier attempt that never ran. Returns the origin."""
    claim = validate_claim(claim)
    origin = origin_for(claim)
    if not db._execute_write(lambda conn: _check_target(conn, claim, origin)):
        fresh = {k: v for k, v in model_config.items()
                 if k not in (FULL_COMPACTION_COUNT_KEY, ROLLOVER_CLAIM_KEY, ROLLOVER_ORIGIN_KEY)}
        db.create_session(claim["target_session_id"], source, model=model, cwd=cwd, profile_name=claim["profile"],
                          model_config={**fresh, ROLLOVER_ORIGIN_KEY: origin})
        if not db._execute_write(lambda conn: _check_target(conn, claim, origin)):
            raise RolloverRecordError(f"target session {claim['target_session_id']} was not created")
    return origin
