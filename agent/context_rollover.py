"""Context rollover: after ``compression.rollover_after`` committed full compactions, an eligible host
resumes the task in a fresh session + fresh agent instead of compacting a summary of summaries again.

This module owns the policy (setting, host eligibility, due check), the durable count readback, and the
deterministic evidence handoff the fresh agent starts from. The count itself is written by the compaction
transaction (``hermes_state_compression.FULL_COMPACTION_COUNT_KEY``), never here. The handoff carries
references and the user's own words, not conclusions: no model call, no credentials, no approvals.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import secrets
import stat
from itertools import pairwise
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import urlsplit

from hermes_state_compression import FULL_COMPACTION_COUNT_KEY, strict_model_config

HANDOFF_SCHEMA = 1
HANDOFF_DIR_NAME = "context_rollover"
ROLLOVER_SURFACES = frozenset({"cli", "tui"})
RESUME_PROMPT_PREFIX = "[Context rollover "
_MAX_HANDOFF_BYTES = 8 * 1024 * 1024
_ROLLOVER_ID_RE = re.compile(r"[A-Za-z0-9_-]{8,64}")
_ROUTE_STR_KEYS = ("model", "provider", "base_url", "api_mode", "service_tier", "cwd")
_ROUTE_LIST_KEYS = ("enabled_toolsets", "disabled_toolsets")
_REASONING_FIELDS = {"enabled": bool, "effort": str}
_ARTIFACT_ARG_KEYS = ("path", "file_path")
_USER_ORIGINS = ("user", "rollover_resume")
_NON_TEXT_PART = "[non-text content not carried]"
_LIMITS = (
    "Tool-call references show that a result was recorded, not that the task step succeeded.",
    "The compaction summary is a lossy model-written digest of earlier turns; it may be incomplete or wrong.",
    "User messages are quoted verbatim in order; a later message may correct or restrict an earlier one.",
    "Unresolved tool calls have an unknown outcome; do not re-run side effects without checking first.",
    "No approvals, permissions or credentials carry over; the new session asks again where policy requires.",
)
_BODY_KEYS = frozenset({
    "rollover_id", "profile", "source_session_id", "full_compaction_count", "created_at", "user_request",
    "user_messages", "todos", "completed_tool_calls", "unresolved_tool_calls", "artifact_refs",
    "compaction_summary", "route", "limits",
})


class RolloverStateError(RuntimeError):
    """The durable count is missing, unreadable or corrupt: never roll over on it."""


class HandoffError(RuntimeError):
    """The handoff cannot be written or trusted (path, permissions, digest, binding, schema)."""


@dataclasses.dataclass(frozen=True)
class RolloverHost:
    """A host's explicit declaration that it can swap in a fresh agent + session and resume.

    Config opt-in alone never makes an agent eligible: the CLI/TUI host constructs this and hands it to
    the agent; gateway platforms, cron, delegated children and forks never do.
    """

    surface: str
    resumes_in_fresh_agent: bool


def rollover_after_setting(compression_cfg: Any) -> tuple[int, Optional[str]]:
    """``(threshold, error)`` from the ``compression`` config section; 0 = off. Anything other than a
    non-negative int (bools, floats, strings) is off with a reason, never a silent enable."""
    value = compression_cfg.get("rollover_after") if isinstance(compression_cfg, Mapping) else None
    if value is None:
        return 0, None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0, f"compression.rollover_after must be 0 (off) or a positive integer, got {value!r}; rollover is off"
    return value, None


def rollover_due(count: int, threshold: int) -> bool:
    return threshold > 0 and count >= threshold


def rollover_eligible(host: Optional[RolloverHost], *, delegated: bool) -> bool:
    return (host is not None and host.resumes_in_fresh_agent and not delegated
            and host.surface in ROLLOVER_SURFACES)


def read_full_compaction_count(session_db: Any, session_id: str) -> int:
    """Committed full compactions recorded for *session_id*; raises :class:`RolloverStateError` when
    the store, the row, the stored config or the value cannot be trusted. Only a NULL/blank (legacy)
    config or an object without the key reads as zero."""
    if session_db is None or not session_id:
        raise RolloverStateError("no session store or session id")
    try:
        session = session_db.get_session(session_id)
    except Exception as exc:
        raise RolloverStateError(f"session lookup failed: {type(exc).__name__}") from exc
    if session is None:
        raise RolloverStateError(f"session not found: {session_id}")
    try:
        config = strict_model_config(session.get("model_config"))
    except ValueError as exc:
        raise RolloverStateError(f"unreadable model_config for {session_id}: {exc}") from None
    value = config.get(FULL_COMPACTION_COUNT_KEY, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RolloverStateError(f"corrupt full-compaction count for {session_id}")
    return value


def _check_rollover_id(rollover_id: Any) -> str:
    if not isinstance(rollover_id, str) or not _ROLLOVER_ID_RE.fullmatch(rollover_id):
        raise HandoffError("unsafe rollover id")
    return rollover_id


def _route_evidence(route: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """The route choices the fresh agent keeps, rebuilt field by field from known types; other top-level
    keys (keys, tokens, clients) are not route choices and are left behind. A known field with the wrong
    type, a reasoning_config with fields beyond enabled/effort, or a base URL carrying userinfo or a query
    is refused rather than stripped: a stripped value may no longer select the same route."""
    if route is None:
        return {}
    if not isinstance(route, Mapping):
        raise HandoffError("route must be a mapping")
    kept: dict[str, Any] = {}
    for key in _ROUTE_STR_KEYS:
        value = route.get(key)
        if value in (None, ""):
            continue
        if not isinstance(value, str):
            raise HandoffError(f"route field {key} must be a string")
        kept[key] = value
    for key in _ROUTE_LIST_KEYS:
        value = route.get(key)
        if value is None:
            continue
        if isinstance(value, (set, frozenset)):
            value = sorted(value) if all(isinstance(item, str) for item in value) else None
        if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
            raise HandoffError(f"route field {key} must be a list of strings")
        kept[key] = list(value)
    reasoning = route.get("reasoning_config")
    if reasoning is not None:
        if not isinstance(reasoning, Mapping) or not set(reasoning) <= set(_REASONING_FIELDS):
            raise HandoffError("route reasoning_config carries unsupported fields")
        rebuilt = {}
        for field, kind in _REASONING_FIELDS.items():
            if reasoning.get(field) is None:
                continue
            if type(reasoning[field]) is not kind:
                raise HandoffError(f"route reasoning_config.{field} has the wrong type")
            rebuilt[field] = reasoning[field]
        kept["reasoning_config"] = rebuilt
    base_url = kept.get("base_url")
    if base_url is not None:
        parts = urlsplit(base_url)
        if parts.username or parts.password or parts.query or parts.fragment:
            raise HandoffError("base_url carries credentials or a query; refusing to serialize it")
    return kept


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and part.get("type") in (None, "text", "input_text") \
                and isinstance(part.get("text"), str):
            parts.append(part["text"])
        else:
            parts.append(_NON_TEXT_PART)
    return "\n".join(parts)


def _user_messages(messages: list) -> list[dict]:
    """The user's own messages, verbatim and in transcript order. Standalone compaction summaries and
    internal scaffolding rows (nudges, todo/background injections) are not the user's words and stay out;
    a summary carrier keeps only its live user part. An earlier rollover's resume prompt is host-written
    and labelled as such."""
    from agent.context_compressor import ContextCompressor, is_compaction_summary_message

    found: list[dict] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        if is_compaction_summary_message(message):
            live = ContextCompressor._strip_context_summary_handoff_message(message)
            if live is None:
                continue
            text = _content_text(live.get("content"))
        elif ContextCompressor._is_synthetic_compression_user_turn(message):
            continue
        else:
            text = _content_text(message.get("content"))
        if text.strip():
            origin = "rollover_resume" if text.startswith(RESUME_PROMPT_PREFIX) else "user"
            found.append({"index": index, "origin": origin, "text": text})
    return found


def _tool_call_evidence(messages: list) -> tuple[list[dict], list[dict], list[str]]:
    """(completed, unresolved, artifact_refs) from assistant tool calls and the tool results present."""
    answered = {m.get("tool_call_id") for m in messages if isinstance(m, dict) and m.get("role") == "tool"}
    completed: list[dict] = []
    unresolved: list[dict] = []
    artifacts: set[str] = set()
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or ():
            function = call.get("function") or {}
            ref = {"tool_call_id": str(call.get("id") or ""), "tool_name": str(function.get("name") or "")}
            if ref["tool_call_id"] not in answered:
                unresolved.append(ref)
                continue
            completed.append(ref)
            try:
                args = json.loads(function.get("arguments") or "{}")
            except (TypeError, ValueError):
                continue
            if isinstance(args, dict):
                artifacts.update(str(args[key]) for key in _ARTIFACT_ARG_KEYS if isinstance(args.get(key), str))
    return completed, unresolved, sorted(artifacts)


def _latest_summary_text(messages: list) -> Optional[str]:
    """Body of the latest compaction summary, without the merged prior tail, prefix or end marker, so the
    resume prompt does not itself read as a summary row."""
    from agent.context_compressor import ContextCompressor, is_compaction_summary_message
    for message in reversed(messages):
        if is_compaction_summary_message(message) and isinstance(message.get("content"), str):
            return ContextCompressor._strip_summary_prefix(message["content"])
    return None


def _digest(body: dict) -> str:
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _require(ok: bool, field: str) -> None:
    if not ok:
        raise HandoffError(f"handoff schema violation: {field}")


def _is_count(value: Any) -> bool:
    return type(value) is int and value >= 1


def _records(value: Any, keys: tuple[str, ...]) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, dict) and set(item) == set(keys) and all(isinstance(item[k], str) for k in keys)
        for item in value)


def _validate_body(body: Any) -> dict[str, Any]:
    """Structural check of a handoff body: exact field set, exact types, values the renderer relies on.
    Integrity (digest) and binding (profile/session/filename) are checked separately."""
    _require(isinstance(body, dict) and set(body) == _BODY_KEYS, "body fields")
    _check_rollover_id(body["rollover_id"])
    for key in ("profile", "source_session_id"):
        _require(isinstance(body[key], str) and bool(body[key]), key)
    _require(_is_count(body["full_compaction_count"]), "full_compaction_count")
    created_at = body["created_at"]
    _require(type(created_at) is float and math.isfinite(created_at) and created_at >= 0, "created_at")
    _require(isinstance(body["user_request"], str), "user_request")
    users = body["user_messages"]
    _require(isinstance(users, list) and all(
        isinstance(m, dict) and set(m) == {"index", "origin", "text"} and type(m["index"]) is int
        and m["index"] >= 0 and m["origin"] in _USER_ORIGINS and isinstance(m["text"], str) for m in users),
        "user_messages")
    _require(all(a["index"] < b["index"] for a, b in pairwise(users)), "user_messages order")
    _require(_records(body["todos"], ("id", "content", "status")), "todos")
    for key in ("completed_tool_calls", "unresolved_tool_calls"):
        _require(_records(body[key], ("tool_call_id", "tool_name")), key)
    for key in ("artifact_refs", "limits"):
        _require(isinstance(body[key], list) and all(isinstance(item, str) for item in body[key]), key)
    summary = body["compaction_summary"]
    _require(isinstance(summary, dict) and set(summary) == {"text", "source_session_id", "full_compaction_count"}
             and (summary["text"] is None or isinstance(summary["text"], str))
             and summary["source_session_id"] == body["source_session_id"]
             and _is_count(summary["full_compaction_count"])
             and summary["full_compaction_count"] == body["full_compaction_count"], "compaction_summary")
    route = body["route"]
    _require(isinstance(route, dict), "route")
    _require(_route_evidence(route) == route, "route")
    return body


def _validate_handoff(handoff: Any) -> dict[str, Any]:
    """Envelope + digest + body schema; returns the body. Raises :class:`HandoffError`, never mutates."""
    _require(isinstance(handoff, dict) and set(handoff) == {"schema", "body", "digest"}, "envelope fields")
    _require(type(handoff["schema"]) is int and handoff["schema"] == HANDOFF_SCHEMA, "schema version")
    body = handoff["body"]
    if not isinstance(body, dict) or handoff["digest"] != _digest(body):
        raise HandoffError("handoff failed integrity check")
    return _validate_body(body)


def build_handoff(
    *, rollover_id: str, profile: str, source_session_id: str, full_compaction_count: int,
    user_request: str, messages: list, todos: Optional[list] = None,
    route: Optional[Mapping[str, Any]] = None, created_at: float,
) -> dict[str, Any]:
    """Deterministic evidence envelope ``{"schema", "body", "digest"}`` for a rollover."""
    if not _is_count(full_compaction_count):
        raise HandoffError("full_compaction_count must be a positive integer")
    if isinstance(created_at, bool) or not isinstance(created_at, (int, float)):
        raise HandoffError("created_at must be a number")
    completed, unresolved, artifacts = _tool_call_evidence(messages)
    body = {
        "rollover_id": _check_rollover_id(rollover_id),
        "profile": str(profile),
        "source_session_id": str(source_session_id),
        "full_compaction_count": full_compaction_count,
        "created_at": float(created_at),
        "user_request": str(user_request),
        "user_messages": _user_messages(messages),
        "todos": [{"id": str(t.get("id", "")), "content": str(t.get("content", "")),
                   "status": str(t.get("status", ""))} for t in (todos or ()) if isinstance(t, dict)],
        "completed_tool_calls": completed,
        "unresolved_tool_calls": unresolved,
        "artifact_refs": artifacts,
        "compaction_summary": {"text": _latest_summary_text(messages), "source_session_id": str(source_session_id),
                               "full_compaction_count": full_compaction_count},
        "route": _route_evidence(route),
        "limits": list(_LIMITS),
    }
    handoff = {"schema": HANDOFF_SCHEMA, "body": body, "digest": _digest(body)}
    _validate_handoff(handoff)
    return handoff


def render_resume_prompt(handoff: dict[str, Any]) -> str:
    """The fresh agent's first user message: the original request, the user's later messages verbatim,
    and evidence to verify first."""
    body = _validate_handoff(handoff)
    lines = [
        f"{RESUME_PROMPT_PREFIX}{body['rollover_id']}] This is a fresh session continuing session "
        f"{body['source_session_id']} after {body['full_compaction_count']} context compactions.",
        "Before doing anything else, verify the current state from real files and sources; resume only the "
        "work that is not actually done. The evidence below is a set of references, not proof of completion.",
        "", "Original request:", body["user_request"], "",
        "User messages in the source session (verbatim, oldest first; later ones take precedence):",
    ]
    for message in body["user_messages"]:
        label = "user" if message["origin"] == "user" else "earlier rollover handoff, host-written"
        lines += [f"--- message {message['index']} ({label}) ---", message["text"]]
    if not body["user_messages"]:
        lines.append("- (none recorded)")
    lines += ["", "Todo state at rollover:"]
    lines += [f"- [{t['status']}] {t['content']}" for t in body["todos"]] or ["- (none recorded)"]
    lines += ["", "Tool calls with a recorded result:"]
    lines += [f"- {c['tool_name']} ({c['tool_call_id']})" for c in body["completed_tool_calls"]] or ["- (none)"]
    lines += ["", "Tool calls with an UNKNOWN outcome (check before any retry):"]
    lines += [f"- {c['tool_name']} ({c['tool_call_id']})" for c in body["unresolved_tool_calls"]] or ["- (none)"]
    lines += ["", "Artifacts referenced:"] + ([f"- {p}" for p in body["artifact_refs"]] or ["- (none)"])
    summary = body["compaction_summary"]["text"]
    lines += ["", "Latest compaction summary (lossy, model-written; not the user's words):",
              summary or "(none available)"]
    lines += ["", "Limits:"] + [f"- {limit}" for limit in body["limits"]]
    prompt = "\n".join(lines)
    # A user's own literal marker text could still make the row read as a compaction summary; the user's
    # words are never rewritten, so that ambiguity fails closed.
    from agent.context_compressor import is_compaction_summary_message
    if is_compaction_summary_message({"role": "user", "content": prompt}):
        raise HandoffError("resume prompt would be misread as a compaction summary")
    return prompt


_DIR_FD = hasattr(os, "O_DIRECTORY") and os.open in os.supports_dir_fd and os.link in os.supports_dir_fd


def _check_private(st: os.stat_result, what: str) -> None:
    if hasattr(os, "getuid") and (st.st_uid != os.getuid() or stat.S_IMODE(st.st_mode) & 0o077):
        raise HandoffError(f"handoff {what} is not private")


def _open_private_dir(directory: Path) -> Optional[int]:
    """Open the handoff directory without following a symlink and check the opened directory itself is
    private; the returned fd anchors every later entry operation (``None`` where dir fds are unsupported,
    then the check is an lstat)."""
    if not _DIR_FD:
        st = os.lstat(directory)
        if not stat.S_ISDIR(st.st_mode):
            raise HandoffError("handoff directory is not a real directory")
        _check_private(st, "directory")
        return None
    # A symlinked or non-directory entry fails here (ELOOP / ENOTDIR); callers turn that into HandoffError.
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
    try:
        st = os.fstat(fd)
        if not stat.S_ISDIR(st.st_mode):
            raise HandoffError("handoff directory is not a real directory")
        _check_private(st, "directory")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _entry(directory: Path, name: str, dir_fd: Optional[int]) -> tuple[str, dict[str, Any]]:
    return (name, {"dir_fd": dir_fd}) if dir_fd is not None else (str(directory / name), {})


def handoff_path(home: Path, rollover_id: str) -> Path:
    return Path(home) / HANDOFF_DIR_NAME / f"{_check_rollover_id(rollover_id)}.json"


def _publish_no_clobber(directory: Path, name: str, payload: bytes, dir_fd: Optional[int]) -> None:
    """Write *payload* to a private temp entry, fsync it, then hard-link it to *name*: the link fails if
    anything (file, symlink, FIFO) already sits there, so exactly one writer publishes and an existing
    handoff is never replaced. The temp entry is always removed; the target never is."""
    temp_name, temp_kw = _entry(directory, f".{name}.{secrets.token_hex(8)}.tmp", dir_fd)
    target_name, _target_kw = _entry(directory, name, dir_fd)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    fd = os.open(temp_name, flags, 0o600, **temp_kw)
    try:
        try:
            view = memoryview(payload)
            while view:
                view = view[os.write(fd, view):]
            os.fsync(fd)
        finally:
            os.close(fd)
        link_kw = {"src_dir_fd": dir_fd, "dst_dir_fd": dir_fd} if dir_fd is not None else {}
        try:
            os.link(temp_name, target_name, **link_kw)
        except FileExistsError:
            raise HandoffError(f"handoff already exists: {directory / name}") from None
    finally:
        try:
            os.unlink(temp_name, **temp_kw)
        except OSError:
            pass
    if dir_fd is not None:
        os.fsync(dir_fd)


def save_handoff(home: Path, handoff: dict[str, Any]) -> Path:
    """Publish *handoff* under the profile *home* (0600 file in a 0700 dir) exactly once; never overwrites."""
    from hermes_constants import mkdir_under_hermes_home

    body = _validate_handoff(handoff)
    path = handoff_path(home, body["rollover_id"])
    payload = json.dumps(handoff, sort_keys=True, ensure_ascii=False).encode("utf-8")
    if len(payload) > _MAX_HANDOFF_BYTES:
        raise HandoffError("handoff exceeds the size bound")
    try:
        if not os.path.lexists(path.parent):
            mkdir_under_hermes_home(path.parent)
            os.chmod(path.parent, 0o700)
        dir_fd = _open_private_dir(path.parent)
        try:
            _publish_no_clobber(path.parent, path.name, payload, dir_fd)
        finally:
            if dir_fd is not None:
                os.close(dir_fd)
    except OSError as exc:
        raise HandoffError(f"could not save handoff: {exc}") from exc
    return path


def _read_bounded(fd: int) -> bytes:
    chunks: list[bytes] = []
    remaining = _MAX_HANDOFF_BYTES + 1
    while remaining > 0:
        chunk = os.read(fd, min(remaining, 1 << 20))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    if remaining <= 0:
        raise HandoffError("handoff exceeds the size bound")
    return b"".join(chunks)


def load_handoff(path: Path, *, profile: str, source_session_id: str) -> dict[str, Any]:
    """Read and verify a handoff: private regular file, intact digest, valid schema, and bound to this
    profile, this source session and its own filename. Any doubt raises :class:`HandoffError`.
    The entry is opened non-blocking without following symlinks, so a FIFO or device is refused
    after ``fstat`` instead of stalling the rollover boundary."""
    path = Path(path)
    try:
        dir_fd = _open_private_dir(path.parent)
        try:
            name, kw = _entry(path.parent, path.name, dir_fd)
            flags = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
                     | getattr(os, "O_BINARY", 0))
            fd = os.open(name, flags, **kw)
            try:
                st = os.fstat(fd)
                if not stat.S_ISREG(st.st_mode) or st.st_size > _MAX_HANDOFF_BYTES:
                    raise HandoffError(f"handoff is not a bounded regular file: {path}")
                _check_private(st, "file")
                raw = _read_bounded(fd)
            finally:
                os.close(fd)
        finally:
            if dir_fd is not None:
                os.close(dir_fd)
        data = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise HandoffError(f"could not read handoff {path}: {type(exc).__name__}") from None
    body = _validate_handoff(data)
    if (body["profile"] != profile or body["source_session_id"] != source_session_id
            or path.name != f"{body['rollover_id']}.json"):
        raise HandoffError(f"handoff is bound to another profile, session or rollover: {path}")
    return data
