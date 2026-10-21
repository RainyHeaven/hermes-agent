"""Context rollover controller: turns a gated ``context_rollover`` turn result into a fresh root session and a
fresh agent that resumes the task exactly once.

The CLI/TUI host supplies a :class:`RolloverHostAdapter`; this module never builds an agent, swaps host state
or runs a turn itself. Order: verify the request against the durable source -> stage the deterministic
handoff (build, private no-clobber save, verified readback) -> reserve one rollover id + one target id
(``prepared``; a retry reuses it only for the same request and evidence) -> claim it (compare-and-set: one
controller wins) -> the host builds the fresh agent -> verify its identity, store and route -> bind the fresh
root target -> recheck stop/input -> commit ``installed`` and end the source in one transaction, only if the
source is still exactly what the handoff was built from -> the host installs the fresh agent -> the host
retires the old agent -> recheck both agents and the host -> ``started`` -> the host runs the resume turn,
authorized for exactly this claim -> ``completed``. Nothing before ``started`` issues a model request, and a
crash after ``claimed`` is left for a person (``hermes_state_rollover``), never replayed.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
import secrets
import time
from pathlib import Path
from typing import Any, Callable, NoReturn, Optional

from agent.context_rollover import (
    RESUME_PROMPT_PREFIX, HandoffError, _content_text, _route_evidence, build_handoff, handoff_path,
    load_handoff, render_resume_prompt, rollover_due, save_handoff,
)
from agent.context_rollover_gate import (
    ARM_ATTR, RESUME_ATTR, HostState, agent_input_block, claim_block, compression_lock_block, host_state_block,
    take_agent_input,
)
from agent.iteration_budget import IterationBudget
from hermes_state_ids import new_session_id
from hermes_state_rollover import (
    RECORD_SCHEMA, STORAGE_ERRORS, RolloverConflict, RolloverRecordError, advance, bind_target, messages_mark,
    read_source, reserve,
)

logger = logging.getLogger(__name__)

# An unreadable/unwritable store: the outcome says ``state_unavailable``, never the raw error text.
_STATE_ERRORS = (RolloverRecordError, *STORAGE_ERRORS)

BUDGET_CARRY_ATTR = "_context_rollover_budget_carry"
# A fresh segment that rolls over again before making a single model call made no progress; the second such
# rollover in a row stops instead of looping on zero-API handoffs.
_MAX_ZERO_PROGRESS_ROLLOVERS = 2
_LATEST_INPUT_HEADER = ("Latest user message, sent while this rollover was starting "
                        "(it takes precedence over everything above):")


@dataclasses.dataclass(frozen=True)
class FreshAgentSpec:
    """What the host's factory must honour: a NEW agent built through normal initialization (fresh
    instructions and memory, the host's own credentials) on *session_id*, using *session_db* (the host's
    shared store, never a new one), exactly the *route* choices (no default-model fallback; ``cwd`` and
    ``service_tier`` included when the source had them) and exactly *run_budget_seconds* as its wall-clock
    cap per turn (None = off). The controller checks all of it before install and again before the resume."""

    session_id: str
    session_db: Any
    route: dict[str, Any]
    rollover_id: str
    source_session_id: str
    profile: str
    run_budget_seconds: Optional[float]


@dataclasses.dataclass(frozen=True)
class RolloverHostAdapter:
    """The host's side of a rollover. ``install`` swaps the host onto the new agent (session id, empty
    history, callbacks, approval scope, re-arm); ``retire`` releases an agent without closing the host's
    SessionDB (also used to discard a built-but-unused agent, and the candidate of an ``install`` that raised,
    whether or not the host had already adopted it); ``run_turn`` runs one user turn the host's normal way and
    returns its result dict."""

    host_state: Callable[[], HostState]
    make_agent: Callable[[FreshAgentSpec], Any]
    install: Callable[[Any, RolloverOutcome], None]
    retire: Callable[[Any], None]
    run_turn: Callable[[Any, str], dict]


@dataclasses.dataclass
class RolloverOutcome:
    """``status``: ``resumed`` (the resume turn ran once), ``deferred`` (installed, resume held back for
    stop/new input: call :func:`resume_deferred`), ``already_completed`` or ``blocked`` (``reason``; no model
    request was made by this call). Call counts are the agents' own: ``source_api_calls`` is the source agent's
    session total, ``resume_api_calls`` the resume turn's."""

    status: str
    reason: Optional[str] = None
    rollover_id: Optional[str] = None
    source_session_id: Optional[str] = None
    target_session_id: Optional[str] = None
    generation: Optional[int] = None
    handoff_path: Optional[Path] = None
    agent: Any = None
    source_api_calls: int = 0
    resume_api_calls: Optional[int] = None
    budget_remaining: Optional[int] = None
    resume_prompt: Optional[str] = dataclasses.field(default=None, repr=False)
    resume_result: Optional[dict[str, Any]] = dataclasses.field(default=None, repr=False)
    # A message the user sent the old agent during the transition, consumed from it exactly once. ``deferred``:
    # the host passes it to :func:`resume_deferred` as ``user_text``; ``blocked``: the host keeps serving the
    # source and runs it as the user's next turn. Never dropped either way.
    pending_input: Optional[str] = dataclasses.field(default=None, repr=False)
    # What the installed agent was verified against; rechecked before the resume turn starts.
    fresh_spec: Optional[FreshAgentSpec] = dataclasses.field(default=None, repr=False)


class _Blocked(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def observed_cwd(agent: Any) -> Optional[str]:
    """The directory *agent* actually works in, canonical: its own session cwd, else the scoped
    ``TERMINAL_CWD`` (the order ``agent_init`` resolves its working dir in); None when neither is set.
    Always read from the agent itself, never from a stored row or the value it is being compared against."""
    from agent.runtime_cwd import scope_terminal_cwd

    try:
        raw = getattr(agent, "session_cwd", None) or scope_terminal_cwd() or None
        if raw is None:
            return None
        if not isinstance(raw, str):
            raise TypeError(type(raw).__name__)
        return os.path.realpath(os.path.expanduser(raw))
    except Exception:  # health: allow BLE001 -- a refusing terminal scope raises its own types; an unknown cwd is not a matching one
        raise HandoffError("the agent's working directory cannot be determined") from None


def route_snapshot(agent: Any) -> dict[str, Any]:
    """The route choices a fresh agent must reproduce, as observed on *agent* (no credentials; see
    ``_route_evidence``)."""
    return _route_evidence({
        "model": agent.model, "provider": agent.provider, "base_url": agent.base_url, "api_mode": agent.api_mode,
        "service_tier": getattr(agent, "service_tier", None), "reasoning_config": agent.reasoning_config,
        "enabled_toolsets": agent.enabled_toolsets, "disabled_toolsets": agent.disabled_toolsets,
        "cwd": observed_cwd(agent),
    })


def _budget_seconds(value: Any) -> Optional[float]:
    """A wall-clock cap as the claim stores it: finite positive seconds, or None (off). ValueError otherwise."""
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("run_budget_seconds must be a finite positive number or None")
    return float(value)


def _source_run_budget(source_agent: Any) -> tuple[Optional[float], Optional[float]]:
    """``(cap, started_at)`` of the source's logical turn; the start is only kept while a cap is on."""
    try:
        cap = _budget_seconds(getattr(source_agent, "run_budget_seconds", None))
    except ValueError:
        raise _Blocked("run_budget_invalid") from None
    started = getattr(source_agent, "_run_budget_started_at", None)
    if cap is None or started is None:
        return cap, None
    if type(started) not in (int, float) or not math.isfinite(started):
        raise _Blocked("run_budget_invalid")
    return cap, float(started)


def apply_budget_carry(agent: Any) -> None:
    """One-shot, at the resume turn's start: continue the source turn's remaining iteration and wall-clock
    budget instead of a fresh full one. Later user turns are untouched."""
    carry = getattr(agent, BUDGET_CARRY_ATTR, None)
    if carry is None:
        return
    setattr(agent, BUDGET_CARRY_ATTR, None)
    agent.iteration_budget = IterationBudget(carry["remaining"])
    if agent._run_budget_started_at is not None and carry["run_budget_started_at"] is not None:
        agent._run_budget_started_at = carry["run_budget_started_at"]


def _verify_request(source_agent: Any, gate_result: Any) -> tuple[Any, Any, Any]:
    """``(arm, db, source)`` for a genuine due request from *source_agent*'s gated turn."""
    arm = getattr(source_agent, ARM_ATTR, None)
    request = gate_result.get("context_rollover") if isinstance(gate_result, dict) else None
    if arm is None or not isinstance(request, dict) or request.get("status") != "due":
        raise _Blocked("not_a_rollover_request")
    sid = source_agent.session_id
    if (request.get("source_session_id") != sid or gate_result.get("session_id") != sid
            or request.get("profile") != arm.profile):
        raise _Blocked("request_mismatch")
    db = source_agent._session_db
    try:
        source = read_source(db, sid)
    except RolloverRecordError:
        raise _Blocked("state_unavailable") from None
    return arm, db, source


def _load_verified(home: Path, claim: dict[str, Any]) -> dict[str, Any]:
    try:
        handoff = load_handoff(handoff_path(home, claim["rollover_id"]), profile=claim["profile"],
                               source_session_id=claim["source_session_id"])
    except HandoffError:
        raise _Blocked("handoff_invalid") from None
    if handoff["digest"] != claim["handoff_digest"] or handoff["body"]["full_compaction_count"] != claim["generation"]:
        raise _Blocked("handoff_invalid")
    return handoff


def _current_request(home: Path, source: Any, gate_result: dict[str, Any]) -> str:
    """The request the gated turn was serving: its own user message, or, for a resume turn, the request the
    previous handoff carried (a host-written resume prompt is not the user's task)."""
    messages, index = gate_result.get("messages"), gate_result.get("current_turn_user_idx")
    if not (isinstance(messages, list) and type(index) is int and 0 <= index < len(messages)
            and isinstance(messages[index], dict) and messages[index].get("role") == "user"):
        raise _Blocked("current_request_unknown")
    text = _content_text(messages[index].get("content"))
    if not text.startswith(RESUME_PROMPT_PREFIX):
        return text
    if source.origin is None:
        raise _Blocked("current_request_unknown")
    return _load_verified(home, source.origin)["body"]["user_request"]


def _build(source_agent, arm, db, home, source, gate_result, *, rollover_id: str, created_at: float) -> dict:
    """The handoff for the gated request as the durable transcript, todos and route stand right now."""
    try:
        evidence = db.get_messages_as_conversation(source.session_id, include_ancestors=True, include_compacted=True)
    except Exception:  # health: allow BLE001 -- a transcript that cannot be read or decoded is no evidence; its text never reaches the outcome
        raise _Blocked("state_unavailable") from None
    todo_store = getattr(source_agent, "_todo_store", None)
    try:
        todos = todo_store.read() if todo_store is not None else []
    except Exception:  # the handoff must carry the real todo state or nothing
        logger.warning("context rollover: reading the todo list failed", exc_info=True)
        raise _Blocked("todo_unavailable") from None
    try:
        return build_handoff(
            rollover_id=rollover_id, profile=arm.profile, source_session_id=source.session_id,
            full_compaction_count=source.full_compaction_count, user_request=_current_request(home, source, gate_result),
            messages=evidence, todos=todos, route=route_snapshot(source_agent), created_at=created_at,
        )
    except HandoffError:
        raise _Blocked("handoff_invalid") from None


def _same_snapshot(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Same request and evidence: every body field but the per-reservation id and timestamp. The user rows
    carry their transcript positions, so identical text sent again in a new turn is a different snapshot."""
    def snapshot(handoff):
        return {k: v for k, v in handoff["body"].items() if k not in ("rollover_id", "created_at")}
    return snapshot(a) == snapshot(b)


def _stage_reservation(source_agent, arm, db, home, source, gate_result) -> tuple[dict, dict]:
    """Build, publish and read back the handoff, then reserve ids for it. Returns ``(claim, handoff)``."""
    remaining = source_agent.iteration_budget.remaining
    if remaining <= 0:
        raise _Blocked("budget_exhausted")
    progress = int(getattr(source_agent, "session_api_calls", 0) or 0)
    streak = 0 if source.origin is None or progress else source.origin["zero_progress_streak"] + 1
    if streak >= _MAX_ZERO_PROGRESS_ROLLOVERS:
        raise _Blocked("no_progress")
    rollover_id = f"ro_{secrets.token_hex(12)}"
    handoff = _build(source_agent, arm, db, home, source, gate_result, rollover_id=rollover_id,
                     created_at=time.time())
    try:
        path = save_handoff(home, handoff)
        if load_handoff(path, profile=arm.profile, source_session_id=source.session_id) != handoff:
            raise _Blocked("handoff_invalid")
    except HandoffError:
        raise _Blocked("handoff_invalid") from None
    now = time.time()
    run_cap, run_started = _source_run_budget(source_agent)
    record = {
        "schema": RECORD_SCHEMA, "rollover_id": rollover_id, "source_session_id": source.session_id,
        "target_session_id": new_session_id(), "profile": arm.profile, "generation": source.full_compaction_count,
        "handoff_digest": handoff["digest"], "state": "prepared", "zero_progress_streak": streak,
        "budget_remaining": remaining, "run_budget_seconds": run_cap, "run_budget_started_at": run_started,
        "source_api_calls": progress, "resume_api_calls": None, "claim_token": None, "reason": None,
        "created_at": now, "updated_at": now,
    }
    try:
        return reserve(db, record, supersede=source.claim), handoff
    except RolloverConflict as exc:
        stored = exc.record
        if stored is not None and stored["state"] == "prepared" and stored["generation"] == source.full_compaction_count:
            joined = _load_verified(home, stored)
            if _same_snapshot(joined, handoff):
                return stored, joined  # an identical reservation won the race; join it
            raise _Blocked("source_changed") from None
        raise _Blocked(claim_block(dataclasses.replace(source, claim=stored), db) or "source_changed") from None
    except _STATE_ERRORS:
        raise _Blocked("state_unavailable") from None


def _reserve_or_recover(source_agent, arm, db, home, source, gate_result) -> tuple[dict, dict]:
    """Reuse a ``prepared`` reservation (nothing was ever invoked for it) only for exactly the request and
    evidence it was built from; a new user turn, a changed todo list or route restages and supersedes it.
    The superseded handoff file stays where it is."""
    claim = source.claim
    # A record a legacy rotation copied from the parent is never reused: the child restages and supersedes it.
    if (claim is not None and claim["source_session_id"] == source.session_id
            and claim["generation"] == source.full_compaction_count
            and claim["profile"] == arm.profile
            and (claim["run_budget_seconds"], claim["run_budget_started_at"]) == _source_run_budget(source_agent)):
        stored = _load_verified(home, claim)
        current = _build(source_agent, arm, db, home, source, gate_result, rollover_id=claim["rollover_id"],
                         created_at=stored["body"]["created_at"])
        if _same_snapshot(stored, current):
            return claim, stored
    return _stage_reservation(source_agent, arm, db, home, source, gate_result)


def _settle(db: Any, claim: dict[str, Any], from_state: str, to_state: str, reason: str) -> None:
    """Record why the transition stopped. If even that write fails the claim stays where it was, which
    blocks every later attempt: fail closed, never silently reset."""
    try:
        advance(db, claim["source_session_id"], claim["rollover_id"], from_state, to_state,
                updates={"reason": reason, **({"claim_token": None} if to_state == "prepared" else {})})
    except (RolloverConflict, *_STATE_ERRORS):
        logger.warning("context rollover %s: could not record %s (%s)", claim["rollover_id"], to_state, reason,
                       exc_info=True)


def _discard(adapter: RolloverHostAdapter, agent: Any, source_agent: Any) -> None:
    if agent is None or agent is source_agent:
        return
    try:
        adapter.retire(agent)
    except Exception:
        logger.warning("context rollover: discarding an unused fresh agent failed", exc_info=True)


def _same_store(a: Any, b: Any) -> bool:
    try:
        return a is b or Path(a.db_path).resolve() == Path(b.db_path).resolve()
    except (AttributeError, TypeError, OSError):
        return False


def _fresh_agent_problem(agent: Any, source_agent: Any, spec: FreshAgentSpec) -> Optional[str]:
    if (agent is None or agent is source_agent or getattr(agent, "session_id", None) != spec.session_id
            or getattr(agent, "_delegate_depth", 0) or getattr(agent, "_persist_disabled", False)):
        return "fresh_agent_invalid"
    if not _same_store(getattr(agent, "_session_db", None), spec.session_db):
        return "target_db_mismatch"
    return _settings_problem(agent, spec)


def _settings_problem(agent: Any, spec: FreshAgentSpec) -> Optional[str]:
    """The route (cwd and service tier included) and wall-clock cap observed on *agent* against *spec*."""
    try:
        if route_snapshot(agent) != spec.route:
            return "route_mismatch"
    except (HandoffError, AttributeError, TypeError, ValueError):  # a malformed agent is not a matching one
        return "route_mismatch"
    try:
        if _budget_seconds(getattr(agent, "run_budget_seconds", None)) != spec.run_budget_seconds:
            return "run_budget_mismatch"
    except ValueError:
        return "run_budget_mismatch"
    return None


def _build_fresh_agent(adapter, db, claim, spec) -> Any:
    try:
        return adapter.make_agent(spec)
    except Exception as exc:  # health: allow BLE001 -- host factory; its errors may carry credentials, so only the type is recorded
        _settle(db, claim, "claimed", "prepared", f"factory_failed: {type(exc).__name__}")
        raise _Blocked("factory_failed") from None


def _release(adapter, agent, source_agent, db, claim, reason: str) -> NoReturn:
    """Nothing was invoked for the claim: dispose of the unused candidate and hand the reservation back."""
    _discard(adapter, agent, source_agent)
    _settle(db, claim, "claimed", "prepared", reason)
    raise _Blocked(reason)


def _abandon(adapter, agent, source_agent, db, claim, reason: str) -> NoReturn:
    """A failure nobody can explain with the candidate built: dispose of it, but never hand the claim back as
    never-invoked. It ends ``blocked`` (or stays ``claimed`` if even that write fails) for a person."""
    _discard(adapter, agent, source_agent)
    _settle(db, claim, "claimed", "blocked", reason)
    raise _Blocked(reason)


def _take_source_input(source_agent: Any, outcome: RolloverOutcome) -> Optional[str]:
    """A stop or new input on the source, consumed exactly once (``take_agent_input``) with its text handed
    back as ``outcome.pending_input``; else a running child (reported, never consumed). Call it before any
    candidate is disposed, so a stop arriving during that cleanup stays on the source for its next reader."""
    reason, text = take_agent_input(source_agent)
    if reason:
        outcome.pending_input = text
        return reason
    return agent_input_block(source_agent)


def _bind_candidate(source_agent, adapter, db, source, claim, agent, spec, mark, outcome) -> None:
    """Verify the built candidate, bind its fresh root target, recheck input and commit ``installed``."""
    if problem := _fresh_agent_problem(agent, source_agent, spec):
        _release(adapter, agent, source_agent, db, claim, problem)
    row_config = agent._session_row_model_config()
    if not isinstance(row_config, dict):
        _abandon(adapter, agent, source_agent, db, claim, "target_invalid")
    try:
        # The target row records the cwd both agents were verified to work in, not the source row's.
        bind_target(db, claim, source=source.source, model=spec.route.get("model"), cwd=spec.route.get("cwd"),
                    model_config=row_config)
    except (RolloverConflict, RolloverRecordError):
        _abandon(adapter, agent, source_agent, db, claim, "target_invalid")
    except STORAGE_ERRORS:  # rolled back; an untouched target row a retry finds again is accepted by bind_target
        _release(adapter, agent, source_agent, db, claim, "state_unavailable")
    if reason := _take_source_input(source_agent, outcome) or host_state_block(adapter.host_state):
        _release(adapter, agent, source_agent, db, claim, reason)  # just before the swap
    # Durable first, host pointer second: the guarded commit re-verifies the live source (profile, generation,
    # no compression lease, no transcript row after *mark*) and ends it, or nothing is installed at all.
    try:
        advance(db, source.session_id, claim["rollover_id"], "claimed", "installed", end_source=True,
                messages_mark=mark)
    except RolloverConflict:
        _release(adapter, agent, source_agent, db, claim, "source_changed")
    except _STATE_ERRORS:
        # Rolled back. If the commit landed after all, releasing fails its compare-and-set and the claim stays
        # ``installed``, which blocks the source and the target alike.
        _release(adapter, agent, source_agent, db, claim, "state_unavailable")


def _transition(source_agent, adapter, db, source, claim, handoff, mark, outcome) -> RolloverOutcome:
    rollover_id = claim["rollover_id"]
    try:
        claim = advance(db, source.session_id, rollover_id, "prepared", "claimed",
                        updates={"claim_token": secrets.token_hex(8), "reason": None})
    except RolloverConflict as exc:
        stored = exc.record
        if stored is not None and stored["rollover_id"] == rollover_id and stored["state"] == "completed":
            outcome.status = "already_completed"
            return outcome
        raise _Blocked("claimed_elsewhere") from None
    except _STATE_ERRORS:
        raise _Blocked("state_unavailable") from None
    spec = FreshAgentSpec(session_id=claim["target_session_id"], session_db=db, route=handoff["body"]["route"],
                          rollover_id=rollover_id, source_session_id=source.session_id, profile=claim["profile"],
                          run_budget_seconds=claim["run_budget_seconds"])
    agent = _build_fresh_agent(adapter, db, claim, spec)
    outcome.fresh_spec = spec
    try:
        _bind_candidate(source_agent, adapter, db, source, claim, agent, spec, mark, outcome)
    except _Blocked:
        raise
    except Exception as exc:  # health: allow BLE001 -- sanitized boundary: candidate errors may carry secrets
        logger.warning("context rollover %s: preparing the fresh agent failed (%s)", rollover_id,
                       type(exc).__name__)
        _abandon(adapter, agent, source_agent, db, claim, "fresh_agent_failed")
    outcome.agent = agent
    try:
        adapter.install(agent, outcome)
    except Exception:  # health: allow BLE001 -- host callback; what it adopted is fenced by the blocked claim
        # Durable first. The candidate is retired either way: unadopted it is unused, and an adopted one the
        # host still points at can no longer run a turn (the target's gate reads the blocked claim).
        _settle(db, claim, "installed", "blocked", "install_failed")
        _discard(adapter, agent, source_agent)
        outcome.agent = None
        raise _Blocked("install_failed") from None
    try:
        adapter.retire(source_agent)
    except Exception:  # health: allow BLE001 -- host callback; any failure blocks the claim durably, its text never reaches the outcome
        _settle(db, claim, "installed", "blocked", "retire_failed")
        raise _Blocked("retire_failed") from None
    return _resume(outcome, adapter, source_agent=source_agent)


def _start_block(source_agent, arm, adapter, db, source, request, outcome) -> Optional[str]:
    if reason := claim_block(source, db):
        return reason
    if source.profile_name != arm.profile:
        return "profile_mismatch"
    if source.ended:
        return "source_ended"
    if (source.full_compaction_count != request.get("full_compaction_count")
            or not rollover_due(source.full_compaction_count, arm.threshold)):
        return "stale_request"
    # A compression lease taken after the gate: its compaction would move the evidence under the handoff.
    return (compression_lock_block(db, source.session_id) or _take_source_input(source_agent, outcome)
            or host_state_block(adapter.host_state))


def perform_rollover(source_agent: Any, gate_result: Any, adapter: RolloverHostAdapter) -> RolloverOutcome:
    """Carry a gated rollover request from *source_agent* through to a resumed fresh agent (see module
    docstring). Never raises for an ordinary failure: the outcome says ``blocked`` and why."""
    outcome = RolloverOutcome(status="blocked", source_session_id=getattr(source_agent, "session_id", None),
                              source_api_calls=int(getattr(source_agent, "session_api_calls", 0) or 0))
    try:
        arm, db, source = _verify_request(source_agent, gate_result)
        if source.claim is not None and source.claim["state"] == "completed" \
                and source.claim["source_session_id"] == source.session_id:
            outcome.status = "already_completed"
            outcome.rollover_id, outcome.target_session_id = (source.claim["rollover_id"],
                                                              source.claim["target_session_id"])
            return outcome
        if reason := _start_block(source_agent, arm, adapter, db, source, gate_result["context_rollover"],
                                  outcome):
            raise _Blocked(reason)
        home = Path(db.db_path).parent  # the store's own profile home, never a caller-supplied path
        try:
            mark = messages_mark(db, source.session_id)  # before the evidence is read: see _transition
        except RolloverRecordError:
            raise _Blocked("state_unavailable") from None
        claim, handoff = _reserve_or_recover(source_agent, arm, db, home, source, gate_result)
        try:
            outcome.resume_prompt = render_resume_prompt(handoff)
        except HandoffError:
            raise _Blocked("handoff_invalid") from None
        outcome.rollover_id, outcome.target_session_id = claim["rollover_id"], claim["target_session_id"]
        outcome.generation, outcome.budget_remaining = claim["generation"], claim["budget_remaining"]
        outcome.handoff_path = handoff_path(home, claim["rollover_id"])
        return _transition(source_agent, adapter, db, source, claim, handoff, mark, outcome)
    except _Blocked as blocked:
        outcome.status, outcome.reason = "blocked", blocked.reason
        return outcome
    except Exception as exc:  # health: allow BLE001 -- sanitized boundary: an outcome never carries raw error text
        # The durable claim stays wherever this left it; every state past ``prepared`` already blocks.
        logger.warning("context rollover: unexpected failure (%s)", type(exc).__name__)
        outcome.status, outcome.reason = "blocked", "internal_error"
        return outcome


def _resume(outcome: RolloverOutcome, adapter: RolloverHostAdapter, user_text: Optional[str] = None, *,
            source_agent: Any = None) -> RolloverOutcome:
    agent = outcome.agent
    if source_agent is not None:
        # The first resume: a stop or message the user sent the source while the host was installing.
        reason, text = take_agent_input(source_agent)
        if reason:
            outcome.status, outcome.reason, outcome.pending_input = "deferred", reason, text
            return outcome
    if reason := agent_input_block(agent) or host_state_block(adapter.host_state):  # just before the resume
        outcome.status, outcome.reason = "deferred", reason
        return outcome
    prompt = outcome.resume_prompt
    if user_text:
        from agent.context_compressor import is_compaction_summary_message

        prompt = f"{prompt}\n\n{_LATEST_INPUT_HEADER}\n{user_text}"
        if is_compaction_summary_message({"role": "user", "content": prompt}):
            raise _Blocked("resume_prompt_ambiguous")
    db = agent._session_db
    if outcome.fresh_spec is None:
        raise _Blocked("fresh_agent_invalid")
    if problem := _settings_problem(agent, outcome.fresh_spec):
        # Installed, but no longer the agent the claim was verified for: it never runs the resume.
        _settle(db, {"source_session_id": outcome.source_session_id, "rollover_id": outcome.rollover_id},
                "installed", "blocked", problem)
        raise _Blocked(problem)
    try:
        started = advance(db, outcome.source_session_id, outcome.rollover_id, "installed", "started")
    except RolloverConflict:
        raise _Blocked("claimed_elsewhere") from None
    except RolloverRecordError:
        raise _Blocked("state_unavailable") from None
    except STORAGE_ERRORS:  # rolled back: still ``installed``, so the held-back resume can be retried
        outcome.status, outcome.reason = "deferred", "state_unavailable"
        return outcome
    # The one-shot carry is armed only here, from the durable ``started`` claim, for the authorized resume
    # itself: a turn refused on the fenced target before this point cannot consume it. A real user turn
    # (*user_text*) runs under its own normal budget instead.
    setattr(agent, BUDGET_CARRY_ATTR, None if user_text else {
        "remaining": started["budget_remaining"], "run_budget_started_at": started["run_budget_started_at"]})
    # The target's gate lets exactly this claim's resume turn through (``origin_block``); nothing else.
    setattr(agent, RESUME_ATTR, (started["rollover_id"], started["claim_token"]))
    try:
        result = adapter.run_turn(agent, prompt)
    except Exception:
        # The resume may have run tools before failing: the claim stays ``started`` for a person.
        logger.warning("context rollover %s: resume turn failed", outcome.rollover_id, exc_info=True)
        raise _Blocked("resume_failed") from None
    finally:
        setattr(agent, RESUME_ATTR, None)
        setattr(agent, BUDGET_CARRY_ATTR, None)  # never left armed for a later ordinary turn
    outcome.resume_result = result
    outcome.resume_api_calls = int(result.get("api_calls") or 0) if isinstance(result, dict) else None
    verdict = result.get("context_rollover") if isinstance(result, dict) else None
    if not outcome.resume_api_calls and not (isinstance(verdict, dict) and verdict.get("status") == "due"):
        # The turn ended before any model call (a stop it honoured, an error, its own gate refusing): nothing
        # was resumed, so the claim is not completed. It stays ``started`` (the turn did begin; it is never
        # replayed) and the target stays fenced for a person. A target already due again at its first boundary
        # is the zero-progress chain instead (``_MAX_ZERO_PROGRESS_ROLLOVERS``), recorded as resumed.
        logger.warning("context rollover %s: resume turn ended before any model call", outcome.rollover_id)
        raise _Blocked("resume_not_run")
    outcome.status, outcome.reason = "resumed", None
    try:
        advance(db, outcome.source_session_id, outcome.rollover_id, "started", "completed",
                updates={"resume_api_calls": outcome.resume_api_calls})
    except (RolloverConflict, *_STATE_ERRORS):
        outcome.reason = "completion_not_recorded"
    return outcome


def resume_deferred(outcome: RolloverOutcome, adapter: RolloverHostAdapter, *,
                    user_text: Optional[str] = None) -> RolloverOutcome:
    """Run the held-back resume of a ``deferred`` outcome once the host is clear, with the user's new
    message (if any) appended after the handoff as the instruction that takes precedence."""
    if outcome.status != "deferred" or outcome.agent is None:
        raise ValueError("only a deferred rollover outcome can be resumed")
    try:
        return _resume(outcome, adapter, user_text)
    except _Blocked as blocked:
        outcome.status, outcome.reason = "blocked", blocked.reason
        return outcome
    except Exception as exc:  # health: allow BLE001 -- sanitized boundary, as in perform_rollover
        logger.warning("context rollover %s: unexpected failure (%s)", outcome.rollover_id, type(exc).__name__)
        outcome.status, outcome.reason = "blocked", "internal_error"
        return outcome
