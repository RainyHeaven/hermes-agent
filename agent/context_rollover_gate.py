"""Safe-boundary gate for context rollover.

A CLI/TUI host opts an agent in with :func:`arm_context_rollover`; config alone never arms anything. The turn
loop calls :func:`rollover_yield` at turn entry (the context build may have compacted) and before every API
attempt (a preflight or reactive overflow compaction may have). Once the durable full-compaction count reaches
the threshold, the turn ends there with a structured ``context_rollover`` request, or a ``blocked`` verdict when
anything is unresolved or unknown, never with another model call. The tool batch that just ran has already
finished; nothing is killed. ``agent/context_rollover_runtime.py`` consumes the request.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from typing import Any, Callable, Optional

from agent.context_rollover import (
    ROLLOVER_SURFACES, RolloverHost, rollover_after_setting, rollover_due, rollover_eligible,
)
from agent.interrupt_control import _ic_signal_tool_workers
from agent.message_metadata import append_message
from hermes_state_compression import ROLLOVER_CLAIM_KEY, ROLLOVER_ORIGIN_KEY
from hermes_state_rollover import (
    STORAGE_ERRORS, RolloverRecordError, compression_descends_from, origin_for, read_source,
)
from tools.interrupt import set_interrupt

logger = logging.getLogger(__name__)

ARM_ATTR = "_context_rollover_arm"
# ``(rollover_id, claim_token)``, set by the controller on the fresh agent only for its one resume turn.
RESUME_ATTR = "_context_rollover_resume"
ROLLOVER_DISPLAY_KIND = "context_rollover"


@dataclasses.dataclass(frozen=True)
class HostState:
    """What only the host knows at a boundary. Every field must be a real bool: anything else is unknown."""

    children_active: bool
    processes_active: bool
    approval_pending: bool
    clarify_pending: bool
    queued_input: bool
    stop_requested: bool

    def blocking_reason(self) -> Optional[str]:
        if any(type(getattr(self, f.name)) is not bool for f in dataclasses.fields(self)):
            return "host_state_unknown"
        # Stop and new input first: they outrank any automatic resume.
        for name in ("stop_requested", "queued_input", "approval_pending", "clarify_pending",
                     "children_active", "processes_active"):
            if getattr(self, name):
                return name
        return None


@dataclasses.dataclass(frozen=True)
class RolloverArm:
    host: RolloverHost
    threshold: int
    profile: str
    host_state: Callable[[], HostState]


@dataclasses.dataclass(frozen=True)
class RolloverVerdict:
    status: str  # "due" | "blocked"
    reason: Optional[str]
    source_session_id: str
    profile: str
    threshold: int
    full_compaction_count: Optional[int] = None
    claim_state: Optional[str] = None
    target_session_id: Optional[str] = None
    # Set when this session is a rollover TARGET whose transition did not complete: the ids a person needs.
    rollover_id: Optional[str] = None
    origin_session_id: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _delegated(agent: Any) -> bool:
    from agent.delegation_context import is_delegated_child_context

    return (getattr(agent, "_delegate_depth", 0) > 0 or bool(getattr(agent, "_persist_disabled", False))
            or getattr(agent, "platform", None) == "cron" or is_delegated_child_context())


def arm_context_rollover(
    agent: Any, host: Optional[RolloverHost], *, compression_cfg: Any, profile: str,
    host_state: Callable[[], HostState],
) -> Optional[str]:
    """The host's explicit opt-in for *agent*. Returns why rollover stays off (None when armed or set to 0);
    always replaces an earlier arm, so a host re-arms after every agent swap or config reload."""
    setattr(agent, ARM_ATTR, None)
    threshold, error = rollover_after_setting(compression_cfg)
    if error or threshold == 0:
        return error
    if not rollover_eligible(host, delegated=_delegated(agent)):
        return "context rollover needs a CLI/TUI host that resumes in a fresh agent"
    if not isinstance(profile, str) or not profile:
        return "context rollover needs the owning profile name"
    setattr(agent, ARM_ATTR, RolloverArm(host=host, threshold=threshold, profile=profile, host_state=host_state))
    return None


def claim_block(source: Any, db: Any = None) -> Optional[str]:
    """Why an existing transition record forbids starting (or re-starting) a rollover from *source*.

    A legacy rotation copies its parent's record onto the child. A copied ``prepared`` record (the parent's
    reservation was handed back: nothing was ever invoked for it) does not wedge the child when *db* proves
    the child continues that parent through compaction only and the parent still holds exactly that record;
    the child's own reservation supersedes the copy. Any other foreign record fails closed."""
    claim = source.claim
    if claim is None:
        return None
    if claim["source_session_id"] != source.session_id:
        if claim["state"] != "prepared" or db is None:
            return "rollover_foreign_claim"
        try:
            released = compression_descends_from(db, source.session_id, claim["source_session_id"],
                                                 profile=claim["profile"], key=ROLLOVER_CLAIM_KEY, record=claim)
        except RolloverRecordError:
            return "state_unavailable"
        return None if released else "rollover_foreign_claim"
    if claim["state"] == "completed":
        return "already_rolled_over"
    if claim["state"] != "prepared":  # only a never-invoked reservation may be picked up again
        return f"rollover_{claim['state']}"
    return None


def _durable_block(agent: Any, source: Any, arm: RolloverArm) -> Optional[str]:
    if reason := claim_block(source, agent._session_db):
        return reason
    if source.profile_name != arm.profile:
        return "profile_mismatch"
    if source.ended:
        return "source_ended"
    return compression_lock_block(agent._session_db, source.session_id)


def compression_lock_block(db: Any, session_id: str) -> Optional[str]:
    try:
        holder = db.get_compression_lock_holder(session_id)
    except STORAGE_ERRORS:
        return "state_unavailable"
    return "compression_in_progress" if holder is not None else None


def origin_block(agent: Any, target: Any) -> Optional[str]:
    """Why *agent* may not run a turn on *target*, a session a rollover created. Until the source's durable
    claim says ``completed`` the only turn allowed is the controller's own resume (``started``, carrying
    :data:`RESUME_ATTR` for exactly this claim); a deferred, failed, crashed or foreign transition needs the
    controller or a person, never an ordinary turn on an empty history. Read from the store every time, so a
    reloaded target fails closed exactly like the agent that was installed. A legacy rotation of the target
    copies the origin onto its child: the child is judged as the target only when it provably continues the
    claim's own target through compaction (``compression_descends_from``)."""
    origin = target.origin
    if origin is None:
        return None
    db = agent._session_db
    try:
        claim = read_source(db, origin["source_session_id"]).claim
        if claim is None or origin_for(claim) != origin:
            return "rollover_origin_mismatch"
        if claim["target_session_id"] != target.session_id and not compression_descends_from(
                db, target.session_id, claim["target_session_id"], profile=claim["profile"],
                key=ROLLOVER_ORIGIN_KEY, record=origin):
            return "rollover_origin_mismatch"
    except RolloverRecordError:
        return "state_unavailable"
    if claim["state"] == "completed":
        return None
    if claim["state"] == "started" and getattr(agent, RESUME_ATTR, None) == (claim["rollover_id"],
                                                                            claim["claim_token"]):
        return None
    return f"rollover_origin_{claim['state']}"


def agent_input_block(agent: Any) -> Optional[str]:
    """Stop, a queued steer/redirect, or a running child on *agent* itself."""
    if getattr(agent, "_interrupt_requested", False):
        return "queued_input" if agent._has_pending_redirect() else "stop_requested"
    if getattr(agent, "_pending_steer", None):
        return "queued_input"
    with agent._active_children_lock:
        return "children_active" if agent._active_children else None


def take_agent_input(agent: Any) -> tuple[Optional[str], Optional[str]]:
    """``(reason, text)`` for the stop or queued input on *agent* NOW, consuming exactly what it reports.

    Read under the lock ``interrupt()``/``redirect()`` publish under, so input that arrived after an earlier
    check is reported instead of erased, and a stop published after this returns stays set for the next
    reader. A stop wins (``interrupt()`` already dropped any redirect) and leaves a buffered steer where it
    is; otherwise the redirect/steer the user already sent comes back as *text* instead of being dropped."""
    with agent._pending_redirect_lock:
        interrupted, redirect = agent._interrupt_requested, agent._pending_redirect
        if interrupted:
            agent._interrupt_requested, agent._pending_redirect = False, None
            agent._interrupt_message = agent._tool_interrupt_reason = None
            getattr(agent, "_hard_interrupt_requested", threading.Event()).clear()
    if interrupted:  # the per-thread tool signals, as clear_interrupt() does after its locked section
        agent._interrupt_thread_signal_pending = False
        if agent._execution_thread_id is not None:
            set_interrupt(False, agent._execution_thread_id)
        _ic_signal_tool_workers(agent, False)
        if not redirect:
            return "stop_requested", None
    text = "\n".join(t for t in (redirect, agent._drain_pending_steer()) if t)
    return ("queued_input", text) if text else (None, None)


def host_state_block(host_state: Callable[[], HostState]) -> Optional[str]:
    try:
        state = host_state()
    except Exception:  # health: allow BLE001 -- host callback: a guard that cannot answer is not a clear host
        return "host_state_unavailable"
    return state.blocking_reason() if isinstance(state, HostState) else "host_state_unknown"


def _origin_verdict(agent: Any, source: Any, verdict: dict[str, Any]) -> Optional[RolloverVerdict]:
    if reason := origin_block(agent, source):  # before the count: a fresh target's count is 0
        return RolloverVerdict(status="blocked", reason=reason, full_compaction_count=source.full_compaction_count,
                               rollover_id=source.origin["rollover_id"],
                               origin_session_id=source.origin["source_session_id"], **verdict)
    return None


def _unarmed_origin_check(agent: Any) -> Optional[RolloverVerdict]:
    """Rollover off (any host, any config) still never runs an ordinary turn on a rollover target whose
    transition did not complete. This arms nothing: a session whose stored config does not even mention an
    origin (every legacy, gateway, cron or delegated session) or that has no row yet carries on unchanged, and
    the full strict read happens only for a row that does mention one. A CLI/TUI session whose row cannot
    be read at all fails closed: it may be an unfinished target; other surfaces keep their old behaviour."""
    db, sid = getattr(agent, "_session_db", None), getattr(agent, "session_id", None)
    if db is None or not isinstance(sid, str) or not sid:
        return None
    try:
        row = db.get_session(sid)
        raw = row["model_config"] if row is not None else None
    except Exception:  # health: allow BLE001 -- the store's own error text never reaches the verdict
        if getattr(agent, "platform", None) not in ROLLOVER_SURFACES or _delegated(agent):
            return None
        return RolloverVerdict(status="blocked", reason="state_unavailable", source_session_id=sid,
                               profile="", threshold=0)
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    if not (isinstance(raw, (str, dict)) and ROLLOVER_ORIGIN_KEY in raw):
        return None
    verdict = dict(source_session_id=sid, profile=row["profile_name"] or "", threshold=0)
    try:
        source = read_source(db, sid)
    except RolloverRecordError:  # it names an origin we cannot verify: fail closed
        return RolloverVerdict(status="blocked", reason="state_unavailable", **verdict)
    return None if source.origin is None else _origin_verdict(agent, source, verdict)


def check_rollover(agent: Any) -> Optional[RolloverVerdict]:
    """None = carry on exactly as before (unarmed and not an unfinished rollover target, or not due).
    Otherwise the verdict for this boundary."""
    arm = getattr(agent, ARM_ATTR, None)
    if arm is None:
        return _unarmed_origin_check(agent)
    verdict = dict(source_session_id=agent.session_id, profile=arm.profile, threshold=arm.threshold)
    try:
        source = read_source(agent._session_db, agent.session_id)
    except RolloverRecordError:
        return RolloverVerdict(status="blocked", reason="state_unavailable", **verdict)
    if blocked := _origin_verdict(agent, source, verdict):
        return blocked
    if not rollover_due(source.full_compaction_count, arm.threshold):
        return None
    claim = source.claim or {}
    verdict.update(full_compaction_count=source.full_compaction_count, claim_state=claim.get("state"),
                   target_session_id=claim.get("target_session_id"))
    reason = _durable_block(agent, source, arm) or agent_input_block(agent) or host_state_block(arm.host_state)
    return RolloverVerdict(status="blocked" if reason else "due", reason=reason, **verdict)


def rollover_notice(verdict: dict[str, Any]) -> str:
    count = verdict.get("full_compaction_count")
    if verdict["status"] == "due":
        return (f"Context rollover: this conversation reached {count} context compactions, so the task pauses "
                "here to continue in a fresh session from a verified handoff. If the handoff cannot start, "
                "Hermes says why. Nothing further runs in this session.")
    if verdict.get("rollover_id"):
        return (f"This session was opened by context rollover {verdict['rollover_id']} from session "
                f"{verdict['origin_session_id']}, which did not complete ({verdict['reason']}). This turn stopped "
                "before any model call: the task has not been resumed here. Recover that rollover, or continue "
                "from the original session's transcript in a new session with /new.")
    lead = (f"Context rollover is due after {count} context compactions but could not start safely"
            if count is not None else "Context rollover state could not be verified")
    return (f"{lead} ({verdict['reason']}). This turn stopped before any further model call. Resolve that "
            "and send a message to continue, or start a new session with /new.")


def rollover_yield(agent: Any, s: Any) -> Optional[dict[str, Any]]:
    """The turn result for a due/blocked rollover at this boundary, or None to carry on."""
    verdict = check_rollover(agent)
    if verdict is None:
        return None
    if s.api_kwargs is None and s.api_call_count > 0:
        # Inside an iteration whose request was never built: give back the call it reserved.
        s.api_call_count -= 1
        agent._api_call_count = s.api_call_count
        agent.iteration_budget.refund()
    # Input that arrived after the check outranks a due rollover; never clear a stop this turn did not see.
    input_reason, pending = take_agent_input(agent)
    if input_reason and (verdict.status == "due" or verdict.reason in ("stop_requested", "queued_input")):
        verdict = dataclasses.replace(verdict, status="blocked", reason=input_reason)
    agent._persist_session(s.messages, s.conversation_history)
    data = verdict.as_dict()
    notice = rollover_notice(data)
    result = {
        "final_response": notice, "messages": s.messages, "api_calls": s.api_call_count, "completed": False,
        "failed": False, "partial": True,
        "interrupted": "stop_requested" in (input_reason, verdict.reason),
        "turn_exit_reason": f"context_rollover_{verdict.status}", "context_rollover": data,
        "session_id": agent.session_id, "model": agent.model, "provider": agent.provider,
    }
    if verdict.status == "blocked":
        result["error"] = notice
    if pending:
        result["pending_steer"] = pending  # the finalizer's leftover-steer contract
    return result


def rollover_yield_before(
    agent: Any, s: Any, entry: Callable[[], Optional[dict[str, Any]]],
) -> Optional[dict[str, Any]]:
    """The turn-entry rollover result, else *entry()*'s own early result (None to carry on). The rollover
    verdict comes first, so nothing *entry* runs (another runtime's whole turn) precedes a due rollover."""
    result = rollover_yield(agent, s)
    return result if result is not None else entry()


def close_rollover_turn(agent: Any, result: Any) -> bool:
    """Close a rollover turn's durable tail with a Hermes-authored rollover row (not the generic failed-turn
    boundary). False when *result* is not a rollover turn. A boundary that cannot be written turns a due
    request into a blocked one: the controller must not build a handoff from a transcript it cannot trust."""
    rollover = result.get("context_rollover") if isinstance(result, dict) else None
    if not isinstance(rollover, dict):
        return False
    db, session_id, messages = agent._session_db, agent.session_id, result.get("messages")
    if db is None or not session_id or not isinstance(messages, list) or getattr(agent, "_persist_disabled", False):
        return True
    try:
        if db.latest_conversation_role(session_id) in ("user", "tool"):
            append_message(messages, {"role": "assistant", "content": result["final_response"],
                                      "display_kind": ROLLOVER_DISPLAY_KIND})
            agent._flush_messages_to_session_db(messages)
    except Exception:
        logger.debug("context rollover boundary not written", exc_info=True)
        if rollover["status"] == "due":
            rollover.update(status="blocked", reason="boundary_not_persisted")
            result["final_response"] = result["error"] = rollover_notice(rollover)
            result["turn_exit_reason"] = "context_rollover_blocked"
    return True
