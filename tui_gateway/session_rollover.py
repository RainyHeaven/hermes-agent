"""Context rollover host for the TUI gateway (``hermes --tui``).

Wiring (``prompt_turn._invoke_agent``): :func:`run_turn` replaces the bare ``run_conversation`` call. It arms
the session agent from the live profile config, runs the turn (or, for the user's own message while a resume
is held back, that resume), then drives a due rollover to completion on the same turn thread, so every
adapter callback is synchronous. The rest of the turn (history commit, ``message.complete``, receipts)
then sees the fresh agent and its session exactly as it sees a compression rotation.

Only the user's own prompts (``prompt.submit`` and its queue) may resume a held rollover; synthesized turns
(goal/loop continuations, heartbeats, wake-ups, auto-continue) never run while one is held. Desktop chat,
Bot Mode and compute-host sessions are never armed. Every server helper is read from ``tui_gateway.server``
at call time (the seam tests patch).
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Optional

from agent.context_rollover_gate import HostState
from agent.context_rollover_host import HostHooks, arm_for_turn, drive, held_for, resume_held
from agent.context_rollover_runtime import FreshAgentSpec, RolloverHostAdapter

logger = logging.getLogger(__name__)

HELD_KEY = "_context_rollover_held"
STOPPED_KEY = "_context_rollover_stopped"
_STATUS_KIND = "context_rollover"


def _server() -> Any:
    from tui_gateway import server

    return server


def _eligible(session: dict, agent: Any) -> bool:
    srv = _server()
    if getattr(agent, "platform", None) != "tui" or srv._is_bot_mode_session(session):
        return False
    try:
        return not srv._session_uses_compute_host(session)
    except Exception:  # health: allow BLE001 -- an unknown isolation mode is not an eligible host
        logger.debug("compute-host probe failed; rollover stays off", exc_info=True)
        return False


def host_state(sid: str, session: dict) -> HostState:
    """What the TUI session is waiting on: open server→client requests, its queue, a Stop, its processes."""
    from tools.approval import get_pending_gateway_approval
    from tools.process_registry import process_registry
    from tui_gateway import server_requests

    key = str(session.get("session_key") or "")
    pending_kind = server_requests.pending_kind(sid)
    return HostState(
        children_active=False,  # the controller reads the agent's own children under its lock
        processes_active=bool(process_registry.has_active_processes(key)
                              or process_registry.has_active_for_session(key)),
        approval_pending=bool(get_pending_gateway_approval(key)) or pending_kind not in ("", "clarify"),
        clarify_pending=pending_kind == "clarify",
        queued_input=bool(session.get("queued_prompt") or session.get("queued_prompts")),
        stop_requested=bool(session.get("_turn_cancel_requested") or session.get("_closing")),
    )


def arm(sid: str, session: dict, agent: Any) -> None:
    srv = _server()
    arm_for_turn(agent, "tui", compression_cfg=(srv._load_cfg() or {}).get("compression"),
                 profile=srv._current_profile_name(), host_state=lambda: host_state(sid, session),
                 eligible=_eligible(session, agent))


def _make_agent(sid: str, session: dict, source: Any, spec: FreshAgentSpec) -> Any:
    """A fresh agent through the session's normal build, on the source's exact route (credentials in memory)."""
    route = spec.route
    override = {"model": route["model"], "provider": source.provider, "base_url": route.get("base_url"),
                "api_key": getattr(source, "api_key", None), "api_mode": route.get("api_mode")}
    agent = _server()._make_agent(
        sid, spec.session_id, session_id=spec.session_id, session_db=spec.session_db, model_override=override,
        exact_route=route, platform_override=getattr(source, "platform", None), cwd_override=route.get("cwd"),
        context_cwd_is_launch_artifact=bool(getattr(source, "_context_cwd_is_launch_artifact", False)))
    agent.run_budget_seconds = spec.run_budget_seconds  # the controller verifies both before install
    agent.max_iterations = source.max_iterations
    return agent


def _install(sid: str, session: dict, st: Any, agent: Any) -> None:
    """Swap the session onto the fresh agent: empty history, re-anchored key/lease/approvals/yolo, DB owner."""
    srv = _server()
    old = session.get("agent")
    with session["history_lock"]:
        session["agent"] = agent
        session["history"] = []
        session["history_version"] = int(session.get("history_version", 0)) + 1
    try:
        srv._sync_session_key_after_compress(sid, session, clear_pending_title=True, restart_slash_worker=True)
        srv._wire_session_agent(sid, agent.session_id, agent)
        if old is not None and getattr(old, "_owns_session_db", False) \
                and srv._transfer_db_to_agent(agent, getattr(old, "_session_db", None)):
            old._owns_session_db = False
        agent.interim_assistant_callback = getattr(old, "interim_assistant_callback", None)
    except BaseException:
        with session["history_lock"]:
            session["agent"] = old  # the target stays fenced by its durable claim; the host keeps the source
        raise
    st.agent, st.history, st.history_version = agent, [], int(session.get("history_version", 0))
    arm(sid, session, agent)
    with contextlib.suppress(Exception):
        srv._emit("session.info", sid, srv._session_info(agent, session))


def _retire(session: dict, agent: Any) -> None:
    """Release *agent*'s clients (never its SessionDB, never ``close()``: that would end the session row)."""
    if agent is None:
        return
    if session.get("agent") is agent:  # defensive: _install puts the source back before a candidate retires
        with session["history_lock"]:
            session["agent"] = None
    if hasattr(agent, "release_clients"):
        agent.release_clients()


def _run_resume(sid: str, session: dict, st: Any, agent: Any, prompt: str) -> dict:
    """The resume turn on this thread, streaming through the turn's own callback, under the new session's
    approval key and session context."""
    srv = _server()
    from tools.approval_context import reset_current_session_key, set_current_session_key

    key = agent.session_id
    approval = set_current_session_key(key)
    tokens = srv._set_session_context(key, ui_session_id=sid)
    try:
        stream = (st.run_kwargs or {}).get("stream_callback")
        # The handoff is model input, not a user bubble: the user's own message (if any) has its own row.
        return agent.run_conversation(prompt, conversation_history=[], stream_callback=stream, task_id=key,
                                      persist_user_display_kind="hidden")
    finally:
        srv._clear_session_context(tokens)
        reset_current_session_key(approval)


def _hooks(sid: str, session: dict, st: Any) -> HostHooks:
    srv = _server()

    def notify(text: str) -> None:
        with contextlib.suppress(Exception):
            srv._emit("status.update", sid, {"kind": _STATUS_KIND, "text": text})

    def requeue(text: str) -> None:
        with session["history_lock"]:
            srv._enqueue_prompt(session, text, session.get("transport"))

    def hold(outcome: Optional[Any]) -> None:
        session[HELD_KEY] = outcome

    adapter = RolloverHostAdapter(
        host_state=lambda: host_state(sid, session),
        make_agent=lambda spec: _make_agent(sid, session, st.agent, spec),
        install=lambda agent, _outcome: _install(sid, session, st, agent),
        retire=lambda agent: _retire(session, agent),
        run_turn=lambda agent, prompt: _run_resume(sid, session, st, agent, prompt))
    return HostHooks(adapter=adapter, notify=notify, requeue=requeue, hold=hold)


def held(session: dict) -> Optional[Any]:
    outcome = held_for(session.get(HELD_KEY), session.get("agent"), session.get("session_key"))
    if outcome is None:
        session.pop(HELD_KEY, None)
    return outcome


def suppresses_followups(session: dict, result: Any) -> bool:
    """No automatic follow-up after a rollover stop or while a resume is held back."""
    return session.get(HELD_KEY) is not None or (
        isinstance(result, dict) and isinstance(result.get("context_rollover"), dict))


def run_turn(sid: str, session: dict, st: Any, run_message: Any, *, user_turn: bool) -> Any:
    """``st.agent``'s turn (the held resume instead, for the user's own message), with a due rollover driven."""
    agent = st.agent
    outcome = held(session)
    if outcome is not None:
        if not user_turn:
            _hooks(sid, session, st).notify("Context rollover resume is waiting for your next message.")
            return {"final_response": "", "messages": list(st.history), "api_calls": 0, "completed": False,
                    "failed": False, "partial": True}
        from agent.context_rollover import _content_text

        text = run_message if isinstance(run_message, str) else _content_text(run_message)
        try:
            return resume_held(outcome, text, list(st.history), _hooks(sid, session, st))
        except Exception:  # health: allow BLE001 -- the controller fails closed; the message is handed back
            logger.warning("context rollover resume failed", exc_info=True)
            session.pop(HELD_KEY, None)
            return {"final_response": "", "messages": list(st.history), "api_calls": 0, "completed": False,
                    "failed": True, "error": "Context rollover resume failed."}
    arm(sid, session, agent)
    result = agent.run_conversation(run_message, **st.run_kwargs)
    try:
        return drive(agent, result, _hooks(sid, session, st))
    except Exception:  # health: allow BLE001 -- never lose the user's turn over the host driver
        logger.warning("context rollover host failed", exc_info=True)
        return result
