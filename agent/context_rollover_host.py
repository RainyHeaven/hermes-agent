"""Host-side driver for context rollover, shared by the CLI and TUI adapters.

The controller (``agent/context_rollover_runtime.py``) does the transition; this module decides what the host
does with each outcome. A host calls :func:`arm_for_turn` before every ordinary turn, :func:`drive` with the
turn's result, and :func:`resume_held` instead of an ordinary turn while an installed target's resume is held
back. Every adapter callback must be synchronous: ``run_turn`` returns only once the resume turn has finished.

Outcome policy: ``resumed`` returns the resume turn's result (a target that is due again at once continues the
zero-progress chain, bounded by the controller); ``deferred`` keeps the outcome for the user's next message,
which then runs as the resume's latest instruction (never automatically); ``blocked`` keeps the result the user
already saw, and input taken from the source during the attempt is handed back as the user's next turn.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Callable, Optional

from agent.context_rollover import RolloverHost
from agent.context_rollover_gate import ARM_ATTR, HostState, arm_context_rollover
from agent.context_rollover_runtime import RolloverHostAdapter, RolloverOutcome, perform_rollover, resume_deferred

logger = logging.getLogger(__name__)

# More than the controller's zero-progress stop (two in a row), so the controller's own refusal ends a chain.
_MAX_CHAIN = 4


@dataclasses.dataclass(frozen=True)
class HostHooks:
    """``notify``: one line for the user; ``requeue``: run text as the user's next ordinary turn; ``hold``: keep
    (or with None, drop) the deferred outcome the next user message resumes."""

    adapter: RolloverHostAdapter
    notify: Callable[[str], None]
    requeue: Callable[[str], None]
    hold: Callable[[Optional[RolloverOutcome]], None]


def arm_for_turn(agent: Any, surface: str, *, compression_cfg: Any, profile: str,
                 host_state: Callable[[], HostState], eligible: bool = True) -> None:
    """(Re-)arm *agent* for this turn from the current config, or disarm it when the host is not eligible."""
    if agent is None:
        return
    if not eligible:
        setattr(agent, ARM_ATTR, None)
        return
    error = arm_context_rollover(agent, RolloverHost(surface=surface, resumes_in_fresh_agent=True),
                                 compression_cfg=compression_cfg, profile=profile, host_state=host_state)
    if error:
        logger.warning("context rollover stays off: %s", error)


def is_due(result: Any) -> bool:
    verdict = result.get("context_rollover") if isinstance(result, dict) else None
    return isinstance(verdict, dict) and verdict.get("status") == "due"


def held_for(outcome: Optional[RolloverOutcome], agent: Any, session_id: Any) -> Optional[RolloverOutcome]:
    """*outcome* while it still belongs to the host's live agent and session, else None (stale)."""
    if (outcome is None or outcome.status != "deferred" or outcome.agent is None or outcome.agent is not agent
            or outcome.target_session_id != session_id):
        return None
    return outcome


def _annotated(result: Any, text: str, *, installed: bool) -> Any:
    """A display copy of *result* with *text* appended to its reply; the transcript is never rewritten. Once the
    fresh agent is installed the host's history is the target's (empty), never the source's messages."""
    if not isinstance(result, dict):
        return result
    final = result.get("final_response") or ""
    copy = {**result, "final_response": f"{final}\n\n{text}" if final else text}
    if installed:
        copy["messages"] = []
    return copy


def _settle(outcome: RolloverOutcome, result: Any, hooks: HostHooks, *, from_controller: bool) -> Any:
    status, reason = outcome.status, outcome.reason
    if status == "resumed":
        note = f"Context rollover: continued in new session {outcome.target_session_id} " \
               f"(from {outcome.source_session_id})."
        if reason:
            note += f" The completion was not recorded ({reason})."
        hooks.notify(note)
        return outcome.resume_result if isinstance(outcome.resume_result, dict) else result
    if status == "deferred":
        hooks.hold(outcome)
        text, outcome.pending_input = outcome.pending_input, None
        if text and from_controller:
            hooks.requeue(text)  # the message the user sent during the swap, as their next turn
        elif text:
            outcome.pending_input = text  # kept for the next attempt; never requeued into a busy loop
        msg = (f"Context rollover: new session {outcome.target_session_id} is ready, but the resume is held "
               f"back ({reason}). Send a message to continue the task there.")
        hooks.notify(msg)
        return _annotated(result, msg, installed=True)
    hooks.hold(None)
    if status == "already_completed":
        return result
    if outcome.pending_input:
        hooks.requeue(outcome.pending_input)
        outcome.pending_input = None
    where = f" New session {outcome.target_session_id} is fenced." if outcome.agent is not None else ""
    msg = f"Context rollover did not start ({reason}).{where}"
    hooks.notify(msg)
    return _annotated(result, msg, installed=outcome.agent is not None)


def _chain(outcome: RolloverOutcome, result: Any, hooks: HostHooks, *, from_controller: bool) -> Any:
    current = _settle(outcome, result, hooks, from_controller=from_controller)
    for _ in range(_MAX_CHAIN):
        if outcome.status != "resumed" or not is_due(current):
            return current
        outcome = perform_rollover(outcome.agent, current, hooks.adapter)
        current = _settle(outcome, current, hooks, from_controller=True)
    return current


def drive(source_agent: Any, result: Any, hooks: HostHooks) -> Any:
    """The result the host should show for *source_agent*'s finished turn (unchanged unless it is due)."""
    if not is_due(result):
        return result
    return _chain(perform_rollover(source_agent, result, hooks.adapter), result, hooks, from_controller=True)


def resume_held(outcome: RolloverOutcome, user_text: Optional[str], history: list, hooks: HostHooks) -> Any:
    """Run the held-back resume with the user's new message (plus any text kept from an earlier attempt)."""
    kept = outcome.pending_input
    outcome.pending_input = None
    text = "\n\n".join(t for t in (kept, user_text) if isinstance(t, str) and t.strip()) or None
    placeholder = {"final_response": "", "messages": list(history), "api_calls": 0, "completed": False,
                   "failed": False, "partial": True}
    resumed = resume_deferred(outcome, hooks.adapter, user_text=text)
    if resumed.status == "deferred" and text:
        resumed.pending_input = text  # still held: the text waits for the next attempt
    result = _chain(resumed, placeholder, hooks, from_controller=False)
    if resumed.status == "blocked" and text and isinstance(result, dict):
        # The fenced target cannot run it: show the text back so the user can resend it elsewhere, never lose it.
        note = f"Your message may not have reached the new session; here it is so you can resend it:\n{text}"
        hooks.notify(note)
        final = result.get("final_response") or ""
        result = {**result, "final_response": f"{final}\n\n{note}" if final else note}
    return result
