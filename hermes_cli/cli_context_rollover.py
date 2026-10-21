"""Context rollover host for the classic CLI (``HermesCLI``).

Wiring (``cli_chat_turn_mixin``): :func:`arm` before the agent thread starts, :func:`run_held_resume` instead of
an ordinary turn while a resume is held back, :func:`after_turn` in the agent thread once
``run_conversation`` returned. Every callback below runs on that agent thread and returns only when its work
finished, so the controller's resume is synchronous by construction.

The fresh agent is built through the same constructor kwargs as ``_init_agent`` with the host's current
credentials (memory only); only the session id and the source's route/cwd/budget differ. The old agent's
clients are released (``release_clients``), never ``close()``d: the SessionDB stays the CLI's.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from agent.context_rollover_gate import HostState
from agent.context_rollover_host import HostHooks, arm_for_turn, drive, held_for, resume_held
from agent.context_rollover_runtime import FreshAgentSpec, RolloverHostAdapter

logger = logging.getLogger(__name__)

HELD_ATTR = "_context_rollover_held"


def _profile() -> str:
    from hermes_cli.profiles import get_active_profile_name

    return get_active_profile_name()


def host_state(cli: Any) -> HostState:
    """What the CLI itself is waiting on. Processes count only when owned by this conversation."""
    from tools.process_registry import process_registry

    agent = cli.agent
    sid = getattr(agent, "session_id", None) or cli.session_id
    pending = getattr(cli, "_pending_input", None)
    interrupts = getattr(cli, "_interrupt_queue", None)
    return HostState(
        children_active=False,  # the controller reads the agent's own children under its lock
        processes_active=bool(process_registry.has_active_processes(sid)),
        approval_pending=bool(cli._approval_state or cli._sudo_state or cli._secret_state),
        clarify_pending=bool(cli._clarify_state or cli._clarify_freetext),
        queued_input=bool((pending is not None and not pending.empty())
                          or (interrupts is not None and not interrupts.empty())),
        stop_requested=bool(getattr(cli, "_should_exit", False)),
    )


def _compression_cfg(cli: Any) -> Any:
    """The CLI's ``compression`` section (its own ``config``, else the module ``CLI_CONFIG``); None = off."""
    config = getattr(cli, "config", None)
    if not isinstance(config, dict):
        import cli as _cli

        config = getattr(_cli, "CLI_CONFIG", None)
    return config.get("compression") if isinstance(config, dict) else None


def arm(cli: Any) -> None:
    """(Re-)arm the CLI's agent from the live config. Single-query (``-q``) runs never roll over."""
    agent = getattr(cli, "agent", None)
    if agent is None:
        return
    arm_for_turn(agent, "cli", compression_cfg=_compression_cfg(cli), profile=_profile(),
                 host_state=lambda: host_state(cli), eligible=not getattr(cli, "_single_query_mode", False))


def source_runtime(source: Any) -> dict:
    """The runtime (credentials included, memory only) the source agent is actually running on, so a fresh
    agent reproduces exactly its route, fallback activation included, never the config default."""
    return {"api_key": getattr(source, "api_key", None), "base_url": source.base_url, "provider": source.provider,
            "requested_provider": getattr(source, "requested_provider", source.provider),
            "api_mode": source.api_mode, "command": getattr(source, "acp_command", None),
            "args": list(getattr(source, "acp_args", None) or []),
            "credential_pool": getattr(source, "_credential_pool", None)}


def _make_agent(cli: Any, spec: FreshAgentSpec) -> Any:
    from cli import _cprint
    from run_agent import AIAgent

    source, route = cli.agent, spec.route
    kwargs = cli._agent_build_kwargs(route["model"], source_runtime(source),
                                     getattr(source, "request_overrides", None))
    kwargs.update(session_id=spec.session_id, session_db=spec.session_db,
                  run_budget_seconds=spec.run_budget_seconds, service_tier=route.get("service_tier"),
                  reasoning_config=route.get("reasoning_config"),
                  enabled_toolsets=route.get("enabled_toolsets"), disabled_toolsets=route.get("disabled_toolsets"),
                  max_iterations=source.max_iterations)
    if route.get("cwd"):
        kwargs["cwd"] = route["cwd"]
    agent = AIAgent(**kwargs)
    agent._print_fn = cli._agent_status_print
    _cprint(f"  Context rollover: starting a fresh agent on session {spec.session_id}")
    return agent


def _install(cli: Any, agent: Any, _outcome: Any) -> None:
    """Swap the CLI onto the fresh agent and its empty history; the approval/yolo scope follows the session.
    All or nothing: a failure puts every pointer back on the source, so nothing runs on a fenced target."""
    import cli as _cli
    from hermes_cli.cli_shutdown import _sync_process_session_id
    from tools.approval_yolo import transfer_session_yolo

    saved = (cli.agent, cli.session_id, cli.conversation_history, cli._pending_title, _cli._active_agent_ref)
    try:
        cli.agent, _cli._active_agent_ref, cli.conversation_history = agent, agent, []
        transfer_session_yolo(saved[1], agent.session_id)
        cli.session_id = agent.session_id
        _sync_process_session_id(cli.session_id)
        cli._write_terminal_breadcrumb()
        cli._pending_title = None
        arm(cli)
    except BaseException:
        cli.agent, cli.session_id, cli.conversation_history, cli._pending_title, _cli._active_agent_ref = saved
        _sync_process_session_id(cli.session_id)
        raise


def _retire(cli: Any, agent: Any) -> None:
    """Release *agent*'s clients (never its SessionDB). A candidate the CLI already adopted before its install
    failed is dropped as well, so the next turn rebuilds instead of running on a fenced agent."""
    if agent is None:
        return
    if agent is cli.agent:  # defensive: _install restores the source pointer before a candidate is retired
        cli.agent = None
    if hasattr(agent, "release_clients"):
        agent.release_clients()


def _run_turn(cli: Any, agent: Any, prompt: str) -> dict:
    """The resume turn, on this (agent) thread, the CLI's normal streaming path; returns once it ended. The
    approval scope is rebound to the new session (its yolo state moved there at install)."""
    from tools.approval_context import reset_current_session_key, set_current_session_key

    cli._flush_stream()  # close whatever the source turn streamed before the resume starts its own box
    cli._reset_stream_state()
    token = set_current_session_key(agent.session_id)
    try:
        # The handoff is model input, not a user row in the transcript view (/resume hides ``hidden`` rows).
        result = agent.run_conversation(user_message=prompt, conversation_history=[], task_id=agent.session_id,
                                        persist_user_display_kind="hidden")
    finally:
        reset_current_session_key(token)
    if isinstance(result, dict) and isinstance(result.get("messages"), list):
        cli.conversation_history = result["messages"]
    return result


def _hooks(cli: Any) -> HostHooks:
    from cli import _cprint

    def hold(outcome: Optional[Any]) -> None:
        setattr(cli, HELD_ATTR, outcome)

    adapter = RolloverHostAdapter(
        host_state=lambda: host_state(cli), make_agent=lambda spec: _make_agent(cli, spec),
        install=lambda agent, outcome: _install(cli, agent, outcome), retire=lambda agent: _retire(cli, agent),
        run_turn=lambda agent, prompt: _run_turn(cli, agent, prompt))
    return HostHooks(adapter=adapter, notify=lambda text: _cprint(f"\n  {text}"),
                     requeue=lambda text: cli._pending_input.put(text), hold=hold)


def suppresses_followups(cli: Any) -> bool:
    """No automatic follow-up (goal/loop continuation) after a rollover stop or while a resume is held back:
    only the user's own next message may continue the task."""
    result = getattr(cli, "_last_turn_result", None)
    return getattr(cli, HELD_ATTR, None) is not None or (
        isinstance(result, dict) and isinstance(result.get("context_rollover"), dict))


def held(cli: Any) -> Optional[Any]:
    """The deferred outcome the next message resumes, while it still matches the CLI's live agent."""
    outcome = held_for(getattr(cli, HELD_ATTR, None), cli.agent, cli.session_id)
    if outcome is None:
        setattr(cli, HELD_ATTR, None)
    return outcome


def after_turn(cli: Any, result: Any) -> Any:
    """The result chat() should show for the turn that just ran (drives a due rollover to completion)."""
    try:
        return drive(cli.agent, result, _hooks(cli))
    except Exception:  # health: allow BLE001 -- the controller already fails closed; never lose the user's turn
        logger.warning("context rollover host failed", exc_info=True)
        return result


def run_held_resume(cli: Any, outcome: Any, message: Any) -> Any:
    """Run the held-back resume with *message* as its latest instruction instead of an ordinary turn."""
    from agent.context_rollover import _content_text

    text = _content_text(message) if not isinstance(message, str) else message
    try:
        return resume_held(outcome, text, cli.conversation_history, _hooks(cli))
    except Exception:  # health: allow BLE001 -- as in after_turn; the message is handed back, not dropped
        logger.warning("context rollover resume failed", exc_info=True)
        setattr(cli, HELD_ATTR, None)
        cli._pending_input.put(message)
        return {"final_response": "Context rollover resume failed; your message was kept.", "messages":
                list(cli.conversation_history), "api_calls": 0, "completed": False, "failed": True,
                "error": "context rollover resume failed"}
