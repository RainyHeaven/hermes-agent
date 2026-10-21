"""Context rollover through the real classic-CLI turn path (``HermesCLI.chat``).

A real ``HermesCLI`` (prompt_toolkit stubbed) drives a real ``AIAgent`` source on a temp SessionDB with a fake
provider. The fresh agent is built by the CLI host's own factory (``_agent_build_kwargs`` → ``AIAgent``), so
the swap, history reset, session id sync, deferral, held resume and follow-up suppression are the CLI's own.
"""

from __future__ import annotations

import importlib
import queue
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.context_rollover import RESUME_PROMPT_PREFIX
from agent.context_rollover_gate import ARM_ATTR
from hermes_state import SessionDB
from hermes_state_compression import FULL_COMPACTION_COUNT_KEY
from hermes_state_rollover import SOURCE_END_REASON, read_source

SID = "CLI_ROLLOVER_SOURCE"
TASK = "Build the parser. Do not publish anything."
_TOOLS = [{"type": "function", "function": {"name": "web_search", "description": "t",
                                            "parameters": {"type": "object", "properties": {}}}}]


def _response(content):
    message = SimpleNamespace(content=content, tool_calls=None, reasoning_content=None, reasoning=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], model="test/model",
                           usage=None)


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    import time as _time

    from agent import retry_utils
    monkeypatch.setattr(_time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(retry_utils, "jittered_backoff", lambda *a, **k: 0.0)


def _make_cli(after):
    config = {"model": {"default": "test/model", "base_url": "https://openrouter.ai/api/v1", "provider": "auto"},
              "display": {"compact": False, "tool_progress": "all"}, "agent": {}, "terminal": {"env_type": "local"},
              "compression": {"enabled": True, "rollover_after": after}}
    stubs = {name: MagicMock() for name in (
        "prompt_toolkit", "prompt_toolkit.history", "prompt_toolkit.styles", "prompt_toolkit.patch_stdout",
        "prompt_toolkit.application", "prompt_toolkit.layout", "prompt_toolkit.layout.processors",
        "prompt_toolkit.filters", "prompt_toolkit.layout.dimension", "prompt_toolkit.layout.menus",
        "prompt_toolkit.widgets", "prompt_toolkit.key_binding", "prompt_toolkit.completion",
        "prompt_toolkit.formatted_text", "prompt_toolkit.auto_suggest")}
    with patch.dict(sys.modules, stubs), patch.dict("os.environ", {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}):
        import cli as cli_mod

        cli_mod = importlib.reload(cli_mod)
        with patch.object(cli_mod, "get_tool_definitions", return_value=[]), \
                patch.dict(cli_mod.__dict__, {"CLI_CONFIG": config}):
            obj = cli_mod.HermesCLI()
    obj.config = config
    return obj


class _Provider:
    """Every fresh agent the CLI factory builds gets this fake client; it records each request."""

    def __init__(self):
        self.requests = []

    def client(self):
        client = MagicMock()

        def create(*_a, **kwargs):
            self.requests.append(kwargs)
            return _response(f"resumed reply {len(self.requests)}")
        client.chat.completions.create.side_effect = create
        return client


@pytest.fixture
def env(tmp_path, monkeypatch):
    from hermes_cli.profiles import get_active_profile_name

    db = SessionDB(tmp_path / "state.db")
    work = tmp_path / "work"
    work.mkdir()
    provider = _Provider()
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda *a, **k: _TOOLS)
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda *a, **k: {})
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", lambda *a, **k: provider.client())
    yield SimpleNamespace(db=db, work=str(work.resolve()), provider=provider, profile=get_active_profile_name())
    db.close()


def _source_cli(env, *, after=3, count=3):
    from run_agent import AIAgent

    env.db.create_session(SID, "cli", model="test/model", profile_name=env.profile, cwd=env.work,
                          model_config={FULL_COMPACTION_COUNT_KEY: count})
    cli = _make_cli(after)
    cli.session_id, cli._session_db = SID, env.db
    cli.api_key, cli.base_url, cli.provider, cli.api_mode = "sk-test-key-0000", "https://openrouter.ai/api/v1", \
        "openrouter", "chat_completions"
    kwargs = cli._agent_build_kwargs("test/model", {"api_key": cli.api_key, "base_url": cli.base_url,
                                                    "provider": cli.provider, "api_mode": cli.api_mode}, None)
    kwargs.update(cwd=env.work, skip_context_files=True, skip_memory=True)
    cli.agent = AIAgent(**kwargs)
    cli.ignore_rules = True  # the fresh agent is built from the same CLI settings
    cli._interrupt_queue, cli._pending_input = queue.Queue(), queue.Queue()
    return cli


def _chat(cli, text):
    with patch.object(cli, "_ensure_runtime_credentials", return_value=True), \
            patch.object(cli, "_resolve_turn_agent_config", return_value={
                "signature": cli._active_agent_route_signature, "model": None, "runtime": None,
                "request_overrides": None}), \
            patch.object(cli, "_init_agent", return_value=True), \
            patch.object(cli, "_chat_print_response_panel", lambda *a, **k: None), \
            patch.object(type(cli.agent), "_save_trajectory", lambda *a, **k: None), \
            patch("model_tools.handle_function_call", return_value="tool output"):
        return cli.chat(text)


def _user_texts(request):
    return [m["content"] for m in request["messages"] if m["role"] == "user"]


def test_cli_rolls_over_into_a_fresh_session_and_agent_on_the_users_turn(env):
    cli = _source_cli(env)
    source = cli.agent
    response = _chat(cli, TASK)
    new = cli.agent
    assert new is not source and new.session_id != SID and cli.session_id == new.session_id
    assert getattr(new, ARM_ATTR) is not None  # the installed agent is re-armed for its own next rollover
    assert len(env.provider.requests) == 1  # one model request in total: the fresh agent's resume
    resume = _user_texts(env.provider.requests[0])[-1]
    assert resume.startswith(RESUME_PROMPT_PREFIX) and "Do not publish anything." in resume
    assert response == "resumed reply 1"
    assert [m["role"] for m in cli.conversation_history][-1] == "assistant"
    assert all(not str(m.get("content", "")).startswith("Context rollover:") for m in cli.conversation_history)
    rows = env.db.get_messages(new.session_id)
    handoff = next(r for r in rows if r["role"] == "user")
    assert handoff["content"].startswith(RESUME_PROMPT_PREFIX) and handoff.get("display_kind") == "hidden"
    src = read_source(env.db, SID)
    assert src.ended and src.claim["state"] == "completed"
    assert env.db.get_session(SID)["end_reason"] == SOURCE_END_REASON
    target = read_source(env.db, new.session_id)
    assert target.full_compaction_count == 0 and target.origin["source_session_id"] == SID
    # The next ordinary turn runs normally on the new session.
    assert _chat(cli, "next step") == "resumed reply 2"
    assert _user_texts(env.provider.requests[1])[-1] == "next step"


def test_cli_holds_a_deferred_resume_until_the_users_next_message(env, monkeypatch):
    from hermes_cli import cli_context_rollover as host

    cli = _source_cli(env)
    real_install = host._install

    def install_then_busy(cli_, agent, outcome):
        real_install(cli_, agent, outcome)
        cli_._clarify_state = {"question": "pending"}  # the host became busy right after the swap
    monkeypatch.setattr(host, "_install", install_then_busy)
    _chat(cli, TASK)
    held = host.held(cli)
    assert held is not None and held.reason == "clarify_pending" and env.provider.requests == []
    assert host.suppresses_followups(cli)  # goal/loop continuations may not resume it on their own
    assert read_source(env.db, SID).claim["state"] == "installed"
    cli._clarify_state = None
    monkeypatch.setattr(host, "_install", real_install)
    response = _chat(cli, "use the faster algorithm")
    assert response == "resumed reply 1" and len(env.provider.requests) == 1
    resume = _user_texts(env.provider.requests[0])[-1]
    assert resume.startswith(RESUME_PROMPT_PREFIX) and resume.endswith("use the faster algorithm")
    assert host.held(cli) is None and read_source(env.db, SID).claim["state"] == "completed"


def test_cli_rollover_off_runs_the_ordinary_turn(env):
    cli = _source_cli(env, after=0)
    source = cli.agent
    assert _chat(cli, TASK) == "resumed reply 1"
    assert cli.agent is source and cli.session_id == SID and getattr(source, ARM_ATTR) is None
    assert read_source(env.db, SID).claim is None


def test_cli_single_query_mode_is_never_armed(env):
    cli = _source_cli(env)
    cli._single_query_mode = True
    source = cli.agent
    _chat(cli, TASK)
    assert cli.agent is source and getattr(source, ARM_ATTR) is None
    assert read_source(env.db, SID).claim is None
