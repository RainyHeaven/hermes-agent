"""Context rollover through the real TUI turn pipeline (``server._run_prompt_submit``).

A real ``AIAgent`` source (platform ``tui``, temp SessionDB, fake provider) runs a user prompt that is due for
rollover. The TUI host (``tui_gateway/session_rollover.py``) builds the fresh agent through ``server._make_agent``
(patched here to a real ``AIAgent`` honouring the same kwargs), installs it on the session record, re-anchors
``session_key`` and runs the resume on the same turn thread. Only environment-heavy side paths are neutralized.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.context_rollover import RESUME_PROMPT_PREFIX
from agent.context_rollover_gate import ARM_ATTR
from hermes_state import SessionDB
from hermes_state_compression import FULL_COMPACTION_COUNT_KEY
from hermes_state_rollover import read_source
from tui_gateway import server, session_rollover

SID_UI = "ui-sid"
SOURCE = "TUI_ROLLOVER_SOURCE"
TASK = "Refactor the loader. Do not push anything."
_TOOLS = [{"type": "function", "function": {"name": "web_search", "description": "t",
                                            "parameters": {"type": "object", "properties": {}}}}]


def _response(content):
    message = SimpleNamespace(content=content, tool_calls=None, reasoning_content=None, reasoning=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], model="test/model",
                           usage=None)


def _inline_work(target, *, name, session=None):
    """``_start_session_work`` that runs the turn on the caller's thread, so the test observes its end state."""
    target()
    return threading.current_thread()


class _Provider:
    def __init__(self):
        self.requests = []

    def client(self):
        client = MagicMock()

        def create(*_a, **kwargs):
            self.requests.append(kwargs)
            return _response(f"tui reply {len(self.requests)}")
        client.chat.completions.create.side_effect = create
        return client


def _build(provider, db, session_id, *, cwd, model="test/model"):
    from unittest.mock import patch

    from run_agent import AIAgent
    with patch("model_tools.get_tool_definitions", return_value=_TOOLS), \
            patch("model_tools.check_toolset_requirements", return_value={}), \
            patch("agent.process_bootstrap.OpenAI"):
        agent = AIAgent(api_key="sk-test-key-0000", base_url="https://openrouter.ai/api/v1", model=model,
                        provider="openrouter", quiet_mode=True, skip_context_files=True, skip_memory=True,
                        session_db=db, session_id=session_id, platform="tui", max_iterations=30, cwd=cwd)
    agent.client = provider.client()
    agent._cached_system_prompt = "You are helpful."
    agent.save_trajectories = False
    agent.valid_tool_names = {"web_search"}
    return agent


@pytest.fixture
def env(tmp_path, monkeypatch):
    import time as _time

    from agent import retry_utils
    monkeypatch.setattr(_time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(retry_utils, "jittered_backoff", lambda *a, **k: 0.0)
    db = SessionDB(tmp_path / "state.db")
    work = tmp_path / "work"
    work.mkdir()
    provider, emitted, built = _Provider(), [], []
    profile = server._current_profile_name()

    def make_agent(sid, key, session_id=None, session_db=None, model_override=None, cwd_override=None, **_kw):
        assert session_id == key and isinstance(model_override, dict)  # the source's exact route is requested
        agent = _build(provider, session_db, session_id, cwd=cwd_override, model=model_override["model"])
        built.append(agent)
        return agent

    # Build agents (their logging listener starts a real thread) before the turn thread is made inline.
    db.create_session(SOURCE, "tui", model="test/model", profile_name=profile, cwd=str(work.resolve()),
                      model_config={FULL_COMPACTION_COUNT_KEY: 3})
    source = _build(provider, db, SOURCE, cwd=str(work.resolve()))
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    monkeypatch.setattr(server, "_start_session_work", _inline_work)
    monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None: emitted.append((event, payload)))
    real_make_agent = server._make_agent
    monkeypatch.setattr(server, "_make_agent", make_agent)
    monkeypatch.setattr(server, "_load_cfg", lambda: {"compression": {"rollover_after": 3}})
    for name in ("_wire_callbacks", "_sync_agent_model_with_config", "_sync_agent_compression_with_config",
                 "_register_session_cwd", "_restart_slash_worker"):
        monkeypatch.setattr(server, name, lambda *a, **k: None)
    monkeypatch.setattr(server, "_session_cwd", lambda session: str(work))
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_get_usage", lambda agent: {})
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda sid, session: None)
    monkeypatch.setattr(server, "_transfer_active_session_slot", lambda sid, session, new_session_id: True)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda session: True)
    session = {"agent": source, "session_key": SOURCE, "history": [], "history_lock": threading.Lock(),
               "history_version": 0, "running": True, "attached_images": [], "image_counter": 0, "cols": 80,
               "slash_worker": None, "show_reasoning": False, "tool_progress_mode": "all", "inflight_turn": None,
               "source": "tui", "transport": None}
    yield SimpleNamespace(db=db, provider=provider, emitted=emitted, built=built, session=session, source=source,
                          real_make_agent=real_make_agent)
    db.close()


def _submit(env, text, *, user=True):
    env.session["running"] = True
    env.session["_rollover_user_turn"] = user
    server._run_prompt_submit("rid", SID_UI, env.session, text)


def _completes(env):
    return [p for e, p in env.emitted if e == "message.complete"]


def test_tui_prompt_rolls_over_into_a_fresh_agent_and_session_key(env):
    _submit(env, TASK)
    new = env.session["agent"]
    assert new is not env.source and env.built == [new]
    assert env.session["session_key"] == new.session_id != SOURCE
    assert getattr(new, ARM_ATTR) is not None
    assert len(env.provider.requests) == 1  # only the fresh agent's resume reached the provider
    user_rows = [m["content"] for m in env.provider.requests[0]["messages"] if m["role"] == "user"]
    assert user_rows[-1].startswith(RESUME_PROMPT_PREFIX) and "Do not push anything." in user_rows[-1]
    assert _completes(env)[-1]["text"] == "tui reply 1"
    # The session history is the target's own transcript, never the source's turn.
    assert env.session["history"] and all(TASK not in str(m.get("content")) for m in env.session["history"]
                                          if m.get("role") == "user" and not str(m.get("content")).startswith(
                                              RESUME_PROMPT_PREFIX))
    assert any(e == "status.update" and p["kind"] == "context_rollover" for e, p in env.emitted)
    assert read_source(env.db, SOURCE).claim["state"] == "completed"
    # The handoff row is model input, hidden from the transcript view.
    rows = env.db.get_messages(new.session_id)
    assert any(r["role"] == "user" and r.get("display_kind") == "hidden" for r in rows)
    _submit(env, "next step")
    assert _completes(env)[-1]["text"] == "tui reply 2" and env.session["agent"] is new


def test_tui_prompt_queued_during_the_swap_resumes_the_held_rollover(env, monkeypatch):
    """Input that arrives mid-swap defers the resume; the queued prompt then drains as the user's own turn and
    runs the held resume with that text as its latest instruction (never an ordinary turn on the target)."""
    real_install = session_rollover._install
    seen = []

    def install_then_queue(sid, session, st, agent):
        real_install(sid, session, st, agent)
        session["queued_prompt"] = {"text": "queued later", "transport": None}  # input arrived mid-swap
        monkeypatch.setattr(session_rollover, "_install", real_install)
    monkeypatch.setattr(session_rollover, "_install", install_then_queue)
    real_resume = session_rollover.resume_held

    def spy(outcome, text, history, hooks):
        seen.append((outcome.status, outcome.reason, text))
        return real_resume(outcome, text, history, hooks)
    monkeypatch.setattr(session_rollover, "resume_held", spy)
    _submit(env, TASK)
    assert seen == [("deferred", "queued_input", "queued later")]
    assert len(env.provider.requests) == 1
    resume = [m["content"] for m in env.provider.requests[0]["messages"] if m["role"] == "user"][-1]
    assert resume.startswith(RESUME_PROMPT_PREFIX) and resume.endswith("queued later")
    assert session_rollover.held(env.session) is None and env.session.get("queued_prompt") is None
    assert read_source(env.db, SOURCE).claim["state"] == "completed"


def test_tui_synthesized_turn_never_resumes_a_held_rollover(env, monkeypatch):
    real_install = session_rollover._install

    def install_then_cancel(sid, session, st, agent):
        real_install(sid, session, st, agent)
        session["_turn_cancel_requested"] = True
    monkeypatch.setattr(session_rollover, "_install", install_then_cancel)
    _submit(env, TASK)
    assert session_rollover.held(env.session) is not None
    monkeypatch.setattr(session_rollover, "_install", real_install)
    env.session["_turn_cancel_requested"] = False
    _submit(env, "goal continuation prompt", user=False)  # e.g. a goal follow-up or a heartbeat
    assert env.provider.requests == [] and session_rollover.held(env.session) is not None
    assert read_source(env.db, SOURCE).claim["state"] == "installed"
    _submit(env, "continue please")
    assert len(env.provider.requests) == 1 and read_source(env.db, SOURCE).claim["state"] == "completed"


def test_tui_rollover_off_keeps_the_ordinary_turn(env, monkeypatch):
    monkeypatch.setattr(server, "_load_cfg", lambda: {"compression": {"rollover_after": 0}})
    _submit(env, TASK)
    assert env.session["agent"] is env.source and env.session["session_key"] == SOURCE
    assert _completes(env)[-1]["text"] == "tui reply 1" and read_source(env.db, SOURCE).claim is None


def test_tui_desktop_platform_is_never_armed(env):
    env.source.platform = "desktop"
    _submit(env, TASK)
    assert env.session["agent"] is env.source and getattr(env.source, ARM_ATTR) is None
    assert read_source(env.db, SOURCE).claim is None


def test_tui_factory_keeps_the_sources_exact_route_over_changed_config(env, monkeypatch):
    """The real ``server._make_agent`` with ``exact_route``: a config whose toolsets/reasoning/tier moved after the
    source was built must not leak into the fresh agent (that would fail ``route_mismatch`` on every turn)."""
    captured = {}

    class _Agent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
    monkeypatch.setattr("run_agent.AIAgent", _Agent)
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda platform=None: ["changed-in-config"])
    monkeypatch.setattr(server, "_load_disabled_toolsets", lambda: ["also-changed"])
    monkeypatch.setattr(server, "_load_reasoning_config", lambda model: {"enabled": True, "effort": "high"})
    monkeypatch.setattr(server, "_load_service_tier", lambda: "priority")
    monkeypatch.setattr(server, "_resolve_agent_model_runtime", lambda mo, po: ("test/model", {"provider": "openrouter"}))
    monkeypatch.setattr(server, "_startup_system_prompt", lambda cfg, key: "")
    monkeypatch.setattr(server, "_agent_cbs", lambda sid: {})
    route = {"model": "test/model", "enabled_toolsets": ["web"], "disabled_toolsets": [], "reasoning_config": None,
             "service_tier": None}
    from tui_gateway import server as real_server
    real_make = env.real_make_agent
    real_make(SID_UI, "NEW", session_id="NEW", session_db=env.db, model_override={"model": "test/model"},
              exact_route=route)
    assert captured["enabled_toolsets"] == ["web"] and captured["disabled_toolsets"] == []
    assert captured["reasoning_config"] is None and captured["service_tier"] is None
    assert real_server is server


def test_tui_blocked_held_resume_hands_the_users_text_back(env, monkeypatch):
    real_install = session_rollover._install

    def install_then_cancel(sid, session, st, agent):
        real_install(sid, session, st, agent)
        session["_turn_cancel_requested"] = True
        monkeypatch.setattr(session_rollover, "_install", real_install)
    monkeypatch.setattr(session_rollover, "_install", install_then_cancel)
    _submit(env, TASK)
    assert session_rollover.held(env.session) is not None
    env.session["_turn_cancel_requested"] = False
    # Something fenced the target durably meanwhile: the resume cannot start.
    from hermes_state_rollover import advance
    claim = read_source(env.db, SOURCE).claim
    advance(env.db, SOURCE, claim["rollover_id"], "installed", "blocked", updates={"reason": "test"})
    _submit(env, "please keep this text")
    assert env.provider.requests == []
    assert "please keep this text" in _completes(env)[-1]["text"]
