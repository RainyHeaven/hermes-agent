"""Context rollover: the fresh agent must keep the source's real runtime settings, and an incomplete-origin
target stays fenced even with rollover off.

Regression for review finding 4. Real ``AIAgent`` instances built with explicit constructor ``cwd``,
``service_tier`` and ``run_budget_seconds`` (fake provider responses, temp SessionDB, real temp directories)
and the real gate/controller. The source row's ``cwd`` is the same real directory the source agent was built
with, so a mismatch can only come from the fresh agent the host's factory built.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from agent.context_rollover import load_handoff
from agent.context_rollover_gate import ARM_ATTR
from agent.context_rollover_runtime import RolloverHostAdapter, perform_rollover
from hermes_state_compression import FULL_COMPACTION_COUNT_KEY
from hermes_state_rollover import read_source
from tests.agent.test_context_rollover_runtime import (
    API_KEY, PROFILE, SID, Host, _no_sleep, _response, _tool_call, arm, model_calls, run_turn,
)
from tests.agent.test_context_rollover_runtime import db as _db_fixture

db = _db_fixture  # the shared fixture registers here under its own name

TASK = "Build the parser. Do not publish anything."


def _agent(db, sid, *, cwd, service_tier=None, run_budget_seconds=None, max_iterations=50, responses=()):
    """``build_agent`` with the constructor settings this file is about passed explicitly."""
    from run_agent import AIAgent

    tools = [{"type": "function", "function": {"name": "web_search", "description": "t",
                                                "parameters": {"type": "object", "properties": {}}}}]
    with (
        patch("model_tools.get_tool_definitions", return_value=tools),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(api_key=API_KEY, base_url="https://openrouter.ai/api/v1", model="test/model",
                        quiet_mode=True, skip_context_files=True, skip_memory=True, session_db=db, session_id=sid,
                        platform="cli", max_iterations=max_iterations, cwd=cwd, service_tier=service_tier,
                        run_budget_seconds=run_budget_seconds)
    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = list(responses) or [_response("resumed and finished")]
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = True
    agent.save_trajectories = False
    agent.valid_tool_names = {"web_search"}
    agent.tool_delay = 0
    return agent


def _dir(tmp_path, name):
    path = tmp_path / "workspaces" / name
    path.mkdir(parents=True)
    return str(path.resolve())


def _gated(db, *, cwd, service_tier=None):
    """A source at the threshold, built with real settings, and the real gated result of its next turn."""
    db.create_session(SID, "cli", model="test/model", profile_name=PROFILE, cwd=cwd,
                      model_config={FULL_COMPACTION_COUNT_KEY: 3})
    source = _agent(db, SID, cwd=cwd, service_tier=service_tier)
    assert source.session_cwd == cwd and source.service_tier == service_tier  # the constructor took effect
    arm(source)
    result = run_turn(source, TASK)
    assert result["context_rollover"]["status"] == "due" and model_calls(source) == 0
    return source, result


def _refused_before_anything_ran(db, host, source, outcome, *, expected_source_calls=0):
    """*expected_source_calls*: the source's own calls made BEFORE the rollover (its gated turn); the refusal
    itself must add none, and no candidate may make any."""
    assert outcome.status == "blocked", outcome
    assert host.installed == [] and host.turns == []
    assert model_calls(source) == expected_source_calls
    assert model_calls(*[a for a in host.made if a is not source]) == 0
    src = read_source(db, SID)
    assert src.claim["state"] == "prepared" and not src.ended


# --- cwd -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("fresh_cwd", ["elsewhere", None])
def test_fresh_agent_in_another_cwd_is_refused(db, tmp_path, fresh_cwd):
    work = _dir(tmp_path, "work")
    elsewhere = _dir(tmp_path, "elsewhere") if fresh_cwd else None
    source, result = _gated(db, cwd=work)
    host = Host(db, factory=lambda spec: _agent(spec.session_db, spec.session_id, cwd=elsewhere))
    outcome = perform_rollover(source, result, host.adapter())
    assert outcome.reason == "route_mismatch", outcome
    _refused_before_anything_ran(db, host, source, outcome)


def test_fresh_agent_in_the_same_cwd_resumes_there(db, tmp_path):
    work = _dir(tmp_path, "work")
    source, result = _gated(db, cwd=work)
    host = Host(db, factory=lambda spec: _agent(spec.session_db, spec.session_id, cwd=work))
    outcome = perform_rollover(source, result, host.adapter())
    assert outcome.status == "resumed", outcome
    new = outcome.agent
    assert new.session_cwd == work and model_calls(new) == 1
    assert db.get_session(new.session_id)["cwd"] == work
    route = load_handoff(outcome.handoff_path, profile=PROFILE, source_session_id=SID)["body"]["route"]
    assert route["cwd"] == work


# --- service tier ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("fresh_tier", [None, "ultrafast"])
def test_fresh_agent_on_another_service_tier_is_refused(db, tmp_path, fresh_tier):
    work = _dir(tmp_path, "work")
    source, result = _gated(db, cwd=work, service_tier="priority")
    host = Host(db, factory=lambda spec: _agent(spec.session_db, spec.session_id, cwd=work,
                                                service_tier=fresh_tier))
    outcome = perform_rollover(source, result, host.adapter())
    assert outcome.reason == "route_mismatch", outcome
    _refused_before_anything_ran(db, host, source, outcome)


def test_service_tier_reaches_the_handoff_and_the_fresh_agent(db, tmp_path):
    work = _dir(tmp_path, "work")
    source, result = _gated(db, cwd=work, service_tier="priority")
    specs = []

    def factory(spec):
        specs.append(spec)
        return _agent(spec.session_db, spec.session_id, cwd=work, service_tier=spec.route.get("service_tier"))

    host = Host(db, factory=factory)
    outcome = perform_rollover(source, result, host.adapter())
    route = load_handoff(outcome.handoff_path, profile=PROFILE, source_session_id=SID)["body"]["route"]
    assert route.get("service_tier") == "priority"
    assert [s.route.get("service_tier") for s in specs] == ["priority"]
    assert outcome.status == "resumed", outcome
    assert outcome.agent.service_tier == "priority" and model_calls(outcome.agent) == 1


# --- wall-clock run budget -------------------------------------------------------------------------------


def _recorder(agent, log, text):
    """Fake provider response that records the budget the turn is actually running under."""
    def respond(*_a, **_k):
        log.append({"started_at": agent._run_budget_started_at, "max_total": agent.iteration_budget.max_total,
                    "seconds": agent.run_budget_seconds})
        return _response(text)
    return respond


def _budgeted_source(db, work):
    """A source with a 600s run budget whose turn rolls over after 3 of 10 iterations."""
    db.create_session(SID, "cli", model="test/model", profile_name=PROFILE, cwd=work,
                      model_config={FULL_COMPACTION_COUNT_KEY: 2})
    source = _agent(db, SID, cwd=work, run_budget_seconds=600, max_iterations=10, responses=[
        _response("", tool_calls=[_tool_call("c1")]), _response("", tool_calls=[_tool_call("c2")]),
        _response("", tool_calls=[_tool_call("c3")]), _response("never requested")])
    assert source.run_budget_seconds == 600.0
    arm(source)
    calls = {"n": 0}

    def tool(*_a, **_k):
        calls["n"] += 1
        if calls["n"] == 3:  # a compaction commits during the third tool batch
            db.patch_session_model_config(SID, {FULL_COMPACTION_COUNT_KEY: 3})
        return "tool output"

    with (patch.object(source, "_save_trajectory"), patch.object(source, "_cleanup_task_resources"),
          patch("model_tools.handle_function_call", side_effect=tool)):
        result = source.run_conversation("long task")
    assert result["context_rollover"]["status"] == "due" and model_calls(source) == 3
    assert source.iteration_budget.remaining == 7 and source._run_budget_started_at is not None
    # The logical turn began two minutes ago: a fresh stamp could not be mistaken for the carried one.
    source._run_budget_started_at -= 120
    return source, result, source._run_budget_started_at


@pytest.mark.parametrize("fresh_budget", [None, 60, "not-a-duration", -5, True])
def test_fresh_agent_without_the_same_run_budget_is_refused(db, tmp_path, fresh_budget):
    work = _dir(tmp_path, "work")
    source, result, _started = _budgeted_source(db, work)
    log = []

    def factory(spec):
        agent = _agent(spec.session_db, spec.session_id, cwd=work, max_iterations=10,
                       run_budget_seconds=fresh_budget)
        agent.client.chat.completions.create.side_effect = _recorder(agent, log, "must not run")
        return agent

    host = Host(db, factory=factory)
    outcome = perform_rollover(source, result, host.adapter())
    assert log == []  # no request ran without the source's wall-clock cap
    assert outcome.reason == "run_budget_mismatch", outcome
    # The source's 3 calls all happened in _budgeted_source's gated turn, before the rollover.
    _refused_before_anything_ran(db, host, source, outcome, expected_source_calls=3)


def test_run_budget_carries_into_the_resume_turn_then_resets(db, tmp_path):
    work = _dir(tmp_path, "work")
    source, result, started = _budgeted_source(db, work)
    log = []

    def factory(spec):
        agent = _agent(spec.session_db, spec.session_id, cwd=work, max_iterations=10, run_budget_seconds=600)
        agent.client.chat.completions.create.side_effect = _recorder(agent, log, "resumed")
        return agent

    host = Host(db, factory=factory)
    outcome = perform_rollover(source, result, host.adapter())
    assert outcome.status == "resumed", outcome
    new = outcome.agent
    # The resume turn continues the logical turn: its start (so its deadline) and its remaining iterations.
    assert log == [{"started_at": started, "max_total": 7, "seconds": 600.0}]
    new.client.chat.completions.create.side_effect = _recorder(new, log, "a later real user turn")
    before = time.time()
    run_turn(new, "next real request")
    later = log[-1]
    assert later["max_total"] == 10 and later["seconds"] == 600.0
    assert later["started_at"] >= before > started + 100  # a real user turn gets its own fresh budget


def test_run_budget_changed_after_install_never_resumes(db, tmp_path):
    work = _dir(tmp_path, "work")
    source, result, _started = _budgeted_source(db, work)
    specs = []

    def factory(spec):
        specs.append(spec)
        return _agent(spec.session_db, spec.session_id, cwd=work, max_iterations=10,
                      run_budget_seconds=spec.run_budget_seconds)

    host = Host(db, factory=factory, on_install=lambda agent: setattr(agent, "run_budget_seconds", 60.0))
    outcome = perform_rollover(source, result, host.adapter())
    assert [s.run_budget_seconds for s in specs] == [600.0]  # the factory was told the source's cap
    assert outcome.status == "blocked" and outcome.reason == "run_budget_mismatch", outcome
    assert host.turns == [] and model_calls(*host.made) == 0 and model_calls(source) == 3
    assert read_source(db, SID).claim["state"] == "blocked"


# --- incomplete-origin target with rollover off ----------------------------------------------------------


def test_completed_target_takes_ordinary_turns_with_rollover_off(db, tmp_path):
    work = _dir(tmp_path, "work")
    source, result = _gated(db, cwd=work)
    host = Host(db, factory=lambda spec: _agent(spec.session_db, spec.session_id, cwd=work))
    outcome = perform_rollover(source, result, host.adapter())
    assert outcome.status == "resumed" and read_source(db, SID).claim["state"] == "completed", outcome
    agent = _agent(db, outcome.target_session_id, cwd=work, responses=[_response("normal answer")])
    assert getattr(agent, ARM_ATTR, None) is None
    turn = run_turn(agent, "an ordinary request")
    assert model_calls(agent) == 1 and turn["completed"] is True
    assert turn.get("context_rollover") is None and turn["final_response"] == "normal answer"


def _failing(host, step):
    adapter = host.adapter()

    def install(agent, outcome):
        host.installed.append((agent, outcome.target_session_id))
        raise RuntimeError("host swap failed")

    def retire(_agent):
        raise RuntimeError("retire failed")

    return RolloverHostAdapter(**{**adapter.__dict__, step: install if step == "install" else retire})


@pytest.mark.parametrize("reopen", ["same_agent_turned_off", "new_unarmed_agent", "new_agent_turned_off"])
@pytest.mark.parametrize("step", ["install", "retire"])
def test_incomplete_origin_target_refuses_an_ordinary_turn_with_rollover_off(db, tmp_path, step, reopen):
    work = _dir(tmp_path, "work")
    source, result = _gated(db, cwd=work)
    host = Host(db, factory=lambda spec: _agent(spec.session_db, spec.session_id, cwd=work))
    outcome = perform_rollover(source, result, _failing(host, step))
    assert outcome.status == "blocked" and outcome.reason == f"{step}_failed", outcome
    target_sid = outcome.target_session_id
    assert read_source(db, target_sid).origin is not None
    assert read_source(db, SID).claim["state"] == "blocked"
    if reopen == "same_agent_turned_off":
        agent = host.made[0]
        arm(agent, after=0)  # a config reload to rollover_after: 0
        agent.client.chat.completions.create.side_effect = [_response("must not run")]
    else:  # a new process opening the target session
        agent = _agent(db, target_sid, cwd=work, responses=[_response("must not run")])
        if reopen == "new_agent_turned_off":
            arm(agent, after=0)
    assert getattr(agent, ARM_ATTR, None) is None  # rollover really is off for this agent
    turn = run_turn(agent, "an ordinary request on the empty target")
    assert model_calls(agent) == 0
    assert turn["completed"] is False
    assert turn["context_rollover"]["reason"] == "rollover_origin_blocked"


@pytest.mark.parametrize("turned_off", [False, True])
def test_legacy_session_still_works_with_rollover_off(db, tmp_path, turned_off):
    work = _dir(tmp_path, "work")
    db.create_session("LEGACY", "cli", model="test/model", profile_name=PROFILE, cwd=work,
                      model_config={FULL_COMPACTION_COUNT_KEY: 7})
    agent = _agent(db, "LEGACY", cwd=work, responses=[_response("normal answer")])
    if turned_off:
        arm(agent, after=0)
    turn = run_turn(agent, "an ordinary request")
    assert model_calls(agent) == 1 and turn["completed"] is True
    assert turn.get("context_rollover") is None and turn["final_response"] == "normal answer"
