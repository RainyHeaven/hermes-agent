"""Safe-boundary gate for context rollover (``agent/context_rollover_gate.py``) in the real turn loop.

A turn on an armed CLI/TUI agent stops BEFORE the next main-model call once the durable full-compaction
count reaches ``compression.rollover_after``: at turn entry (the context build may have compacted) and before
every API attempt (a preflight or reactive overflow compaction inside the turn). The current synchronous tool
batch finishes first. Anything unresolved or unknown blocks instead, also without a model call. Unarmed
agents behave exactly as before. Fake provider responses only; real temp SessionDB.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.context_rollover import RolloverHost
from agent.context_rollover_gate import (
    ROLLOVER_DISPLAY_KIND, HostState, arm_context_rollover, check_rollover,
)
from hermes_state import SessionDB
from hermes_state_compression import FULL_COMPACTION_COUNT_KEY, ROLLOVER_CLAIM_KEY

PROFILE = "rollover-test"
SID = "GATE_SOURCE"
CLI = RolloverHost(surface="cli", resumes_in_fresh_agent=True)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    import time as _time

    from agent import retry_utils
    monkeypatch.setattr(_time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(retry_utils, "jittered_backoff", lambda *a, **k: 0.0)


def _tool_defs(*names):
    return [{"type": "function", "function": {"name": n, "description": "t",
                                              "parameters": {"type": "object", "properties": {}}}}
            for n in names]


def _response(content="done", tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls, reasoning_content=None, reasoning=None)
    finish = "tool_calls" if tool_calls else "stop"
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish)],
                           model="test/model", usage=None)


def _tool_call(call_id="call_1", name="web_search"):
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name=name, arguments="{}"))


def clean_state(**overrides):
    values = dict(children_active=False, processes_active=False, approval_pending=False,
                  clarify_pending=False, queued_input=False, stop_requested=False)
    values.update(overrides)
    return HostState(**values)


@pytest.fixture
def db(tmp_path):
    store = SessionDB(tmp_path / "state.db")
    yield store
    store.close()


def make_agent(db, sid=SID, *, count=0, seed=True):
    from run_agent import AIAgent

    if seed:
        db.create_session(sid, "cli", model="test/model", profile_name=PROFILE,
                          model_config={FULL_COMPACTION_COUNT_KEY: count} if count else None)
    with (
        patch("model_tools.get_tool_definitions", return_value=_tool_defs("web_search")),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1",
                        model="test/model", quiet_mode=True, skip_context_files=True, skip_memory=True,
                        session_db=db, session_id=sid, platform="cli")
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = True
    agent.save_trajectories = False
    agent.valid_tool_names = {"web_search"}
    agent.tool_delay = 0
    return agent


def arm(agent, after=3, state=clean_state):
    return arm_context_rollover(agent, CLI, compression_cfg={"rollover_after": after}, profile=PROFILE,
                                host_state=state)


def run(agent, text="please continue the task"):
    with patch.object(agent, "_save_trajectory"), patch.object(agent, "_cleanup_task_resources"):
        return agent.run_conversation(text)


def bump_count(db, sid=SID):
    current = json.loads(db.get_session(sid)["model_config"] or "{}").get(FULL_COMPACTION_COUNT_KEY, 0)
    db.patch_session_model_config(sid, {FULL_COMPACTION_COUNT_KEY: current + 1})


def durable_rows(db, sid=SID):
    return db.get_messages_as_conversation(sid)


def test_threshold_two_yields_where_three_continues(db):
    calm = make_agent(db, "CALM", count=2)
    assert arm(calm, after=3) is None
    calm.client.chat.completions.create.return_value = _response("normal answer")
    result = run(calm)
    assert result["completed"] is True and "context_rollover" not in result
    assert calm.client.chat.completions.create.call_count == 1

    due = make_agent(db, SID, count=2)
    assert arm(due, after=2) is None
    result = run(due)
    assert due.client.chat.completions.create.call_count == 0  # no model call once due
    rollover = result["context_rollover"]
    assert rollover["status"] == "due" and rollover["full_compaction_count"] == 2
    assert rollover["source_session_id"] == SID and rollover["profile"] == PROFILE
    assert result["completed"] is False and result["failed"] is False and result["api_calls"] == 0
    rows = durable_rows(db)
    assert [r["role"] for r in rows] == ["user", "assistant"]
    assert rows[0]["content"] == "please continue the task"
    assert rows[1]["display_kind"] == ROLLOVER_DISPLAY_KIND  # not a generic failed-turn boundary
    assert ROLLOVER_CLAIM_KEY not in json.loads(db.get_session(SID)["model_config"])  # the gate only reads


@pytest.mark.parametrize("arming", [
    dict(host=CLI, cfg={"rollover_after": 0}),
    dict(host=CLI, cfg={}),
    dict(host=CLI, cfg={"rollover_after": "3"}),
    dict(host=CLI, cfg={"rollover_after": True}),
    dict(host=CLI, cfg={"rollover_after": -1}),
    dict(host=RolloverHost(surface="telegram", resumes_in_fresh_agent=True), cfg={"rollover_after": 1}),
    dict(host=RolloverHost(surface="cli", resumes_in_fresh_agent=False), cfg={"rollover_after": 1}),
    dict(host=None, cfg={"rollover_after": 1}),
])
def test_disabled_or_ineligible_hosts_keep_existing_behaviour(db, arming):
    agent = make_agent(db, count=7)
    arm_context_rollover(agent, arming["host"], compression_cfg=arming["cfg"], profile=PROFILE,
                         host_state=clean_state)
    before = db.get_session(SID)["model_config"]
    agent.client.chat.completions.create.return_value = _response("normal answer")
    result = run(agent)
    assert result["completed"] is True and result["final_response"] == "normal answer"
    assert "context_rollover" not in result
    assert db.get_session(SID)["model_config"] == before


def test_delegated_and_forked_agents_are_never_armed(db):
    child = make_agent(db, count=7)
    child._delegate_depth = 1
    assert arm(child, after=1) is not None
    assert check_rollover(child) is None
    fork = make_agent(db, "FORK", count=7)
    fork._persist_disabled = True
    assert arm(fork, after=1) is not None
    assert check_rollover(fork) is None


def test_reactive_overflow_compaction_yields_before_the_retry(db):
    agent = make_agent(db, count=2)
    arm(agent, after=3)
    overflow = Exception("Request entity too large")
    overflow.status_code = 413
    agent.client.chat.completions.create.side_effect = [overflow, _response("must not be requested")]

    def _committed_compaction(messages, *_a, **_k):
        bump_count(db)  # the durable count the compaction transaction writes
        return messages[-1:], "You are helpful."

    history = [{"role": "user", "content": "earlier question"}, {"role": "assistant", "content": "earlier answer"}]
    with (patch.object(agent, "_compress_context", side_effect=_committed_compaction),
          patch.object(agent, "_save_trajectory"), patch.object(agent, "_cleanup_task_resources")):
        result = agent.run_conversation("please continue the task", conversation_history=history)
    assert agent.client.chat.completions.create.call_count == 1
    assert result["context_rollover"]["status"] == "due"
    assert result["context_rollover"]["full_compaction_count"] == 3
    # The compaction restart refunds the overflowed iteration (existing loop accounting); the yield then
    # refunds the iteration it never sent, so the budget does not count the request that was not made.
    assert agent.iteration_budget.used == result["api_calls"] == 0


def test_preflight_compaction_yields_before_the_request_it_prepared(db):
    agent = make_agent(db, count=2)
    arm(agent, after=3)
    agent.client.chat.completions.create.return_value = _response("must not be requested")

    def _preflight_committed(_agent, verdict, **_kwargs):
        bump_count(db)  # the pre-API pass committed the third full compaction
        return verdict

    with patch("agent.turn_preflight_gate.run_preflight_compression", side_effect=_preflight_committed):
        result = run(agent)
    assert agent.client.chat.completions.create.call_count == 0
    assert result["context_rollover"]["status"] == "due"
    assert result["api_calls"] == 0 and agent.iteration_budget.used == 0


def test_tool_batch_finishes_then_post_tool_compaction_yields(db):
    agent = make_agent(db, count=2)
    arm(agent, after=3)
    agent.client.chat.completions.create.side_effect = [
        _response("", tool_calls=[_tool_call()]), _response("must not be requested")]
    ran = []

    def _tool(*_a, **_k):
        ran.append("web_search")
        bump_count(db)  # a compaction committed while the batch ran
        return "tool output"

    with patch("model_tools.handle_function_call", side_effect=_tool):
        result = run(agent)
    assert ran == ["web_search"]
    assert agent.client.chat.completions.create.call_count == 1
    assert result["context_rollover"]["status"] == "due"
    assert [m["role"] for m in result["messages"]][-3:] == ["assistant", "tool", "assistant"]
    assert result["api_calls"] == 1
    assert agent.iteration_budget.used == 1  # the unsent second call is refunded


@pytest.mark.parametrize("state, reason", [
    (clean_state(children_active=True), "children_active"),
    (clean_state(processes_active=True), "processes_active"),
    (clean_state(approval_pending=True), "approval_pending"),
    (clean_state(clarify_pending=True), "clarify_pending"),
    (clean_state(queued_input=True), "queued_input"),
    (clean_state(stop_requested=True), "stop_requested"),
    (clean_state(children_active=None), "host_state_unknown"),
    ("not-a-host-state", "host_state_unknown"),
    (RuntimeError("guard broke"), "host_state_unavailable"),
])
def test_unresolved_or_unknown_host_state_blocks_without_a_model_call(db, state, reason):
    agent = make_agent(db, count=3)

    def _host_state():
        if isinstance(state, Exception):
            raise state
        return state

    arm(agent, state=_host_state)
    result = run(agent)
    assert agent.client.chat.completions.create.call_count == 0
    assert result["context_rollover"]["status"] == "blocked"
    assert result["context_rollover"]["reason"] == reason
    assert result["completed"] is False
    assert durable_rows(db)[-1]["display_kind"] == ROLLOVER_DISPLAY_KIND


def test_unknown_durable_state_blocks(db):
    agent = make_agent(db, count=3)
    arm(agent)
    db.patch_session_model_config(SID, {FULL_COMPACTION_COUNT_KEY: "3"})
    assert check_rollover(agent).as_dict()["reason"] == "state_unavailable"
    db.patch_session_model_config(SID, {FULL_COMPACTION_COUNT_KEY: 3})
    assert db.try_acquire_compression_lock(SID, "another-surface")
    assert check_rollover(agent).as_dict()["reason"] == "compression_in_progress"


def test_profile_mismatch_blocks(db):
    agent = make_agent(db, count=3)
    arm_context_rollover(agent, CLI, compression_cfg={"rollover_after": 3}, profile="someone-else",
                         host_state=clean_state)
    verdict = check_rollover(agent)
    assert verdict.status == "blocked" and verdict.reason == "profile_mismatch"


def test_stop_and_new_input_outrank_the_rollover(db):
    agent = make_agent(db, count=3)
    arm(agent)
    agent.steer("actually, only touch the docs")
    verdict = check_rollover(agent)
    assert verdict.status == "blocked" and verdict.reason == "queued_input"
    result = run(agent, "original request")
    # A steer left over at the boundary is handed back so it becomes the next user turn.
    assert result["context_rollover"]["reason"] == "queued_input"
    assert result["pending_steer"] == "actually, only touch the docs"
    assert agent.client.chat.completions.create.call_count == 0
    agent._interrupt_requested = True
    assert check_rollover(agent).reason == "stop_requested"


@pytest.mark.parametrize("state, reason", [
    ("claimed", "rollover_claimed"), ("installed", "rollover_installed"), ("started", "rollover_started"),
    ("blocked", "rollover_blocked"), ("completed", "already_rolled_over"),
])
def test_an_existing_transition_blocks_the_source(db, state, reason):
    agent = make_agent(db, count=3)
    arm(agent)
    claim = {
        "schema": 1, "rollover_id": "ro_0123456789abcdef", "source_session_id": SID, "target_session_id": "TGT",
        "profile": PROFILE, "generation": 3, "handoff_digest": "sha256:" + "0" * 64, "state": state,
        "zero_progress_streak": 0, "budget_remaining": 40, "run_budget_seconds": None, "run_budget_started_at": None,
        "source_api_calls": 5,
        "resume_api_calls": None, "claim_token": None, "reason": None, "created_at": 1.0, "updated_at": 1.0,
    }
    db.patch_session_model_config(SID, {ROLLOVER_CLAIM_KEY: claim})
    verdict = check_rollover(agent)
    assert verdict.status == "blocked" and verdict.reason == reason
    assert verdict.target_session_id == "TGT"
