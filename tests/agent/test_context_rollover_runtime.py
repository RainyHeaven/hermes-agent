"""Context rollover controller (``agent/context_rollover_runtime.py``) driven by REAL gated turn results.

A test host adapter builds genuine ``AIAgent`` instances (fake provider responses only, temp SessionDB, temp
profile home) and records every install/retire/turn, so each test can count model requests and resumes.
"""

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.context_rollover import RESUME_PROMPT_PREFIX, handoff_path, load_handoff
from agent.context_rollover_gate import HostState, arm_context_rollover
from agent.context_rollover_runtime import (
    FreshAgentSpec, RolloverHostAdapter, perform_rollover, resume_deferred,
)
from hermes_state import SessionDB
from hermes_state_compression import FULL_COMPACTION_COUNT_KEY
from hermes_state_rollover import SOURCE_END_REASON, read_source
from agent.context_rollover import RolloverHost

PROFILE = "rollover-test"
SID = "RT_SOURCE"
API_KEY = "sk-test-SECRET-never-persisted-123456"
CLI = RolloverHost(surface="cli", resumes_in_fresh_agent=True)


class Crash(BaseException):
    """A process dying mid-transition: not an ordinary exception the controller may handle."""


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    import time as _time

    from agent import retry_utils
    monkeypatch.setattr(_time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(retry_utils, "jittered_backoff", lambda *a, **k: 0.0)


def _response(content="done", tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls, reasoning_content=None, reasoning=None)
    finish = "tool_calls" if tool_calls else "stop"
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish)],
                           model="test/model", usage=None)


def _tool_call(call_id):
    return SimpleNamespace(id=call_id, type="function",
                           function=SimpleNamespace(name="web_search", arguments='{"path": "notes.md"}'))


def clean_state():
    return HostState(children_active=False, processes_active=False, approval_pending=False,
                     clarify_pending=False, queued_input=False, stop_requested=False)


@pytest.fixture
def db(tmp_path):
    store = SessionDB(tmp_path / "state.db")
    yield store
    store.close()


def workdir(db):
    """The real directory every agent on *db* works in (the store's own parent, canonical): seeded on the
    session rows AND passed to the constructors, so a row's cwd is what its agent actually uses."""
    return str(Path(db.db_path).parent.resolve())


def build_agent(db, sid, *, model="test/model", max_iterations=50, responses=(), cwd=None):
    from run_agent import AIAgent

    tools = [{"type": "function", "function": {"name": "web_search", "description": "t",
                                                "parameters": {"type": "object", "properties": {}}}}]
    with (
        patch("model_tools.get_tool_definitions", return_value=tools),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(api_key=API_KEY, base_url="https://openrouter.ai/api/v1", model=model, quiet_mode=True,
                        skip_context_files=True, skip_memory=True, session_db=db, session_id=sid,
                        platform="cli", max_iterations=max_iterations, cwd=cwd or workdir(db))
    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = list(responses) or [_response("resumed and finished")]
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = True
    agent.save_trajectories = False
    agent.valid_tool_names = {"web_search"}
    agent.tool_delay = 0
    return agent


def run_turn(agent, text, **kwargs):
    with (patch.object(agent, "_save_trajectory"), patch.object(agent, "_cleanup_task_resources"),
          patch("model_tools.handle_function_call", return_value="tool output")):
        return agent.run_conversation(text, **kwargs)


def arm(agent, after=3, state=clean_state):
    assert arm_context_rollover(agent, CLI, compression_cfg={"rollover_after": after}, profile=PROFILE,
                                host_state=state) is None


class Host:
    """A test CLI/TUI host: owns the SessionDB, swaps agents, counts every model request it caused."""

    def __init__(self, db, *, factory=None, responses=None, state=clean_state, threshold=3, on_install=None):
        self.db, self.agent, self.state = db, None, state
        self.made, self.installed, self.retired, self.turns = [], [], [], []
        self.factory, self.responses, self.threshold, self.on_install = factory, responses, threshold, on_install

    def make_agent(self, spec: FreshAgentSpec):
        if self.factory is not None:
            agent = self.factory(spec)
        else:
            agent = build_agent(spec.session_db, spec.session_id, model=spec.route["model"],
                                responses=self.responses or ())
        self.made.append(agent)
        return agent

    def install(self, agent, outcome):
        self.installed.append((agent, outcome.target_session_id))
        self.agent = agent
        arm(agent, after=self.threshold, state=self.state)
        if self.on_install:
            self.on_install(agent)

    def retire(self, agent):
        self.retired.append(agent)

    def run(self, agent, prompt):
        self.turns.append((agent, prompt))
        return run_turn(agent, prompt)

    def adapter(self):
        return RolloverHostAdapter(host_state=lambda: self.state(), make_agent=self.make_agent,
                                   install=self.install, retire=self.retire, run_turn=self.run)


def model_calls(*agents):
    return sum(a.client.chat.completions.create.call_count for a in agents)


def gated_source(db, *, count=3, history=None, text="Build the parser. Do not publish anything."):
    """A source session at the threshold and the real gated result of its next turn."""
    db.create_session(SID, "cli", model="test/model", profile_name=PROFILE, cwd=workdir(db),
                      model_config={FULL_COMPACTION_COUNT_KEY: count})
    source = build_agent(db, SID)
    arm(source)
    result = run_turn(source, text, conversation_history=history or [])
    assert result["context_rollover"]["status"] == "due"
    assert model_calls(source) == 0
    return source, result


def test_rollover_resumes_once_in_a_fresh_root_session_and_agent(db, tmp_path):
    source, result = gated_source(db)
    host = Host(db)
    outcome = perform_rollover(source, result, host.adapter())
    assert outcome.status == "resumed", outcome
    new = outcome.agent
    assert new is not source and host.agent is new and host.retired == [source]
    assert outcome.target_session_id == new.session_id != SID
    assert outcome.source_session_id == SID and outcome.generation == 3
    # Exactly one model request, on the NEW agent, whose first user turn is the rendered handoff.
    assert model_calls(source) == 0 and model_calls(new) == 1 and len(host.turns) == 1
    first_user = new.client.chat.completions.create.call_args.kwargs["messages"]
    resume = [m for m in first_user if m["role"] == "user"][-1]["content"]
    assert resume.startswith(RESUME_PROMPT_PREFIX) and "Do not publish anything." in resume
    assert outcome.resume_api_calls == 1
    # Durable state: target is a fresh root with count 0 and origin; source keeps count, transcript, claim.
    target = read_source(db, new.session_id)
    assert target.full_compaction_count == 0 and target.claim is None
    assert target.origin["rollover_id"] == outcome.rollover_id and target.origin["source_session_id"] == SID
    assert db.get_session(new.session_id)["parent_session_id"] is None
    assert db.get_session(new.session_id)["cwd"] == new.session_cwd == workdir(db)  # the cwd both agents use
    src = read_source(db, SID)
    assert src.full_compaction_count == 3 and src.claim["state"] == "completed" and src.ended
    assert db.get_session(SID)["end_reason"] == SOURCE_END_REASON
    assert [m["content"] for m in db.get_messages_as_conversation(SID) if m["role"] == "user"] == [
        "Build the parser. Do not publish anything."]
    # The handoff lives under the owning profile home, verifies, and carries no credential.
    path = handoff_path(tmp_path, outcome.rollover_id)
    assert outcome.handoff_path == path
    stored = load_handoff(path, profile=PROFILE, source_session_id=SID)
    assert stored["digest"] == src.claim["handoff_digest"]
    assert API_KEY not in path.read_text()


def test_archived_user_restrictions_reach_the_handoff(db):
    db.create_session(SID, "cli", model="test/model", profile_name=PROFILE, cwd=workdir(db))
    db.append_message(session_id=SID, role="user", content="Original task: refactor the loader.")
    db.append_message(session_id=SID, role="assistant", content="ok")
    db.append_message(session_id=SID, role="user", content="Never push to the remote.")
    db.append_message(session_id=SID, role="assistant", content="understood")
    # A committed full compaction soft-archived those rows behind a lossy summary.
    for _ in range(3):
        db.archive_and_compact(SID, [{"role": "user", "content": "[CONTEXT COMPACTION] lossy summary"},
                                     {"role": "assistant", "content": "continuing"}], count_full_compaction=True)
    source = build_agent(db, SID)
    arm(source)
    result = run_turn(source, "continue", conversation_history=db.get_messages_as_conversation(SID))
    host = Host(db)
    outcome = perform_rollover(source, result, host.adapter())
    assert outcome.status == "resumed", outcome
    prompt = host.turns[0][1]
    for text in ("Original task: refactor the loader.", "Never push to the remote.", "continue"):
        assert text in prompt


def test_duplicate_controller_call_does_not_resume_again(db):
    source, result = gated_source(db)
    host = Host(db)
    first = perform_rollover(source, result, host.adapter())
    again = perform_rollover(source, result, host.adapter())
    assert first.status == "resumed" and again.status == "already_completed"
    assert again.target_session_id == first.target_session_id and again.agent is None
    assert len(host.made) == 1 and len(host.turns) == 1


def test_concurrent_controllers_resume_exactly_once(db):
    source, result = gated_source(db)
    barrier = threading.Barrier(2)
    hosts = [Host(db), Host(db)]
    outcomes = [None, None]

    def go(i):
        barrier.wait(timeout=5)
        outcomes[i] = perform_rollover(source, result, hosts[i].adapter())

    threads = [threading.Thread(target=go, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    statuses = sorted(o.status for o in outcomes)
    assert statuses.count("resumed") == 1, outcomes
    assert sum(len(h.turns) for h in hosts) == 1
    assert sum(model_calls(*h.made) for h in hosts) == 1


def test_crash_while_prepared_recovers_with_the_same_ids(db, monkeypatch):
    import agent.context_rollover_runtime as runtime

    source, result = gated_source(db)
    real_advance = runtime.advance

    def crash_on_claim(*args, **kwargs):
        if args[3] == "prepared":
            raise Crash()
        return real_advance(*args, **kwargs)

    monkeypatch.setattr(runtime, "advance", crash_on_claim)
    with pytest.raises(Crash):
        perform_rollover(source, result, Host(db).adapter())
    prepared = read_source(db, SID).claim
    assert prepared["state"] == "prepared"
    monkeypatch.setattr(runtime, "advance", real_advance)
    host = Host(db)  # a restarted process: new host, same durable store
    outcome = perform_rollover(source, result, host.adapter())
    assert outcome.status == "resumed"
    assert (outcome.rollover_id, outcome.target_session_id) == (
        prepared["rollover_id"], prepared["target_session_id"])
    assert len(host.turns) == 1


@pytest.mark.parametrize("crash_point, state", [("make_agent", "claimed"), ("run_turn", "started")])
def test_crash_after_the_claim_blocks_for_manual_recovery(db, crash_point, state):
    source, result = gated_source(db)
    host = Host(db)
    adapter = host.adapter()

    def crash(*_a, **_k):
        raise Crash()

    crashing = RolloverHostAdapter(**{**adapter.__dict__, crash_point: crash})
    with pytest.raises(Crash):
        perform_rollover(source, result, crashing)
    assert read_source(db, SID).claim["state"] == state
    retry = Host(db)
    outcome = perform_rollover(source, result, retry.adapter())
    assert outcome.status == "blocked" and outcome.reason == f"rollover_{state}"
    assert retry.made == [] and retry.turns == []
    # The source itself stays blocked too: its next turn makes no model call.
    fresh_turn = run_turn(source, "hello again")
    assert fresh_turn["context_rollover"]["reason"] == f"rollover_{state}" and model_calls(source) == 0


def test_factory_failure_issues_no_model_call_and_keeps_the_source(db, tmp_path):
    source, result = gated_source(db)

    def broken_factory(_spec):
        raise RuntimeError("provider credentials could not be resolved")

    host = Host(db, factory=broken_factory)
    outcome = perform_rollover(source, result, host.adapter())
    assert outcome.status == "blocked" and outcome.reason == "factory_failed"
    assert host.installed == [] and host.retired == [] and host.turns == [] and model_calls(source) == 0
    claim = read_source(db, SID).claim
    assert claim["state"] == "prepared" and claim["reason"].startswith("factory_failed")
    assert not read_source(db, SID).ended
    assert handoff_path(tmp_path, claim["rollover_id"]).exists()
    working = Host(db)  # the next attempt reuses the preserved reservation
    retried = perform_rollover(source, result, working.adapter())
    assert retried.status == "resumed" and retried.rollover_id == claim["rollover_id"]


@pytest.mark.parametrize("mutation, reason", [
    ("model", "route_mismatch"), ("session_id", "fresh_agent_invalid"), ("source_agent", "fresh_agent_invalid"),
    ("other_db", "target_db_mismatch"),
])
def test_wrong_fresh_agent_is_refused_before_any_model_call(db, tmp_path, mutation, reason):
    source, result = gated_source(db)
    other_db = SessionDB(tmp_path / "other" / "state.db")

    def factory(spec):
        if mutation == "source_agent":
            return source
        if mutation == "model":
            return build_agent(spec.session_db, spec.session_id, model="other/default-model")
        if mutation == "session_id":
            return build_agent(spec.session_db, "SOMETHING_ELSE")
        return build_agent(other_db, spec.session_id)

    host = Host(db, factory=factory)
    try:
        outcome = perform_rollover(source, result, host.adapter())
    finally:
        other_db.close()
    assert outcome.status == "blocked" and outcome.reason == reason
    assert host.installed == [] and host.turns == [] and model_calls(source, *[a for a in host.made if a is not source]) == 0
    assert read_source(db, SID).claim["state"] == "prepared"


def test_wrong_profile_or_corrupt_handoff_fails_closed(db, tmp_path, monkeypatch):
    import agent.context_rollover_runtime as runtime

    source, result = gated_source(db)
    forged = {**result, "context_rollover": {**result["context_rollover"], "profile": "other-profile"}}
    assert perform_rollover(source, forged, Host(db).adapter()).reason == "request_mismatch"
    db._conn.execute("UPDATE sessions SET profile_name = 'other-profile' WHERE id = ?", (SID,))
    db._conn.commit()
    assert perform_rollover(source, result, Host(db).adapter()).reason == "profile_mismatch"
    db._conn.execute("UPDATE sessions SET profile_name = ? WHERE id = ?", (PROFILE, SID))
    db._conn.commit()
    # Reserve, then corrupt the preserved handoff before recovery.
    real_advance = runtime.advance
    monkeypatch.setattr(runtime, "advance", lambda *a, **k: (_ for _ in ()).throw(Crash()))
    with pytest.raises(Crash):
        perform_rollover(source, result, Host(db).adapter())
    monkeypatch.setattr(runtime, "advance", real_advance)
    path = handoff_path(tmp_path, read_source(db, SID).claim["rollover_id"])
    data = json.loads(path.read_text())
    data["body"]["user_request"] = "Publish everything."
    path.chmod(0o600)
    path.write_text(json.dumps(data))
    host = Host(db)
    outcome = perform_rollover(source, result, host.adapter())
    assert outcome.status == "blocked" and outcome.reason == "handoff_invalid"
    assert host.made == [] and host.turns == []


def test_remaining_budget_carries_into_the_resume_turn_only(db):
    db.create_session(SID, "cli", model="test/model", profile_name=PROFILE, cwd=workdir(db),
                      model_config={FULL_COMPACTION_COUNT_KEY: 2})
    source = build_agent(db, SID, max_iterations=10, responses=[
        _response("", tool_calls=[_tool_call("c1")]), _response("", tool_calls=[_tool_call("c2")]),
        _response("", tool_calls=[_tool_call("c3")]), _response("never requested")])
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
    remaining = source.iteration_budget.remaining
    assert remaining == 7
    host = Host(db, factory=lambda spec: build_agent(spec.session_db, spec.session_id, max_iterations=10))
    outcome = perform_rollover(source, result, host.adapter())
    assert outcome.status == "resumed" and outcome.budget_remaining == 7
    new = outcome.agent
    assert new.iteration_budget.max_total == 7  # the resume turn continues the logical turn's budget
    new.client.chat.completions.create.side_effect = [_response("a later real user turn")]
    run_turn(new, "next real request")
    assert new.iteration_budget.max_total == 10  # later user turns get a normal budget


def test_two_consecutive_zero_progress_rollovers_stop(db):
    source, result = gated_source(db)

    def compacts_before_any_call(agent):
        # The fresh session's own context build commits compactions before its first model call.
        db.patch_session_model_config(agent.session_id, {FULL_COMPACTION_COUNT_KEY: 3})

    host = Host(db, on_install=compacts_before_any_call)
    first = perform_rollover(source, result, host.adapter())
    assert first.status == "resumed" and first.resume_result["context_rollover"]["status"] == "due"
    second = perform_rollover(first.agent, first.resume_result, host.adapter())
    assert second.status == "resumed" and second.resume_result["context_rollover"]["status"] == "due"
    third = perform_rollover(second.agent, second.resume_result, host.adapter())
    assert third.status == "blocked" and third.reason == "no_progress"
    assert len(host.made) == 2 and model_calls(source, *host.made) == 0


def test_input_before_the_swap_keeps_the_source(db):
    source, result = gated_source(db)
    flags = {"queued": False}

    def state():
        return HostState(**{**clean_state().__dict__, "queued_input": flags["queued"]})

    def factory(spec):
        flags["queued"] = True  # the user typed while the fresh agent was being built
        return build_agent(spec.session_db, spec.session_id)

    host = Host(db, factory=factory, state=state)
    outcome = perform_rollover(source, result, host.adapter())
    assert outcome.status == "blocked" and outcome.reason == "queued_input"
    assert host.installed == [] and host.retired == [host.made[0]] and model_calls(*host.made) == 0
    assert read_source(db, SID).claim["state"] == "prepared" and not read_source(db, SID).ended


def test_input_after_the_swap_defers_then_resumes_with_the_user_text(db):
    source, result = gated_source(db)
    flags = {"queued": False}

    def state():
        return HostState(**{**clean_state().__dict__, "queued_input": flags["queued"]})

    host = Host(db, state=state, on_install=lambda _agent: flags.update(queued=True))
    outcome = perform_rollover(source, result, host.adapter())
    assert outcome.status == "deferred" and outcome.reason == "queued_input"
    assert host.agent is outcome.agent and host.turns == [] and model_calls(outcome.agent) == 0
    assert read_source(db, SID).claim["state"] == "installed"
    flags["queued"] = False
    resumed = resume_deferred(outcome, host.adapter(), user_text="Only update the docs, please.")
    assert resumed.status == "resumed" and len(host.turns) == 1
    prompt = host.turns[0][1]
    assert prompt.startswith(RESUME_PROMPT_PREFIX) and prompt.rstrip().endswith("Only update the docs, please.")
    assert read_source(db, SID).claim["state"] == "completed"
    assert outcome.agent.iteration_budget.max_total == 50  # a real user turn: normal budget


def test_a_result_that_is_not_a_gated_request_is_refused(db):
    source, result = gated_source(db)
    host = Host(db)
    plain = {k: v for k, v in result.items() if k != "context_rollover"}
    assert perform_rollover(source, plain, host.adapter()).reason == "not_a_rollover_request"
    stale = {**result, "context_rollover": {**result["context_rollover"], "full_compaction_count": 2}}
    assert perform_rollover(source, stale, host.adapter()).reason == "stale_request"
    assert host.made == [] and read_source(db, SID).claim is None
