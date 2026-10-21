"""User input outranks an automatic context rollover, even when it arrives mid-transition.

Real ``AIAgent`` instances (fake provider responses, temp SessionDB) and the real gate/controller; only the
moment input arrives is staged: right after the gate's own check, right after the host installs the fresh
agent, or as a new user turn between a failed attempt and its retry.
"""

import pytest

import agent.context_rollover_gate as gate
from agent.context_rollover import RESUME_PROMPT_PREFIX, handoff_path, load_handoff
from agent.context_rollover_gate import HostState
from agent.context_rollover_runtime import perform_rollover, resume_deferred
from hermes_state_compression import FULL_COMPACTION_COUNT_KEY
from hermes_state_rollover import read_source
from tests.agent.test_context_rollover_runtime import (
    PROFILE, SID, Host, _no_sleep, arm, build_agent, clean_state, gated_source, model_calls, run_turn,
)
from tests.agent.test_context_rollover_runtime import db as _db_fixture

db = _db_fixture  # the shared fixture registers here under its own name

RESTRICTION = "Do not publish; only inspect docs"


def _after_check(monkeypatch, action):
    """Run *action(agent, verdict)* once, right after the gate's real check returned a verdict."""
    real = gate.check_rollover
    fired = []

    def wrapped(agent):
        verdict = real(agent)
        if verdict is not None and not fired:
            fired.append(verdict)
            action(agent, verdict)
        return verdict

    monkeypatch.setattr(gate, "check_rollover", wrapped)
    return fired


def _armed_at_threshold(db):
    db.create_session(SID, "cli", model="test/model", profile_name=PROFILE, cwd="/work",
                      model_config={FULL_COMPACTION_COUNT_KEY: 3})
    source = build_agent(db, SID)
    arm(source)
    return source


def _redirect(agent, text):
    """A real active-turn correction, as the input thread sends it while a model request is in flight."""
    agent._model_request_active.set()
    try:
        assert agent.redirect(text)
    finally:
        agent._model_request_active.clear()


# --- 1. the gate never erases input that arrived after its check ------------------------------------------

def test_stop_after_the_due_check_survives_and_nothing_resumes(db, monkeypatch):
    source = _armed_at_threshold(db)
    fired = _after_check(monkeypatch, lambda agent, v: v.status == "due" and agent.interrupt())
    result = run_turn(source, "Build the parser.")
    assert fired and fired[0].status == "due"
    rollover = result["context_rollover"]
    assert rollover["status"] == "blocked" and rollover["reason"] == "stop_requested"
    assert result["interrupted"] is True
    host = Host(db)
    outcome = perform_rollover(source, result, host.adapter())
    assert outcome.status == "blocked"
    assert host.made == [] and host.turns == [] and model_calls(source) == 0
    assert read_source(db, SID).claim is None


def test_steer_after_the_due_check_is_handed_back_not_resumed_over(db, monkeypatch):
    source = _armed_at_threshold(db)
    _after_check(monkeypatch, lambda agent, v: agent.steer(RESTRICTION))
    result = run_turn(source, "Build the parser.")
    assert result["context_rollover"]["status"] == "blocked"
    assert result["context_rollover"]["reason"] == "queued_input"
    assert result["pending_steer"] == RESTRICTION
    host = Host(db)
    assert perform_rollover(source, result, host.adapter()).status == "blocked"
    assert host.made == [] and model_calls(source) == 0


def test_queued_redirect_is_preserved_and_its_interrupt_consumed(db, monkeypatch):
    source = _armed_at_threshold(db)
    real = gate.check_rollover

    def redirect_then_check(agent):
        if not agent._has_pending_redirect() and not getattr(agent, "_redirected", False):
            agent._redirected = True
            _redirect(agent, RESTRICTION)
        return real(agent)

    monkeypatch.setattr(gate, "check_rollover", redirect_then_check)
    result = run_turn(source, "Build the parser.")
    assert result["context_rollover"]["reason"] == "queued_input"
    assert result["pending_steer"] == RESTRICTION and result["interrupted"] is False
    assert source._interrupt_requested is False and not source._has_pending_redirect()


def test_a_stop_after_the_check_is_not_cleared_while_consuming_an_older_redirect(db, monkeypatch):
    source = _armed_at_threshold(db)
    real = gate.check_rollover
    seen = []

    def redirect_check_then_stop(agent):
        if not seen:
            _redirect(agent, "older correction")
        verdict = real(agent)
        if not seen:
            seen.append(verdict)
            agent.interrupt()  # the user's newer stop lands after the gate decided "queued_input"
        return verdict

    monkeypatch.setattr(gate, "check_rollover", redirect_check_then_stop)
    result = run_turn(source, "Build the parser.")
    assert seen[0].reason == "queued_input"
    assert result["context_rollover"]["reason"] == "stop_requested" and result["interrupted"] is True
    host = Host(db)
    assert perform_rollover(source, result, host.adapter()).status == "blocked"
    assert host.made == [] and model_calls(source) == 0


# --- 2. a prepared reservation is reused only for the exact same request and evidence ----------------------

def _fail_once():
    calls = []

    def factory(spec):
        calls.append(spec)
        if len(calls) == 1:
            raise RuntimeError("provider credentials could not be resolved")
        return build_agent(spec.session_db, spec.session_id)

    return factory


def _queued_during_build():
    """A host state that reports queued input from the moment the factory runs (the user typed meanwhile)."""
    flags = {"queued": False}

    def state():
        return HostState(**{**clean_state().__dict__, "queued_input": flags["queued"]})

    def factory(spec):
        flags["queued"] = True
        return build_agent(spec.session_db, spec.session_id)

    return state, factory


def _prepared_by_factory_failure(db):
    source, result = gated_source(db)
    first = perform_rollover(source, result, Host(db, factory=_fail_once()).adapter())
    assert first.status == "blocked" and first.reason == "factory_failed"
    claim = read_source(db, SID).claim
    assert claim["state"] == "prepared"
    return source, result, claim


def _prepared_by_queued_input(db):
    source, result = gated_source(db)
    state, factory = _queued_during_build()
    first = perform_rollover(source, result, Host(db, factory=factory, state=state).adapter())
    assert first.status == "blocked" and first.reason == "queued_input"
    claim = read_source(db, SID).claim
    assert claim["state"] == "prepared"
    return source, result, claim


@pytest.mark.parametrize("prepare", [_prepared_by_factory_failure, _prepared_by_queued_input])
def test_a_new_user_turn_restages_instead_of_resuming_the_old_capsule(db, tmp_path, prepare):
    source, _old_result, old = prepare(db)
    new_result = run_turn(source, RESTRICTION)
    assert new_result["context_rollover"]["status"] == "due"
    host = Host(db)
    outcome = perform_rollover(source, new_result, host.adapter())
    assert outcome.status == "resumed", outcome
    assert outcome.rollover_id != old["rollover_id"]
    prompt = host.turns[0][1]
    assert prompt.startswith(RESUME_PROMPT_PREFIX)
    assert prompt.split("Original request:\n", 1)[1].startswith(RESTRICTION)
    assert read_source(db, SID).claim["rollover_id"] == outcome.rollover_id
    # The superseded reservation's handoff is kept, untouched.
    kept = load_handoff(handoff_path(tmp_path, old["rollover_id"]), profile=PROFILE, source_session_id=SID)
    assert kept["digest"] == old["handoff_digest"]


def test_identical_text_in_a_new_durable_turn_is_a_new_request(db):
    source, _old_result, old = _prepared_by_factory_failure(db)
    text ="Build the parser. Do not publish anything."
    again = run_turn(source, text)
    assert again["context_rollover"]["status"] == "due"
    host = Host(db)
    outcome = perform_rollover(source, again, host.adapter())
    assert outcome.status == "resumed" and outcome.rollover_id != old["rollover_id"]
    assert host.turns[0][1].count(text) >= 3  # request + both verbatim user rows


@pytest.mark.parametrize("prepare", [_prepared_by_factory_failure, _prepared_by_queued_input])
def test_the_same_gated_result_retried_reuses_the_reservation(db, prepare):
    source, result, old = prepare(db)
    host = Host(db)
    outcome = perform_rollover(source, result, host.adapter())
    assert outcome.status == "resumed"
    assert (outcome.rollover_id, outcome.target_session_id) == (old["rollover_id"], old["target_session_id"])


def test_changed_todo_state_restages(db):
    source, result, old = _prepared_by_factory_failure(db)
    source._todo_store.write([{"id": "1", "content": "inspect docs only", "status": "in_progress"}])
    host = Host(db)
    outcome = perform_rollover(source, result, host.adapter())
    assert outcome.status == "resumed" and outcome.rollover_id != old["rollover_id"]
    assert "[in_progress] inspect docs only" in host.turns[0][1]


# --- 7. a stop or new input on the source while the host installs the fresh agent -------------------------

def test_stop_on_the_source_during_install_defers_without_a_model_call(db):
    source, result = gated_source(db)
    host = Host(db, on_install=lambda _agent: source.interrupt())
    outcome = perform_rollover(source, result, host.adapter())
    assert outcome.status == "deferred" and outcome.reason == "stop_requested"
    assert host.turns == [] and model_calls(source, *host.made) == 0
    assert read_source(db, SID).claim["state"] == "installed"
    retry = Host(db)
    again = perform_rollover(source, result, retry.adapter())
    assert again.status == "blocked" and again.reason == "rollover_installed"
    assert retry.made == [] and retry.turns == []


def test_input_on_the_source_during_install_is_kept_for_the_deferred_resume(db):
    source, result = gated_source(db)
    host = Host(db, on_install=lambda _agent: source.steer(RESTRICTION))
    outcome = perform_rollover(source, result, host.adapter())
    assert outcome.status == "deferred" and outcome.reason == "queued_input"
    assert outcome.pending_input == RESTRICTION
    assert host.turns == [] and model_calls(source, *host.made) == 0
    resumed = resume_deferred(outcome, host.adapter(), user_text=outcome.pending_input)
    assert resumed.status == "resumed" and len(host.turns) == 1
    assert host.turns[0][1].rstrip().endswith(RESTRICTION)
    with pytest.raises(ValueError):
        resume_deferred(resumed, host.adapter())
    assert len(host.turns) == 1
