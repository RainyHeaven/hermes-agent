"""Context rollover transition fencing and failure cleanup.

Real ``AIAgent`` instances (fake provider responses, temp SessionDB) and the real gate/controller. Each test
moves the durable source under a gate verdict that already said ``due`` (a compression lease, a committed
compaction, an ended source, a newer user row), or injects an ordinary storage/host failure at one step of the
transition, then runs a real turn on whatever agent the host is left holding: nothing but a valid resume may
reach the model.
"""

import sqlite3
from pathlib import Path

import pytest

import agent.context_rollover_runtime as runtime
from agent.context_rollover import handoff_path
from agent.context_rollover_gate import HostState
from agent.context_rollover_runtime import perform_rollover, resume_deferred
from hermes_state_compression import FULL_COMPACTION_COUNT_KEY
from hermes_state_rollover import read_source
from tests.agent.test_context_rollover_runtime import (
    PROFILE, SID, Host, _no_sleep, _response, arm, build_agent, clean_state, gated_source, model_calls,
    run_turn,
)
from tests.agent.test_context_rollover_runtime import db as _db_fixture

db = _db_fixture  # the shared fixture registers here under its own name

SENTINEL = "SENTINEL-raw-storage-detail-7f3a"
LOCK_HOLDER = "another-compressor"
NEWER_USER = "Newer instruction: only touch the docs."


def _host(db, source, **kwargs):
    host = Host(db, **kwargs)
    host.agent = source  # the host is serving the source agent when the rollover starts
    return host


def _compact(db):
    db.archive_and_compact(SID, [{"role": "user", "content": "[CONTEXT COMPACTION] newer summary"},
                                 {"role": "assistant", "content": "continuing"}], count_full_compaction=True)


_MOVES = {
    "compression_lease": lambda db: db.try_acquire_compression_lock(SID, LOCK_HOLDER),
    "compaction": _compact,
    "source_ended": lambda db: db.end_session(SID, "new_session"),
    "user_append": lambda db: db.append_message(session_id=SID, role="user", content=NEWER_USER),
}


def _assert_no_fork(db, host, source, outcome):
    assert outcome.status == "blocked", outcome
    assert host.installed == [] and host.turns == [] and host.agent is source
    assert all(agent in host.retired for agent in host.made)  # every unused candidate disposed
    assert model_calls(source, *host.made) == 0
    claim = read_source(db, SID).claim
    assert claim is None or claim["state"] == "prepared"
    for agent in host.made:
        assert db.get_session(agent.session_id) is None or not read_source(db, agent.session_id).ended


# --- 3. a stale gate never forks the conversation ---------------------------------------------------------

def test_compression_lease_taken_after_the_gate_blocks_at_controller_entry(db):
    source, result = gated_source(db)
    assert db.try_acquire_compression_lock(SID, LOCK_HOLDER)
    host = _host(db, source)
    outcome = perform_rollover(source, result, host.adapter())
    assert outcome.reason == "compression_in_progress"
    _assert_no_fork(db, host, source, outcome)
    assert host.made == [] and read_source(db, SID).claim is None


@pytest.mark.parametrize("move", sorted(_MOVES))
def test_source_moving_while_the_factory_runs_blocks_without_install(db, move):
    source, result = gated_source(db)

    def factory(spec):
        agent = build_agent(spec.session_db, spec.session_id)
        _MOVES[move](db)
        return agent

    host = _host(db, source, factory=factory)
    outcome = perform_rollover(source, result, host.adapter())
    _assert_no_fork(db, host, source, outcome)
    assert len(host.made) == 1 and host.retired == host.made
    after = read_source(db, SID)
    assert after.ended is (move == "source_ended")
    assert after.full_compaction_count == (4 if move == "compaction" else 3)  # the newer compaction is kept


@pytest.mark.parametrize("move", sorted(_MOVES))
def test_source_moving_just_before_the_swap_is_caught_by_the_install_transaction(db, move):
    """The move lands after every pre-swap recheck (inside the last host-state probe), so only the guarded
    ``installed`` transaction can see it."""
    source, result = gated_source(db)
    calls = {"n": 0}

    def state():
        calls["n"] += 1
        if calls["n"] == 2:  # 1: controller entry, 2: just before the swap
            _MOVES[move](db)
        return clean_state()

    host = _host(db, source, state=state)
    outcome = perform_rollover(source, result, host.adapter())
    assert calls["n"] >= 2
    _assert_no_fork(db, host, source, outcome)
    assert len(host.made) == 1


def test_newer_user_row_after_a_blocked_attempt_is_carried_by_the_next_rollover(db):
    source, result = gated_source(db)

    def factory(spec):
        agent = build_agent(spec.session_db, spec.session_id)
        db.append_message(session_id=SID, role="user", content=NEWER_USER)
        return agent

    first = perform_rollover(source, result, _host(db, source, factory=factory).adapter())
    assert first.status == "blocked"
    again = run_turn(source, "carry on")
    assert again["context_rollover"]["status"] == "due"
    host = _host(db, source)
    outcome = perform_rollover(source, again, host.adapter())
    assert outcome.status == "resumed" and outcome.rollover_id != first.rollover_id
    assert NEWER_USER in host.turns[0][1]


def test_compression_lease_released_lets_the_same_request_roll_over(db):
    source, result = gated_source(db)

    def factory(spec):
        agent = build_agent(spec.session_db, spec.session_id)
        db.try_acquire_compression_lock(SID, LOCK_HOLDER)
        return agent

    assert perform_rollover(source, result, _host(db, source, factory=factory).adapter()).status == "blocked"
    db.release_compression_lock(SID, LOCK_HOLDER)
    host = _host(db, source)
    assert perform_rollover(source, result, host.adapter()).status == "resumed"
    assert len(host.turns) == 1


# --- 5. ordinary storage/todo errors: sanitized blocked outcome, candidate retired ------------------------

def _raise_sqlite(*_a, **_k):
    raise sqlite3.OperationalError(f"disk I/O error {SENTINEL}")


def _sanitized(outcome):
    assert SENTINEL not in repr(outcome) and SENTINEL not in str(outcome.reason)


def test_sqlite_error_creating_the_target_blocks_and_retires_the_candidate(db, monkeypatch):
    source, result = gated_source(db)
    host = _host(db, source)
    real_create, switch = db.create_session, {"on": True}
    monkeypatch.setattr(db, "create_session",
                        lambda *a, **k: _raise_sqlite() if switch["on"] else real_create(*a, **k))
    outcome = perform_rollover(source, result, host.adapter())
    _sanitized(outcome)
    assert outcome.reason == "state_unavailable"
    _assert_no_fork(db, host, source, outcome)
    assert len(host.made) == 1 and db.get_session(outcome.target_session_id) is None
    switch["on"] = False
    retry = _host(db, source)  # nothing was invoked: the same reservation rolls over once storage is back
    resumed = perform_rollover(source, result, retry.adapter())
    assert resumed.status == "resumed" and resumed.rollover_id == outcome.rollover_id
    assert db.get_session(SID) is not None  # candidate cleanup never closed the host's store


def test_sqlite_error_reserving_blocks_before_any_agent(db, monkeypatch):
    source, result = gated_source(db)
    monkeypatch.setattr(runtime, "reserve", _raise_sqlite)
    host = _host(db, source)
    outcome = perform_rollover(source, result, host.adapter())
    _sanitized(outcome)
    assert outcome.status == "blocked" and outcome.reason == "state_unavailable"
    assert host.made == [] and read_source(db, SID).claim is None


def _fail_advance(monkeypatch, to_state):
    """Fail every *to_state* transition while the returned switch is on."""
    real, switch = runtime.advance, {"on": True}

    def advance(*args, **kwargs):
        if switch["on"] and args[4] == to_state:
            _raise_sqlite()
        return real(*args, **kwargs)

    monkeypatch.setattr(runtime, "advance", advance)
    return switch


def test_sqlite_error_claiming_blocks_and_keeps_the_reservation(db, monkeypatch):
    source, result = gated_source(db)
    _fail_advance(monkeypatch, "claimed")
    host = _host(db, source)
    outcome = perform_rollover(source, result, host.adapter())
    _sanitized(outcome)
    assert outcome.status == "blocked" and outcome.reason == "state_unavailable"
    assert host.made == [] and read_source(db, SID).claim["state"] == "prepared"


def test_todo_store_failure_blocks_before_any_reservation(db):
    source, result = gated_source(db)

    def broken_read():
        raise OSError(f"todo file unreadable {SENTINEL}")

    source._todo_store.read = broken_read
    host = _host(db, source)
    outcome = perform_rollover(source, result, host.adapter())
    _sanitized(outcome)
    assert outcome.status == "blocked" and outcome.reason == "todo_unavailable"
    assert host.made == [] and read_source(db, SID).claim is None


# --- 5/6. failures around the swap: no empty-history model turn on whatever the host holds ----------------

def _turn_on(agent, text="what next?"):
    agent.client.chat.completions.create.side_effect = [_response("must never be requested")]
    before = model_calls(agent)
    result = run_turn(agent, text)
    return result, model_calls(agent) - before


def test_installed_commit_failure_keeps_the_host_on_the_source(db, monkeypatch):
    source, result = gated_source(db)
    switch = _fail_advance(monkeypatch, "installed")
    host = _host(db, source)
    outcome = perform_rollover(source, result, host.adapter())
    _sanitized(outcome)
    assert outcome.reason == "state_unavailable"
    _assert_no_fork(db, host, source, outcome)
    assert not read_source(db, SID).ended
    switch["on"] = False
    turn, calls = _turn_on(host.agent)
    assert calls == 0 and turn["context_rollover"]["status"] == "due"  # the source rolls over again, no call
    # The bound-but-never-installed target cannot be used on its own either.
    orphan = build_agent(db, outcome.target_session_id)
    arm(orphan)
    turn, calls = _turn_on(orphan)
    assert calls == 0 and turn["context_rollover"]["status"] == "blocked"


def _install_then_fail(host):
    def install(agent, outcome):
        host.install(agent, outcome)  # the host already swapped, then its own bookkeeping failed
        raise sqlite3.OperationalError(f"host bookkeeping failed {SENTINEL}")
    return install


def _retire_fails(_agent):
    raise sqlite3.OperationalError(f"retire failed {SENTINEL}")


@pytest.mark.parametrize("step, reason", [("install", "install_failed"), ("retire", "retire_failed")])
def test_failure_after_the_swap_blocks_every_agent_the_host_may_hold(db, step, reason):
    source, result = gated_source(db)
    host = _host(db, source)
    adapter = host.adapter()
    if step == "install":
        adapter = runtime.RolloverHostAdapter(**{**adapter.__dict__, "install": _install_then_fail(host)})
    else:
        adapter = runtime.RolloverHostAdapter(**{**adapter.__dict__, "retire": _retire_fails})
    outcome = perform_rollover(source, result, adapter)
    _sanitized(outcome)
    assert outcome.status == "blocked" and outcome.reason == reason
    assert outcome.rollover_id and outcome.source_session_id == SID and outcome.target_session_id
    assert host.turns == [] and model_calls(source, *host.made) == 0
    claim = read_source(db, SID).claim
    assert claim["state"] == "blocked" and claim["reason"] == reason
    assert handoff_path(Path(db.db_path).parent, claim["rollover_id"]).exists()  # the handoff is preserved
    target = host.made[0]
    assert host.agent is target
    for agent in (target, source):
        turn, calls = _turn_on(agent)
        assert calls == 0, agent.session_id
        assert turn["context_rollover"]["status"] == "blocked"
    verdict = _turn_on(target)[0]["context_rollover"]
    assert verdict["reason"] == "rollover_origin_blocked"
    assert verdict["rollover_id"] == outcome.rollover_id and verdict["origin_session_id"] == SID
    # A reloaded target (a new process resuming that session) fails closed from durable state alone.
    reloaded = build_agent(db, outcome.target_session_id)
    arm(reloaded)
    turn, calls = _turn_on(reloaded)
    assert calls == 0 and turn["context_rollover"]["reason"] == "rollover_origin_blocked"


def test_deferred_target_refuses_an_ordinary_turn_until_the_authorized_resume(db):
    source, result = gated_source(db)
    host = _host(db, source, on_install=lambda _agent: source.steer(NEWER_USER))
    outcome = perform_rollover(source, result, host.adapter())
    assert outcome.status == "deferred"
    target = outcome.agent
    turn, calls = _turn_on(target, "an ordinary turn")
    assert calls == 0 and turn["context_rollover"]["reason"] == "rollover_origin_installed"
    target.client.chat.completions.create.side_effect = [_response("resumed with the newer instruction")]
    resumed = resume_deferred(outcome, host.adapter(), user_text=outcome.pending_input)
    assert resumed.status == "resumed" and model_calls(target) == 1
    target.client.chat.completions.create.side_effect = [_response("a normal later turn")]
    later = run_turn(target, "and now a normal request")
    assert model_calls(target) == 2 and later.get("context_rollover") is None


def test_sqlite_error_starting_the_resume_defers_and_can_be_retried(db, monkeypatch):
    source, result = gated_source(db)
    switch = _fail_advance(monkeypatch, "started")
    host = _host(db, source)
    outcome = perform_rollover(source, result, host.adapter())
    _sanitized(outcome)
    assert outcome.status == "deferred" and outcome.reason == "state_unavailable"
    assert host.turns == [] and read_source(db, SID).claim["state"] == "installed"
    turn, calls = _turn_on(outcome.agent)
    assert calls == 0 and turn["context_rollover"]["status"] == "blocked"
    switch["on"] = False
    outcome.agent.client.chat.completions.create.side_effect = [_response("resumed")]
    resumed = resume_deferred(outcome, host.adapter())
    assert resumed.status == "resumed" and model_calls(outcome.agent) == 1


def test_completion_record_failure_is_reported_and_blocks_later_turns(db, monkeypatch):
    source, result = gated_source(db)
    switch = _fail_advance(monkeypatch, "completed")
    host = _host(db, source)
    outcome = perform_rollover(source, result, host.adapter())
    _sanitized(outcome)
    assert outcome.status == "resumed" and outcome.reason == "completion_not_recorded"
    assert model_calls(outcome.agent) == 1 and read_source(db, SID).claim["state"] == "started"
    switch["on"] = False
    turn, calls = _turn_on(outcome.agent)
    assert calls == 0 and turn["context_rollover"]["reason"] == "rollover_origin_started"


def test_completed_target_with_a_higher_threshold_still_works_normally(db):
    source, result = gated_source(db)
    host = _host(db, source)
    outcome = perform_rollover(source, result, host.adapter())
    assert outcome.status == "resumed"
    target = outcome.agent
    arm(target, after=10)
    target.client.chat.completions.create.side_effect = [_response("normal")]
    later = run_turn(target, "next request")
    assert model_calls(target) == 2 and later.get("context_rollover") is None


def test_state_db_failure_on_a_target_gate_fails_closed(db, monkeypatch):
    source, result = gated_source(db)
    outcome = perform_rollover(source, result, _host(db, source).adapter())
    assert outcome.status == "resumed"
    target = outcome.agent
    real_get = db.get_session

    def get_session(sid):
        if sid == SID:
            _raise_sqlite()
        return real_get(sid)

    monkeypatch.setattr(db, "get_session", get_session)
    turn, calls = _turn_on(target)
    assert calls == 0 and turn["context_rollover"]["reason"] == "state_unavailable"
    assert SENTINEL not in str(turn)
