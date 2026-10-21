"""Context rollover closure: residual paths after the transition is already fenced.

Real ``AIAgent`` instances (fake provider responses, temp SessionDB, temp profile home) and the real
gate/controller. Covers (1) legacy rotation of a rollover source or target through the real
``compress_context`` rotation path, (2) a stop the core honours inside the resume turn, (3) input the
controller handled on the source, consumed exactly once, (4) storage errors reading a target's origin, and
(5) cleanup and fail-closed state after unexpected host/candidate failures.
"""

import re

import pytest

import agent.context_rollover_gate as gate
from agent.context_rollover import RESUME_PROMPT_PREFIX
from agent.context_rollover_gate import arm_context_rollover
from agent.context_rollover_runtime import RolloverHostAdapter, perform_rollover
from hermes_state_rollover import read_source
from tests.agent.test_context_rollover_input_races import _redirect
from tests.agent.test_context_rollover_runtime import (
    CLI, PROFILE, SID, Crash, Host, _no_sleep, _response, arm, build_agent, clean_state, gated_source,
    model_calls, run_turn,
)
from tests.agent.test_context_rollover_runtime import db as _db_fixture
from tests.agent.test_context_rollover_transition_failures import (
    SENTINEL, _host, _raise_sqlite, _sanitized, _turn_on,
)

db = _db_fixture  # the shared fixture registers here under its own name

QUEUED = "Correction: only inspect the docs."


def _disarm(agent):
    """Rollover turned off for *agent* (``rollover_after: 0``): the host's own off path."""
    assert arm_context_rollover(agent, CLI, compression_cfg={"rollover_after": 0}, profile=PROFILE,
                                host_state=clean_state) is None
    assert getattr(agent, gate.ARM_ATTR) is None


def _ordinary_turn(agent, text="a normal request"):
    """``(result, model calls this turn made)`` for one ordinary turn that may reach the model once."""
    agent.client.chat.completions.create.side_effect = [_response("normal answer")]
    before = model_calls(agent)
    result = run_turn(agent, text)
    return result, model_calls(agent) - before


def _rotate(agent, db):
    """One committed full compaction on *agent*'s session through the legacy ROTATION path (as
    ``test_compression_rollover_counter`` drives it). Returns ``(parent_id, child_id)``."""
    from agent.conversation_compression import compress_context

    compressor = agent.context_compressor

    def fake_compress(messages, current_tokens=None, focus_topic=None, force=False):
        compressor._last_compression_made_progress = True
        return [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary of prior turns"},
            {"role": "assistant", "content": "recent reply 1"},
            {"role": "user", "content": "follow-up q"},
            {"role": "assistant", "content": "recent reply 2"},
        ]

    compressor.compress = fake_compress
    compressor._last_compress_aborted = False
    compressor._last_summary_error = None
    agent.compression_in_place = False
    agent._session_db_created = True
    parent = agent.session_id
    messages = db.get_messages_as_conversation(parent)
    if len(messages) < 4:  # a fresh target: what its in-flight context would hold at turn entry
        messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"in-flight msg {i}"}
                    for i in range(8)]
    agent._last_flushed_db_idx = len(messages)
    compress_context(agent, messages, approx_tokens=100_000, system_message="sys")
    child = agent.session_id
    assert child != parent and db.get_session(child)["parent_session_id"] == parent
    assert read_source(db, parent).ended  # the parent was closed by the rotation, not by this test
    return parent, child


def _completed(db):
    source, result = gated_source(db)
    host = _host(db, source)
    outcome = perform_rollover(source, result, host.adapter())
    assert outcome.status == "resumed", outcome
    return host, outcome


# --- L1. legacy rotation of a rollover target or source -----------------------------------------------------

@pytest.mark.parametrize("rollover_on", [True, False], ids=["armed", "off"])
def test_completed_target_rotated_by_legacy_compaction_keeps_working_below_threshold(db, rollover_on):
    _, outcome = _completed(db)
    target = outcome.agent
    parent, child = _rotate(target, db)
    assert parent == outcome.target_session_id
    assert read_source(db, child).full_compaction_count == 1  # below the threshold of 3
    if not rollover_on:
        _disarm(target)
    turn, calls = _ordinary_turn(target)
    assert calls == 1, turn.get("context_rollover")
    assert turn.get("context_rollover") is None
    # A reloaded child (new process resuming the rotated session), unarmed and armed.
    for armed in (False, True):
        reloaded = build_agent(db, child)
        if armed:
            arm(reloaded)
        turn, calls = _ordinary_turn(reloaded)
        assert calls == 1 and turn.get("context_rollover") is None, (armed, turn.get("context_rollover"))


class _RotatingHost(Host):
    """A host whose resume turn rotates its session at turn entry (a legacy compaction of the in-flight
    context) before the turn proceeds."""

    def __init__(self, db, **kwargs):
        super().__init__(db, **kwargs)
        self.rotations = []

    def run(self, agent, prompt):
        if prompt.startswith(RESUME_PROMPT_PREFIX) and not self.rotations:
            self.rotations.append(_rotate(agent, self.db))
        return super().run(agent, prompt)


def test_authorized_resume_that_rotates_still_resumes_once(db):
    source, result = gated_source(db)
    host = _RotatingHost(db)
    host.agent = source
    outcome = perform_rollover(source, result, host.adapter())
    assert host.rotations and host.rotations[0][0] == outcome.target_session_id
    assert outcome.status == "resumed", outcome
    assert outcome.resume_api_calls == 1 and model_calls(source, *host.made) == 1
    assert read_source(db, SID).claim["state"] == "completed"
    target = outcome.agent
    assert target.session_id == host.rotations[0][1]
    turn, calls = _ordinary_turn(target)
    assert calls == 1 and turn.get("context_rollover") is None


def _released_prepared(db):
    """A source whose reservation was handed back (``claimed -> prepared``) after a factory failure."""
    source, result = gated_source(db)

    def broken_factory(_spec):
        raise RuntimeError("provider credentials could not be resolved")

    first = perform_rollover(source, result, _host(db, source, factory=broken_factory).adapter())
    assert first.status == "blocked" and first.reason == "factory_failed"
    claim = read_source(db, SID).claim
    assert claim["state"] == "prepared"
    return source, claim


def test_released_source_rotated_by_legacy_compaction_restages_from_the_child(db):
    source, old = _released_prepared(db)
    parent, child = _rotate(source, db)
    assert parent == SID
    stored = read_source(db, child)
    assert stored.full_compaction_count == 4 and stored.profile_name == PROFILE
    again = run_turn(source, "carry on with the parser")
    assert again["context_rollover"]["status"] == "due", again["context_rollover"]
    assert again["context_rollover"]["source_session_id"] == child
    host = _host(db, source)
    outcome = perform_rollover(source, again, host.adapter())
    assert outcome.status == "resumed", outcome
    assert outcome.source_session_id == child and outcome.rollover_id != old["rollover_id"]
    assert outcome.generation == 4 and len(host.turns) == 1
    claim = read_source(db, child).claim
    assert claim["source_session_id"] == child and claim["state"] == "completed"
    assert read_source(db, outcome.target_session_id).origin["source_session_id"] == child


def _crash_at(host, step):
    adapter = host.adapter()

    def crash(*_a, **_k):
        raise Crash()

    return RolloverHostAdapter(**{**adapter.__dict__, step: crash})


def test_source_rotated_while_its_claim_is_held_still_fails_closed(db):
    source, result = gated_source(db)
    with pytest.raises(Crash):
        perform_rollover(source, result, _crash_at(Host(db), "make_agent"))
    assert read_source(db, SID).claim["state"] == "claimed"
    _parent, child = _rotate(source, db)
    turn, calls = _turn_on(source)
    assert calls == 0 and turn["context_rollover"]["status"] == "blocked"
    assert source.session_id == child
    retry = Host(db)
    outcome = perform_rollover(source, turn, retry.adapter())
    assert outcome.status == "blocked" and retry.made == [] and retry.turns == []
    assert read_source(db, SID).claim["state"] == "claimed"  # the copied claim is not a fresh start


def _install_then_fail(host):
    def install(agent, outcome):
        host.install(agent, outcome)
        raise RuntimeError(f"host bookkeeping failed {SENTINEL}")
    return install


def test_blocked_target_rotated_by_legacy_compaction_still_fails_closed(db):
    source, result = gated_source(db)
    host = _host(db, source)
    adapter = RolloverHostAdapter(**{**host.adapter().__dict__, "install": _install_then_fail(host)})
    outcome = perform_rollover(source, result, adapter)
    assert outcome.status == "blocked" and read_source(db, SID).claim["state"] == "blocked"
    target = host.agent
    assert target is host.made[0]
    _parent, child = _rotate(target, db)
    for agent in (target, build_agent(db, child)):  # the rotated agent, and an unarmed reloaded one
        turn, calls = _turn_on(agent)
        assert calls == 0 and turn["context_rollover"]["status"] == "blocked", agent.session_id
    _disarm(target)
    turn, calls = _turn_on(target)
    assert calls == 0 and turn["context_rollover"]["status"] == "blocked"


# --- L2. a stop the core honours inside the resume turn is not a completed resume --------------------------

class _StopFirstHost(Host):
    """The user's stop lands on the fresh agent right as the host starts the resume turn."""

    def run(self, agent, prompt):
        agent.interrupt()
        return super().run(agent, prompt)


def test_stop_honoured_by_the_resume_turn_is_not_recorded_as_resumed(db):
    source, result = gated_source(db)
    host = _StopFirstHost(db)
    host.agent = source
    outcome = perform_rollover(source, result, host.adapter())
    target = host.made[0]
    assert len(host.turns) == 1 and model_calls(source, target) == 0  # the core respected the stop
    assert outcome.status != "resumed", outcome
    assert outcome.status == "blocked"
    claim = read_source(db, SID).claim
    # Ambiguous after ``started``: kept for manual recovery, never completed, never handed back to replay.
    assert claim["state"] == "started"
    assert outcome.rollover_id == claim["rollover_id"] and outcome.target_session_id == target.session_id
    retry = Host(db)
    again = perform_rollover(source, result, retry.adapter())
    assert again.status == "blocked" and again.reason == "rollover_started"
    assert retry.made == [] and retry.turns == []
    target.clear_interrupt()
    turn, calls = _turn_on(target)
    assert calls == 0 and turn["context_rollover"]["reason"] == "rollover_origin_started"


# --- L3. input handled on the source is consumed exactly once -----------------------------------------------

def _publish(kind, agent):
    if kind == "stop":
        agent.interrupt()
    elif kind == "steer":
        assert agent.steer(QUEUED)
    else:
        _redirect(agent, QUEUED)


def _input_host(db, source, point, kind):
    """A host where *kind* input reaches the source during the factory, or during the controller-entry host
    probe (past the entry check, so only the final pre-install guard can see it)."""
    if point == "factory":
        def factory(spec):
            agent = build_agent(spec.session_db, spec.session_id)
            _publish(kind, source)
            return agent
        return _host(db, source, factory=factory)
    calls = {"n": 0}

    def state():
        calls["n"] += 1
        if calls["n"] == 1:
            _publish(kind, source)
        return clean_state()

    return _host(db, source, state=state)


@pytest.mark.parametrize("point", ["factory", "preinstall"])
@pytest.mark.parametrize("kind", ["stop", "steer", "redirect"])
def test_input_during_the_transition_is_consumed_once_and_returned(db, point, kind):
    source, result = gated_source(db)
    host = _input_host(db, source, point, kind)
    outcome = perform_rollover(source, result, host.adapter())
    assert outcome.status == "blocked", outcome
    assert outcome.reason == ("stop_requested" if kind == "stop" else "queued_input")
    assert host.installed == [] and host.turns == [] and host.agent is source
    assert host.made and all(agent in host.retired for agent in host.made)
    assert model_calls(source, *host.made) == 0
    if kind == "stop":
        assert outcome.pending_input is None
    else:
        assert outcome.pending_input == QUEUED  # handed back, not silently lost
    # Handled exactly once: nothing of it is left on the source to poison the next turn.
    assert source._interrupt_requested is False
    assert not source._has_pending_redirect() and not source._pending_steer
    # The source is still at count 3 and armed: its next turn is a clean due request, not a stale stop.
    nxt = run_turn(source, "next request")
    assert nxt["context_rollover"]["status"] == "due", nxt["context_rollover"]
    assert nxt["interrupted"] is False and nxt.get("pending_steer") is None
    assert model_calls(source) == 0
    # With rollover off the next real input reaches the model normally.
    _disarm(source)
    turn, calls = _ordinary_turn(source, "plain question")
    assert calls == 1 and turn.get("interrupted") is not True and turn.get("context_rollover") is None


def test_a_later_stop_is_not_cleared_while_handling_an_earlier_redirect(db):
    source, result = gated_source(db)

    def factory(spec):
        agent = build_agent(spec.session_db, spec.session_id)
        _redirect(source, QUEUED)
        return agent

    host = _host(db, source, factory=factory)
    real_retire = host.retire

    def retire(agent):
        real_retire(agent)
        if agent is not source:
            source.interrupt()  # the user's newer stop lands while the unused candidate is disposed

    host.retire = retire
    outcome = perform_rollover(source, result, host.adapter())
    assert outcome.status == "blocked" and host.installed == [] and model_calls(source, *host.made) == 0
    # The newer stop either is the outcome or is still pending on the source; it is never erased.
    assert outcome.reason == "stop_requested" or source._interrupt_requested is True


# --- L4. a storage error reading a target's origin fails closed --------------------------------------------

def _incomplete_target(db):
    source, result = gated_source(db)
    host = _host(db, source, on_install=lambda _agent: source.steer(QUEUED))
    outcome = perform_rollover(source, result, host.adapter())
    assert outcome.status == "deferred" and read_source(db, SID).claim["state"] == "installed"
    return outcome.target_session_id


def _gate_scoped_lookup_error(monkeypatch, db, session_id):
    """``db.get_session(session_id)`` raises a real storage error, but only while the gate is reading."""
    real_check, real_get, scope = gate.check_rollover, db.get_session, {"on": False}

    def check(agent):
        scope["on"] = True
        try:
            return real_check(agent)
        finally:
            scope["on"] = False

    def get_session(sid, *args, **kwargs):
        if scope["on"] and sid == session_id:
            _raise_sqlite()
        return real_get(sid, *args, **kwargs)

    monkeypatch.setattr(gate, "check_rollover", check)
    monkeypatch.setattr(db, "get_session", get_session)


@pytest.mark.parametrize("armed", [False, True], ids=["unarmed", "armed"])
def test_origin_lookup_error_on_a_reloaded_target_fails_closed(db, monkeypatch, armed):
    target_sid = _incomplete_target(db)
    reloaded = build_agent(db, target_sid)
    if armed:
        arm(reloaded)
    _gate_scoped_lookup_error(monkeypatch, db, target_sid)
    verdict = gate.check_rollover(reloaded)
    assert verdict is not None and verdict.status == "blocked" and verdict.reason == "state_unavailable"
    assert SENTINEL not in str(verdict)
    turn, calls = _turn_on(reloaded)
    assert calls == 0 and turn["context_rollover"]["reason"] == "state_unavailable"
    assert SENTINEL not in str(turn)


def test_no_row_and_unrelated_platform_keep_their_existing_behaviour(db, monkeypatch):
    stateless = build_agent(db, "NO_ROW_YET")  # a CLI agent whose session row does not exist yet
    assert gate.check_rollover(stateless) is None
    db.create_session("LEGACY_GW", "telegram", model="test/model", profile_name=PROFILE)
    legacy = build_agent(db, "LEGACY_GW")
    legacy.platform = "telegram"
    _gate_scoped_lookup_error(monkeypatch, db, "LEGACY_GW")
    assert gate.check_rollover(legacy) is None


# --- L5. unexpected failures: candidates disposed, ambiguous state fails closed ----------------------------

def _row_config_factory(mode):
    def factory(spec):
        agent = build_agent(spec.session_db, spec.session_id)
        if mode == "raises":
            def row_config():
                raise TypeError(f"provider config unreadable {SENTINEL}")
        else:
            def row_config():
                return ["not", "a", "mapping", SENTINEL]
        agent._session_row_model_config = row_config
        return agent
    return factory


def _fixed_reason(outcome):
    _sanitized(outcome)
    assert isinstance(outcome.reason, str) and re.fullmatch(r"[a-z_]+", outcome.reason), outcome.reason


@pytest.mark.parametrize("mode", ["raises", "malformed"])
def test_unexpected_candidate_failure_disposes_it_and_never_resets_the_claim(db, mode):
    source, result = gated_source(db)
    host = _host(db, source, factory=_row_config_factory(mode))
    outcome = perform_rollover(source, result, host.adapter())  # an ordinary exception never escapes
    assert outcome.status == "blocked"
    _fixed_reason(outcome)
    assert "TypeError" not in outcome.reason and "AttributeError" not in outcome.reason
    assert host.installed == [] and host.turns == [] and host.agent is source
    assert len(host.made) == 1 and host.retired == host.made  # the unused candidate is disposed
    assert model_calls(source, *host.made) == 0
    stored = read_source(db, SID)
    assert stored.claim["state"] in ("claimed", "blocked")  # unexplained: not handed back as never-invoked
    assert SENTINEL not in str(stored.claim) and not stored.ended
    retry = _host(db, source)
    again = perform_rollover(source, result, retry.adapter())
    assert again.status == "blocked" and retry.made == [] and retry.turns == []
    turn, calls = _turn_on(source)
    assert calls == 0 and turn["context_rollover"]["status"] == "blocked"


def test_install_failing_before_the_host_adopts_the_candidate_disposes_it(db):
    source, result = gated_source(db)
    host = _host(db, source)

    def install(_agent, _outcome):
        raise RuntimeError(f"host swap failed {SENTINEL}")

    outcome = perform_rollover(source, result, RolloverHostAdapter(**{**host.adapter().__dict__,
                                                                      "install": install}))
    assert outcome.status == "blocked" and outcome.reason == "install_failed"
    _fixed_reason(outcome)
    assert host.agent is source and host.turns == []
    assert len(host.made) == 1 and host.made[0] in host.retired  # built, never adopted: disposed
    assert model_calls(source, *host.made) == 0
    claim = read_source(db, SID).claim
    assert claim["state"] == "blocked" and claim["reason"] == "install_failed"
    assert outcome.rollover_id == claim["rollover_id"] and outcome.target_session_id == claim["target_session_id"]
    turn, calls = _turn_on(source)
    assert calls == 0 and turn["context_rollover"]["status"] == "blocked"
    reloaded = build_agent(db, outcome.target_session_id)
    turn, calls = _turn_on(reloaded)
    assert calls == 0 and turn["context_rollover"]["reason"] == "rollover_origin_blocked"


def test_install_failing_after_the_host_adopts_the_candidate_keeps_it_blocked(db):
    source, result = gated_source(db)
    host = _host(db, source)
    adapter = RolloverHostAdapter(**{**host.adapter().__dict__, "install": _install_then_fail(host)})
    outcome = perform_rollover(source, result, adapter)
    assert outcome.status == "blocked" and outcome.reason == "install_failed"
    _fixed_reason(outcome)
    assert outcome.rollover_id and outcome.source_session_id == SID and outcome.target_session_id
    held = host.agent
    assert held is host.made[0] and host.turns == []
    turn, calls = _turn_on(held)  # never a runnable empty-history model
    assert calls == 0
    verdict = turn["context_rollover"]
    assert verdict["status"] == "blocked" and verdict["reason"] == "rollover_origin_blocked"
    assert verdict["rollover_id"] == outcome.rollover_id and verdict["origin_session_id"] == SID


@pytest.mark.parametrize("step, state", [("row_config", "claimed"), ("install", "installed")])
def test_a_crash_is_not_swallowed_and_leaves_the_claim_where_it_was(db, step, state):
    source, result = gated_source(db)
    if step == "row_config":
        def factory(spec):
            agent = build_agent(spec.session_db, spec.session_id)

            def row_config():
                raise Crash()
            agent._session_row_model_config = row_config
            return agent
        adapter = _host(db, source, factory=factory).adapter()
    else:
        adapter = _crash_at(_host(db, source), "install")
    with pytest.raises(Crash):
        perform_rollover(source, result, adapter)
    stored = read_source(db, SID)
    assert stored.claim["state"] == state and stored.full_compaction_count == 3
