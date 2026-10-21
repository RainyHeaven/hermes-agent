"""Context rollover: a deferred resume still continues the source's logical turn budget.

Independent regression for the logical-turn budget contract. A real ``AIAgent`` source (fake provider, temp
SessionDB) rolls over after 3 of 10 iterations with a 600s wall-clock cap whose turn began 120s earlier. The
host turns busy right after installing the fresh agent, so ``perform_rollover`` returns ``deferred`` before
the target ran anything. The held-back resume (``resume_deferred``, no user text) is the same logical turn:
it must run under the carried 7 remaining iterations, the original start and the 600s cap, whether or not an
ordinary direct turn was refused on the fenced target in between. A genuine new user turn would reset the
budget on purpose, so no user text is involved here.
"""

from agent.context_rollover_gate import HostState
from agent.context_rollover_runtime import perform_rollover, resume_deferred
from hermes_state_rollover import read_source
from tests.agent.test_context_rollover_runtime import Host, _no_sleep, clean_state, model_calls, run_turn
from tests.agent.test_context_rollover_runtime import db as _db_fixture
from tests.agent.test_context_rollover_runtime_settings import _agent, _budgeted_source, _dir, _recorder

db = _db_fixture  # the shared fixture registers here under its own name


def _busy_state():
    return HostState(children_active=False, processes_active=True, approval_pending=False,
                     clarify_pending=False, queued_input=False, stop_requested=False)


def _deferred_by_busy_host(db, tmp_path):
    """The real budgeted source, and a host that turns busy right after installing its fresh agent."""
    work = _dir(tmp_path, "work")
    source, result, started = _budgeted_source(db, work)
    log, busy = [], {"on": False}

    def state():
        return _busy_state() if busy["on"] else clean_state()

    def factory(spec):
        agent = _agent(spec.session_db, spec.session_id, cwd=work, max_iterations=10,
                       run_budget_seconds=spec.run_budget_seconds)
        agent.client.chat.completions.create.side_effect = _recorder(agent, log, "resumed")
        return agent

    host = Host(db, factory=factory, state=state, on_install=lambda _installed: busy.update(on=True))
    outcome = perform_rollover(source, result, host.adapter())
    assert outcome.status == "deferred", outcome
    new = outcome.agent
    assert new is not source and new.session_id == outcome.target_session_id != source.session_id
    assert new.run_budget_seconds == 600.0  # the installed agent matches the spec's cap
    assert host.turns == [] and log == [] and model_calls(new) == 0  # deferred before the target ran
    assert read_source(db, source.session_id).claim["state"] == "installed"
    busy["on"] = False  # the host is clear again; the outcome is still deferred
    return host, outcome, new, log, started


def _assert_resume_carried_the_budget(db, host, outcome, new, log, started):
    resumed = resume_deferred(outcome, host.adapter(), user_text=None)
    assert resumed.status == "resumed", resumed
    assert model_calls(new) == 1  # the authorized resume really reached the provider, exactly once
    # The resume continues the source's logical turn: its remaining iterations, its start and its cap.
    assert log == [{"started_at": started, "max_total": 7, "seconds": 600.0}]
    assert read_source(db, outcome.source_session_id).claim["state"] == "completed"


def test_지연된_재개는_원래_턴_예산을_이어받는다(db, tmp_path):
    """Positive control: deferred, then resumed directly with no attempt in between."""
    host, outcome, new, log, started = _deferred_by_busy_host(db, tmp_path)
    _assert_resume_carried_the_budget(db, host, outcome, new, log, started)


def test_거부된_일반_턴_뒤의_지연_재개도_원래_턴_예산을_이어받는다(db, tmp_path):
    """An ordinary direct turn on the still-deferred target is origin-blocked without a model call; it is not
    new authorized work, so it must not consume the one-shot carry the held-back resume runs under."""
    host, outcome, new, log, started = _deferred_by_busy_host(db, tmp_path)
    turn = run_turn(new, "an ordinary request on the fenced target")
    assert turn["completed"] is False
    assert turn["context_rollover"]["reason"] == "rollover_origin_installed"
    assert model_calls(new) == 0 and log == []
    assert read_source(db, outcome.source_session_id).claim["state"] == "installed"
    _assert_resume_carried_the_budget(db, host, outcome, new, log, started)
