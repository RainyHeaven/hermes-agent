"""Durable context-rollover transition records (``hermes_state_rollover``).

One compare-and-set record per source session: a reservation is tied to the source's profile and its exact
committed full-compaction count, every state change is a CAS inside one write transaction, the target is a
fresh root session carrying only the origin, and no other model_config writer can erase or forge a claim.
"""

import json
import threading
import time

import pytest

from hermes_state import SessionDB
from hermes_state_compression import FULL_COMPACTION_COUNT_KEY
from hermes_state_rollover import (
    ROLLOVER_CLAIM_KEY, ROLLOVER_ORIGIN_KEY, SOURCE_END_REASON, RolloverConflict, RolloverRecordError,
    advance, bind_target, messages_mark, origin_for, read_source, reserve,
)


@pytest.fixture
def db(tmp_path):
    store = SessionDB(tmp_path / "state.db")
    yield store
    store.close()


def _source(db, sid="SRC", count=3, profile="work"):
    db.create_session(sid, "cli", model="m", model_config={FULL_COMPACTION_COUNT_KEY: count, "keep": "me"},
                      profile_name=profile, cwd="/work")
    return sid


def _claim(**overrides):
    now = time.time()
    record = {
        "schema": 1, "rollover_id": "ro_0123456789abcdef", "source_session_id": "SRC",
        "target_session_id": "TGT", "profile": "work", "generation": 3, "handoff_digest": "sha256:" + "0" * 64,
        "state": "prepared", "zero_progress_streak": 0, "budget_remaining": 40, "run_budget_seconds": None,
        "run_budget_started_at": None, "source_api_calls": 5, "resume_api_calls": None, "claim_token": None, "reason": None,
        "created_at": now, "updated_at": now,
    }
    record.update(overrides)
    return record


def test_reservation_is_bound_to_profile_generation_and_live_source(db):
    _source(db)
    with pytest.raises(RolloverConflict):
        reserve(db, _claim(profile="other"))
    with pytest.raises(RolloverConflict):
        reserve(db, _claim(generation=2))  # the count moved since the gate read it
    stored = reserve(db, _claim())
    assert read_source(db, "SRC").claim == stored
    assert read_source(db, "SRC").full_compaction_count == 3
    with pytest.raises(RolloverConflict) as excinfo:
        reserve(db, _claim(rollover_id="ro_ffffffffffffffff", target_session_id="TGT2"))
    assert excinfo.value.record["rollover_id"] == "ro_0123456789abcdef"


def test_ended_source_cannot_be_reserved(db):
    _source(db)
    db.end_session("SRC", "new_session")
    with pytest.raises(RolloverConflict):
        reserve(db, _claim())


def test_only_an_exact_prepared_record_can_be_superseded(db):
    _source(db)
    first = reserve(db, _claim())
    with pytest.raises(RolloverConflict):
        reserve(db, _claim(rollover_id="ro_1111111111111111"), supersede={**first, "budget_remaining": 1})
    second = reserve(db, _claim(rollover_id="ro_1111111111111111"), supersede=first)
    claimed = advance(db, "SRC", second["rollover_id"], "prepared", "claimed", updates={"claim_token": "tok"})
    with pytest.raises(RolloverConflict):
        reserve(db, _claim(rollover_id="ro_2222222222222222"), supersede=claimed)


def test_cas_transitions_have_exactly_one_winner(db):
    _source(db)
    reserve(db, _claim())
    barrier = threading.Barrier(4)
    wins, losses = [], []

    def claim(token):
        barrier.wait(timeout=5)
        try:
            wins.append(advance(db, "SRC", "ro_0123456789abcdef", "prepared", "claimed",
                                updates={"claim_token": token}))
        except RolloverConflict as exc:
            losses.append(exc.record["state"])

    threads = [threading.Thread(target=claim, args=(f"t{i}",)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert len(wins) == 1 and losses == ["claimed"] * 3
    assert read_source(db, "SRC").claim["claim_token"] == wins[0]["claim_token"]


def test_illegal_transitions_and_identity_rewrites_are_refused(db):
    _source(db)
    reserve(db, _claim())
    with pytest.raises(ValueError):
        advance(db, "SRC", "ro_0123456789abcdef", "prepared", "started")
    with pytest.raises(ValueError):
        advance(db, "SRC", "ro_0123456789abcdef", "prepared", "claimed", updates={"target_session_id": "X"})
    with pytest.raises(RolloverConflict):
        advance(db, "SRC", "ro_ffffffffffffffff", "prepared", "claimed")
    advance(db, "SRC", "ro_0123456789abcdef", "prepared", "claimed")
    advance(db, "SRC", "ro_0123456789abcdef", "claimed", "installed", end_source=True)
    advance(db, "SRC", "ro_0123456789abcdef", "installed", "started")
    with pytest.raises(ValueError):
        advance(db, "SRC", "ro_0123456789abcdef", "started", "prepared")  # a started resume is never released
    row = db.get_session("SRC")
    assert row["ended_at"] is not None and row["end_reason"] == SOURCE_END_REASON


def test_ordinary_writers_cannot_erase_or_forge_transition_keys(db):
    _source(db)
    stored = reserve(db, _claim())
    stale = json.dumps({"keep": "stale-snapshot"})
    db.update_session_meta("SRC", stale, "m2")
    db.update_session_meta("SRC", json.dumps({ROLLOVER_CLAIM_KEY: {**stored, "state": "completed"}}))
    db.patch_session_model_config("SRC", {"other": 1})
    db.update_conversation_model_config_value("SRC", "other", lambda v: (v or 0) + 1)
    db.archive_and_compact("SRC", [{"role": "user", "content": "pruned"}], model_config_patch={"x": 1})
    assert read_source(db, "SRC").claim == stored
    db.create_session("T", "cli", model_config={"a": 1})
    db.update_session_meta("T", json.dumps({ROLLOVER_ORIGIN_KEY: {"forged": True}}))
    assert ROLLOVER_ORIGIN_KEY not in json.loads(db.get_session("T")["model_config"])


def test_target_is_a_fresh_root_with_origin_and_no_source_ledger(db):
    _source(db)
    claim = advance(db, "SRC", reserve(db, _claim())["rollover_id"], "prepared", "claimed")
    origin = bind_target(db, claim, source="cli", model="m", cwd="/work",
                         model_config={"reasoning_config": {"effort": "high"}, FULL_COMPACTION_COUNT_KEY: 3,
                                       ROLLOVER_CLAIM_KEY: claim})
    assert origin == origin_for(claim)
    target = read_source(db, "TGT")
    assert target.full_compaction_count == 0 and target.claim is None and target.origin == origin
    row = db.get_session("TGT")
    assert row["parent_session_id"] is None and row["profile_name"] == "work"
    assert json.loads(row["model_config"])["reasoning_config"] == {"effort": "high"}
    source = read_source(db, "SRC")  # the source keeps its own ledger, claim and other metadata
    assert source.full_compaction_count == 3 and source.claim == claim
    assert db.get_session_model_config_value("SRC", "keep") == "me"
    # Re-binding the same claim (recovery before anything ran) accepts its own empty target.
    assert bind_target(db, claim, source="cli", model="m", cwd="/work", model_config={}) == origin


def test_target_id_collision_or_used_target_is_refused(db):
    _source(db)
    db.create_session("TGT", "cli", profile_name="work")
    claim = advance(db, "SRC", reserve(db, _claim())["rollover_id"], "prepared", "claimed")
    with pytest.raises(RolloverConflict):
        bind_target(db, claim, source="cli", model="m", cwd="/work", model_config={})
    db2_claim = {**claim, "target_session_id": "TGT2"}
    bind_target(db, db2_claim, source="cli", model="m", cwd="/work", model_config={})
    db.append_message(session_id="TGT2", role="user", content="something already ran here")
    with pytest.raises(RolloverConflict):
        bind_target(db, db2_claim, source="cli", model="m", cwd="/work", model_config={})


def _claimed(db):
    _source(db)
    return advance(db, "SRC", reserve(db, _claim())["rollover_id"], "prepared", "claimed",
                   updates={"claim_token": "tok"})


def _profile(db, name):
    db._conn.execute("UPDATE sessions SET profile_name = ? WHERE id = 'SRC'", (name,))
    db._conn.commit()


@pytest.mark.parametrize("move", [
    lambda db: db.try_acquire_compression_lock("SRC", "another-compressor"),
    lambda db: db.patch_session_model_config("SRC", {FULL_COMPACTION_COUNT_KEY: 4}),
    lambda db: db.end_session("SRC", "new_session"),
    lambda db: _profile(db, "other"),
    lambda db: db.append_message(session_id="SRC", role="user", content="newer"),
], ids=["compression_lock", "count_moved", "ended", "profile", "newer_row"])
def test_installed_edge_is_fenced_on_the_live_source_in_one_transaction(db, move):
    _claimed(db)
    mark = messages_mark(db, "SRC")
    move(db)
    with pytest.raises(RolloverConflict):
        advance(db, "SRC", "ro_0123456789abcdef", "claimed", "installed", end_source=True, messages_mark=mark)
    assert read_source(db, "SRC").claim["state"] == "claimed"  # nothing of the refused edge landed


def test_installed_edge_commits_and_ends_the_unchanged_source(db):
    _claimed(db)
    db.release_compression_lock("SRC", "nobody")
    installed = advance(db, "SRC", "ro_0123456789abcdef", "claimed", "installed", end_source=True,
                        messages_mark=messages_mark(db, "SRC"))
    assert installed["state"] == "installed"
    assert db.get_session("SRC")["end_reason"] == SOURCE_END_REASON


@pytest.mark.parametrize("stored", [
    {"state": "prepared"}, "nope", {**_claim(), "state": "weird"}, {**_claim(), "generation": True},
    # A prototype claim from before the wall-clock cap was recorded, and caps/times that are not real.
    {k: v for k, v in _claim().items() if k != "run_budget_seconds"},
    {**_claim(), "run_budget_seconds": 0.0}, {**_claim(), "run_budget_seconds": -5.0},
    {**_claim(), "run_budget_seconds": 600}, {**_claim(), "run_budget_seconds": True},
    {**_claim(), "run_budget_seconds": float("inf")}, {**_claim(), "run_budget_started_at": float("nan")},
    {**_claim(), "created_at": float("-inf")},
])
def test_corrupt_claim_fails_closed(db, stored):
    _source(db)
    db.patch_session_model_config("SRC", {ROLLOVER_CLAIM_KEY: stored})
    with pytest.raises(RolloverRecordError):
        read_source(db, "SRC")
    with pytest.raises(RolloverRecordError):
        advance(db, "SRC", "ro_0123456789abcdef", "prepared", "claimed")


def test_unreadable_source_config_fails_closed(db):
    _source(db)
    db._conn.execute("UPDATE sessions SET model_config = 'not-json' WHERE id = 'SRC'")
    db._conn.commit()
    with pytest.raises(RolloverRecordError):
        read_source(db, "SRC")
    with pytest.raises(RolloverRecordError):
        reserve(db, _claim())
    with pytest.raises(RolloverRecordError):
        read_source(db, "MISSING")
