"""Durable full-compaction ledger behind ``compression.rollover_after`` (context rollover).

The count a rollover policy reads must come from the same transaction that commits the
compaction, survive process/agent recreation and legacy rotation, and ignore every path that is
not a successful full compaction (no progress, lease contention, failed commit, unrelated
``archive_and_compact`` callers such as prune / micro-compaction).
"""

import json
import os
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.context_rollover import FULL_COMPACTION_COUNT_KEY, RolloverStateError, read_full_compaction_count
from hermes_state import SessionDB


def _make_agent(db, sid, *, in_place=True, progress=True, summary_fallback=False):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
            quiet_mode=True, session_db=db, session_id=sid, platform="cli",
            skip_context_files=True, skip_memory=True,
        )
    agent.compression_in_place = in_place
    compressor = agent.context_compressor

    def _fake_compress(messages, current_tokens=None, focus_topic=None, force=False):
        compressor._last_compression_made_progress = progress
        compressor._last_summary_fallback_used = summary_fallback
        return [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary of prior turns"},
            {"role": "assistant", "content": "recent reply 1"},
            {"role": "user", "content": "follow-up q"},
            {"role": "assistant", "content": "recent reply 2"},
        ]

    compressor.compress = _fake_compress
    compressor._last_compress_aborted = False
    compressor._last_summary_error = None
    agent._session_db_created = True
    return agent


def _seed(db, sid, n=8):
    db.create_session(sid, "cli", model="test/model")
    for i in range(n):
        db.append_message(session_id=sid, role="user" if i % 2 == 0 else "assistant", content=f"seed msg {i}")


def _compact(agent, db):
    from agent.conversation_compression import compress_context

    messages = db.get_messages_as_conversation(agent.session_id)
    agent._last_flushed_db_idx = len(messages)
    return compress_context(agent, messages, approx_tokens=100_000, system_message="sys")


def _store_raw_config(db, sid, raw):
    db._conn.execute("UPDATE sessions SET model_config = ? WHERE id = ?", (raw, sid))
    db._conn.commit()


def _active_rows(db, sid):
    return db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = ? AND active = 1", (sid,)).fetchone()[0]


def test_in_place_full_compaction_counts_and_survives_process_recreation(tmp_path: Path):
    db_path = tmp_path / "state.db"
    sid = "ROLLOVER_IN_PLACE"
    with closing(SessionDB(db_path=db_path)) as db:
        _seed(db, sid)
        assert read_full_compaction_count(db, sid) == 0
        _compact(_make_agent(db, sid), db)
        assert _active_rows(db, sid) == 4  # the compaction really committed
        assert read_full_compaction_count(db, sid) == 1

    # A fresh store + fresh agent (restart / eviction) continues the same durable count.
    with closing(SessionDB(db_path=db_path)) as db:
        assert read_full_compaction_count(db, sid) == 1
        agent = _make_agent(db, sid)
        assert agent.context_compressor.compression_count == 0  # memory counter is not the source of truth
        _compact(agent, db)
        assert read_full_compaction_count(db, sid) == 2


def test_no_progress_compaction_does_not_count(tmp_path: Path):
    with closing(SessionDB(db_path=tmp_path / "state.db")) as db:
        sid = "ROLLOVER_NO_PROGRESS"
        _seed(db, sid)
        _compact(_make_agent(db, sid, progress=False), db)
        assert read_full_compaction_count(db, sid) == 0


@pytest.mark.parametrize("in_place", [True, False])
def test_summary_fallback_compaction_commits_but_does_not_count(tmp_path: Path, in_place):
    """A compaction whose summary call failed drops turns behind a placeholder: it still commits, but a
    rollover must not treat that lost context as a successful full compaction."""
    with closing(SessionDB(db_path=tmp_path / "state.db")) as db:
        sid = f"ROLLOVER_FALLBACK_{in_place}"
        _seed(db, sid)
        agent = _make_agent(db, sid, in_place=in_place, summary_fallback=True)
        _compact(agent, db)
        live = agent.session_id
        assert _active_rows(db, live) == 4  # the compaction itself still committed
        assert read_full_compaction_count(db, live) == 0
        # A later compaction with a real summary counts normally on the same conversation.
        good = _make_agent(db, live, in_place=in_place)
        _compact(good, db)
        assert read_full_compaction_count(db, good.session_id) == 1


def test_lease_contention_skip_does_not_count(tmp_path: Path):
    with closing(SessionDB(db_path=tmp_path / "state.db")) as db:
        sid = "ROLLOVER_LOCKED"
        _seed(db, sid)
        assert db.try_acquire_compression_lock(sid, "another-surface")
        _compact(_make_agent(db, sid), db)
        assert _active_rows(db, sid) == 8
        assert read_full_compaction_count(db, sid) == 0


def test_failed_in_place_transaction_does_not_count(tmp_path: Path):
    from hermes_state import SessionCompressionInProgressError

    with closing(SessionDB(db_path=tmp_path / "state.db")) as db:
        sid = "ROLLOVER_LOST_LEASE"
        _seed(db, sid)
        with pytest.raises(SessionCompressionInProgressError):
            db.archive_and_compact(
                sid, [{"role": "user", "content": "summary"}], lock_holder="not-the-holder",
                count_full_compaction=True)
        assert _active_rows(db, sid) == 8
        assert read_full_compaction_count(db, sid) == 0


def test_archive_without_full_compaction_intent_does_not_count(tmp_path: Path):
    """Prune and micro-compaction reuse archive_and_compact; only the full-compaction intent counts."""
    with closing(SessionDB(db_path=tmp_path / "state.db")) as db:
        sid = "ROLLOVER_UNRELATED"
        _seed(db, sid)
        db.archive_and_compact(sid, [{"role": "user", "content": "pruned"}], model_config_patch={"keep": 1})
        assert read_full_compaction_count(db, sid) == 0
        db.archive_and_compact(sid, [{"role": "user", "content": "full"}], count_full_compaction=True)
        assert read_full_compaction_count(db, sid) == 1
        assert db.get_session_model_config_value(sid, "keep") == 1


def test_legacy_rotation_carries_and_increments_the_count(tmp_path: Path):
    with closing(SessionDB(db_path=tmp_path / "state.db")) as db:
        sid = "ROLLOVER_ROTATE_PARENT"
        _seed(db, sid)
        db.patch_session_model_config(sid, {FULL_COMPACTION_COUNT_KEY: 2})
        agent = _make_agent(db, sid, in_place=False)
        _compact(agent, db)
        child = agent.session_id
        assert child != sid
        assert read_full_compaction_count(db, child) == 3
        assert read_full_compaction_count(db, sid) == 2  # the closed parent keeps its own generation


def test_rotation_publication_owns_the_count(tmp_path: Path):
    """A caller-supplied model_config cannot reset the ledger; no intent carries it unchanged."""
    with closing(SessionDB(db_path=tmp_path / "state.db")) as db:
        db.create_session("P", "cli", model_config={FULL_COMPACTION_COUNT_KEY: 4})
        db.publish_compression_child(
            parent_session_id="P", child_session_id="C1", source="cli", require_compression_lease=False,
            messages=[{"role": "user", "content": "handoff"}], model_config={FULL_COMPACTION_COUNT_KEY: 0})
        assert read_full_compaction_count(db, "C1") == 4
        db.publish_compression_child(
            parent_session_id="C1", child_session_id="C2", source="cli", require_compression_lease=False,
            messages=[{"role": "user", "content": "handoff"}], model_config={FULL_COMPACTION_COUNT_KEY: 0},
            count_full_compaction=True)
        assert read_full_compaction_count(db, "C2") == 5


def test_runtime_meta_persist_cannot_clobber_the_count(tmp_path: Path):
    """TUI/gateway/ACP persist model_config wholesale; the ledger must survive that writer."""
    with closing(SessionDB(db_path=tmp_path / "state.db")) as db:
        sid = "ROLLOVER_META"
        db.create_session(sid, "tui", model_config={FULL_COMPACTION_COUNT_KEY: 2, "model": "a"})
        db.update_session_meta(sid, json.dumps({"model": "b", FULL_COMPACTION_COUNT_KEY: 0}), "b")
        assert read_full_compaction_count(db, sid) == 2
        assert db.get_session_model_config_value(sid, "model") == "b"


@pytest.mark.parametrize("corrupt", ["3", True, -1, 2.5])
def test_corrupt_count_fails_closed_without_blocking_compaction(tmp_path: Path, corrupt):
    with closing(SessionDB(db_path=tmp_path / "state.db")) as db:
        sid = "ROLLOVER_CORRUPT"
        _seed(db, sid)
        db.patch_session_model_config(sid, {FULL_COMPACTION_COUNT_KEY: corrupt})
        db.archive_and_compact(sid, [{"role": "user", "content": "full"}], count_full_compaction=True)
        assert _active_rows(db, sid) == 1
        with pytest.raises(RolloverStateError):
            read_full_compaction_count(db, sid)


@pytest.mark.parametrize("stored", ["not-json", "[]", "null", "5", json.dumps({FULL_COMPACTION_COUNT_KEY: None})])
def test_unreadable_config_is_never_reset_to_a_trustworthy_zero(tmp_path: Path, stored):
    """Compaction, prune, wholesale meta writes and rotation all commit, but the readback keeps failing."""
    with closing(SessionDB(db_path=tmp_path / "state.db")) as db:
        sid = "ROLLOVER_UNREADABLE"
        _seed(db, sid)
        _store_raw_config(db, sid, stored)
        with pytest.raises(RolloverStateError):
            read_full_compaction_count(db, sid)
        db.archive_and_compact(sid, [{"role": "user", "content": "full"}], count_full_compaction=True)
        assert _active_rows(db, sid) == 1
        with pytest.raises(RolloverStateError):
            read_full_compaction_count(db, sid)
        db.archive_and_compact(sid, [{"role": "user", "content": "pruned"}], model_config_patch={"keep": 1})
        assert db.get_session_model_config_value(sid, "keep") == 1  # unrelated fields still written
        db.update_session_meta(sid, json.dumps({"model": "b"}), "b")  # wholesale: replaces "keep"
        db.patch_session_model_config(sid, {"other": 2})
        with pytest.raises(RolloverStateError):
            read_full_compaction_count(db, sid)
        assert db.get_session_model_config_value(sid, "model") == "b"
        assert db.get_session_model_config_value(sid, "other") == 2


def test_rotation_from_unreadable_parent_does_not_restart_the_count(tmp_path: Path):
    with closing(SessionDB(db_path=tmp_path / "state.db")) as db:
        db.create_session("P", "cli")
        _store_raw_config(db, "P", "{broken")
        for child, counting in (("C1", True), ("C2", False)):
            db.publish_compression_child(
                parent_session_id="P" if child == "C1" else "C1", child_session_id=child, source="cli",
                require_compression_lease=False, messages=[{"role": "user", "content": "handoff"}],
                model_config={"keep": child}, count_full_compaction=counting)
            with pytest.raises(RolloverStateError):
                read_full_compaction_count(db, child)
            assert db.get_session_model_config_value(child, "keep") == child


def test_legacy_null_config_reads_zero(tmp_path: Path):
    with closing(SessionDB(db_path=tmp_path / "state.db")) as db:
        db.create_session("LEGACY", "cli")
        assert db.get_session("LEGACY")["model_config"] is None
        assert read_full_compaction_count(db, "LEGACY") == 0
        db.archive_and_compact("LEGACY", [{"role": "user", "content": "full"}], count_full_compaction=True)
        assert read_full_compaction_count(db, "LEGACY") == 1


def test_missing_session_fails_closed(tmp_path: Path):
    with closing(SessionDB(db_path=tmp_path / "state.db")) as db:
        with pytest.raises(RolloverStateError):
            read_full_compaction_count(db, "NOPE")
        with pytest.raises(RolloverStateError):
            read_full_compaction_count(None, "NOPE")
