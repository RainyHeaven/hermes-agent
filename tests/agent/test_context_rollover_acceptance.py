"""Coordinator-owned behavioral acceptance for rollover boundaries."""
import json
import hashlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from agent.context_rollover import (
    HandoffError, RolloverStateError, build_handoff, load_handoff,
    read_full_compaction_count, render_resume_prompt, save_handoff,
)
from hermes_state import SessionDB


def capsule(**overrides):
    args: dict[str, Any] = dict(rollover_id="ro_acceptance_123456", profile="test", source_session_id="source",
                full_compaction_count=3, user_request="Implement the approved change.",
                messages=[], created_at=1700000000.0)
    args.update(overrides)
    return build_handoff(**args)


def test_malformed_metadata_is_not_a_trustworthy_zero(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("source", source="cli", model="test")
        db.update_session_meta("source", "not-json")
        with pytest.raises(RolloverStateError):
            read_full_compaction_count(db, "source")
    finally:
        db.close()


def test_later_user_restrictions_survive_the_handoff():
    from agent.context_compressor import SUMMARY_PREFIX
    restriction = "Do not deploy or publish; only run local tests."
    messages = [
        {"role": "user", "content": SUMMARY_PREFIX + "\nEarlier approved implementation."},
        {"role": "assistant", "content": "I will proceed."},
        {"role": "user", "content": restriction},
    ]
    assert restriction in render_resume_prompt(capsule(messages=messages))


def test_resume_request_is_not_misclassified_as_a_compaction_summary():
    from agent.context_compressor import (
        SUMMARY_PREFIX, _MERGED_PRIOR_CONTEXT_HEADER, _MERGED_SUMMARY_DELIMITER,
        _SUMMARY_END_MARKER, is_compaction_summary_message,
    )
    merged = "\n".join([
        _MERGED_PRIOR_CONTEXT_HEADER, "Preserve the approved local-only scope.",
        _MERGED_SUMMARY_DELIMITER, SUMMARY_PREFIX, "Prior verified work remains done.",
        _SUMMARY_END_MARKER,
    ])
    prompt = render_resume_prompt(capsule(messages=[{"role": "user", "content": merged}]))
    assert "Preserve the approved local-only scope." in prompt
    assert "Prior verified work remains done." in prompt
    assert not is_compaction_summary_message({"role": "user", "content": prompt})


def test_nested_route_secrets_are_not_serialized():
    secret = "DO_NOT_PERSIST_AUTH_SENTINEL"
    try:
        data = capsule(route={"reasoning_config": {"effort": "high", "api_key": secret}})
    except HandoffError:
        return
    assert secret not in json.dumps(data)


def test_valid_digest_does_not_make_an_incomplete_body_valid(tmp_path):
    data = capsule()
    del data["body"]["user_request"]
    canonical = json.dumps(data["body"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    data["digest"] = "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
    directory = tmp_path / "context_rollover"
    directory.mkdir(mode=0o700)
    path = directory / "ro_acceptance_123456.json"
    path.write_text(json.dumps(data))
    path.chmod(0o600)
    with pytest.raises(HandoffError):
        load_handoff(path, profile="test", source_session_id="source")


@pytest.mark.platforms("posix")
def test_concurrent_publish_has_exactly_one_winner(tmp_path, monkeypatch):
    import utils
    directory = tmp_path / "context_rollover"
    directory.mkdir(mode=0o700)
    barrier = threading.Barrier(2)
    atomic_write = utils.atomic_write_bytes

    def simultaneous_write(*args, **kwargs):
        barrier.wait(timeout=5)
        return atomic_write(*args, **kwargs)

    monkeypatch.setattr(utils, "atomic_write_bytes", simultaneous_write)

    def publish(label):
        data = capsule(user_request=label)
        try:
            save_handoff(tmp_path, data)
            return label
        except HandoffError:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(publish, ["first", "second"]))
    winners = [r for r in results if r is not None]
    assert len(winners) == 1, "both writers reported success for one immutable handoff"
    stored = load_handoff(directory / "ro_acceptance_123456.json", profile="test", source_session_id="source")
    assert stored["body"]["user_request"] == winners[0]


@pytest.mark.platforms("posix")
def test_fifo_is_rejected_without_waiting_for_a_writer(tmp_path):
    directory = tmp_path / "context_rollover"
    directory.mkdir(mode=0o700)
    path = directory / "ro_acceptance_123456.json"
    os.mkfifo(path, 0o600)
    done = threading.Event()
    outcomes = []

    def read():
        try:
            load_handoff(path, profile="test", source_session_id="source")
            outcomes.append("accepted")
        except HandoffError:
            outcomes.append("rejected")
        finally:
            done.set()

    worker = threading.Thread(target=read, daemon=True)
    worker.start()
    timely = done.wait(2)
    rescue = None
    try:
        if not timely:
            rescue = os.open(path, os.O_RDWR | os.O_NONBLOCK)
        worker.join(timeout=5)
    finally:
        if rescue is not None:
            os.close(rescue)
    assert not worker.is_alive(), "test reader cleanup failed"
    assert timely, "non-regular FIFO blocked the rollover boundary"
    assert outcomes == ["rejected"]
