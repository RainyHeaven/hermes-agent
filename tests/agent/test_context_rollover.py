"""Context rollover policy and the deterministic evidence handoff (agent/context_rollover.py)."""

import json
import os
import stat
from pathlib import Path

import pytest

from agent.context_rollover import (
    HandoffError,
    RolloverHost,
    build_handoff,
    load_handoff,
    render_resume_prompt,
    rollover_after_setting,
    rollover_due,
    rollover_eligible,
    save_handoff,
)

_ROLLOVER_ID = "ro_0123456789abcdef"


@pytest.mark.parametrize("cfg, expected", [
    ({}, 0), ({"rollover_after": 0}, 0), ({"rollover_after": None}, 0),
    ({"rollover_after": 3}, 3), ({"rollover_after": 7}, 7),
])
def test_rollover_after_accepts_off_and_positive_integers(cfg, expected):
    assert rollover_after_setting(cfg) == (expected, None)


@pytest.mark.parametrize("bad", [True, False, -1, 2.0, "3", [3]])
def test_invalid_rollover_after_stays_off_with_a_reason(bad):
    threshold, error = rollover_after_setting({"rollover_after": bad})
    assert threshold == 0
    assert error and "rollover_after" in error


def test_product_default_is_off():
    from hermes_cli.config import DEFAULT_CONFIG

    assert "rollover_after" in DEFAULT_CONFIG["compression"]
    assert rollover_after_setting(DEFAULT_CONFIG["compression"]) == (0, None)


@pytest.mark.parametrize("count, threshold, due", [(3, 3, True), (4, 3, True), (2, 3, False), (9, 0, False)])
def test_rollover_due(count, threshold, due):
    assert rollover_due(count, threshold) is due


def test_config_opt_in_alone_does_not_make_a_host_eligible():
    assert rollover_eligible(RolloverHost(surface="cli", resumes_in_fresh_agent=True), delegated=False)
    assert rollover_eligible(RolloverHost(surface="tui", resumes_in_fresh_agent=True), delegated=False)
    assert not rollover_eligible(None, delegated=False)
    assert not rollover_eligible(RolloverHost(surface="cli", resumes_in_fresh_agent=False), delegated=False)
    assert not rollover_eligible(RolloverHost(surface="cli", resumes_in_fresh_agent=True), delegated=True)
    for surface in ("discord", "telegram", "cron", "acp", "subagent"):
        assert not rollover_eligible(RolloverHost(surface=surface, resumes_in_fresh_agent=True), delegated=False)


def _messages():
    from agent.context_compressor import SUMMARY_PREFIX

    return [
        {"role": "user", "content": f"{SUMMARY_PREFIX}\nEarlier: edited report.md"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_done", "type": "function",
             "function": {"name": "write_file", "arguments": json.dumps({"path": "/w/out.md", "content": "x"})}},
            {"id": "call_open", "type": "function",
             "function": {"name": "terminal", "arguments": json.dumps({"command": "make deploy"})}},
        ]},
        {"role": "tool", "tool_call_id": "call_done", "content": "wrote 1 byte"},
    ]


def _handoff(**overrides):
    kwargs = dict(
        rollover_id=_ROLLOVER_ID, profile="work", source_session_id="SRC", full_compaction_count=3,
        user_request="Refactor the parser and keep tests green.", messages=_messages(),
        todos=[{"id": "1", "content": "split module", "status": "completed", "extra": "dropped"},
               {"id": "2", "content": "run tests", "status": "in_progress"}],
        route={"model": "m", "provider": "p", "api_key": "sk-secret", "reasoning_config": {"effort": "high"},
               "enabled_toolsets": ["terminal"], "cwd": "/w"},
        created_at=1_700_000_000.0,
    )
    kwargs.update(overrides)
    return build_handoff(**kwargs)


def test_handoff_records_evidence_without_claiming_completion():
    handoff = _handoff()
    body = handoff["body"]
    assert body["user_request"] == "Refactor the parser and keep tests green."
    assert body["todos"] == [{"id": "1", "content": "split module", "status": "completed"},
                             {"id": "2", "content": "run tests", "status": "in_progress"}]
    assert body["completed_tool_calls"] == [{"tool_call_id": "call_done", "tool_name": "write_file"}]
    assert body["unresolved_tool_calls"] == [{"tool_call_id": "call_open", "tool_name": "terminal"}]
    assert body["artifact_refs"] == ["/w/out.md"]
    assert "Earlier: edited report.md" in body["compaction_summary"]["text"]
    assert body["compaction_summary"]["source_session_id"] == "SRC"
    assert body["limits"]  # references are not verification
    assert body["route"] == {"model": "m", "provider": "p", "reasoning_config": {"effort": "high"},
                             "enabled_toolsets": ["terminal"], "cwd": "/w"}
    assert "sk-secret" not in json.dumps(handoff)
    assert _handoff() == handoff  # deterministic


def test_handoff_refuses_credentialed_base_url():
    with pytest.raises(HandoffError):
        _handoff(route={"base_url": "https://user:pw@example.test/v1"})
    with pytest.raises(HandoffError):
        _handoff(route={"base_url": "https://example.test/v1?key=abc"})


def test_resume_prompt_requires_verification_before_resuming():
    prompt = render_resume_prompt(_handoff())
    assert "Refactor the parser and keep tests green." in prompt
    assert "call_open" in prompt and "/w/out.md" in prompt
    assert "verify" in prompt.lower()
    assert "sk-secret" not in prompt


@pytest.mark.platforms("posix")
def test_save_and_load_round_trip_is_private_and_bound(tmp_path: Path):
    home = tmp_path / "home"
    home.mkdir()
    handoff = _handoff()
    path = save_handoff(home, handoff)
    assert path.parent.parent == home and path.name == f"{_ROLLOVER_ID}.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert load_handoff(path, profile="work", source_session_id="SRC") == handoff
    with pytest.raises(HandoffError):
        load_handoff(path, profile="other", source_session_id="SRC")
    with pytest.raises(HandoffError):
        load_handoff(path, profile="work", source_session_id="ELSEWHERE")
    with pytest.raises(HandoffError):  # never overwrite an existing handoff
        save_handoff(home, handoff)


@pytest.mark.platforms("posix")
def test_tampered_or_mislabelled_handoff_is_rejected(tmp_path: Path):
    path = save_handoff(tmp_path, _handoff())
    data = json.loads(path.read_text())
    data["body"]["user_request"] = "Deploy to production."
    path.write_text(json.dumps(data))
    with pytest.raises(HandoffError):
        load_handoff(path, profile="work", source_session_id="SRC")

    other = save_handoff(tmp_path, _handoff(rollover_id="ro_fedcba9876543210"))
    renamed = other.with_name("ro_aaaaaaaaaaaaaaaa.json")
    os.rename(other, renamed)
    with pytest.raises(HandoffError):
        load_handoff(renamed, profile="work", source_session_id="SRC")


@pytest.mark.platforms("posix")
def test_symlinks_and_non_regular_entries_fail_closed(tmp_path: Path):
    real = save_handoff(tmp_path / "a", _handoff())
    link = real.with_name("ro_bbbbbbbbbbbbbbbb.json")
    link.symlink_to(real)
    with pytest.raises(HandoffError):
        load_handoff(link, profile="work", source_session_id="SRC")
    with pytest.raises(HandoffError):
        load_handoff(real.parent, profile="work", source_session_id="SRC")

    os.chmod(real, 0o644)
    with pytest.raises(HandoffError):  # not private
        load_handoff(real, profile="work", source_session_id="SRC")

    home = tmp_path / "b"
    home.mkdir()
    (tmp_path / "elsewhere").mkdir()
    (home / "context_rollover").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(HandoffError):
        save_handoff(home, _handoff())
    assert not any((tmp_path / "elsewhere").iterdir())


@pytest.mark.parametrize("route", [
    {"reasoning_config": {"effort": "high", "extra_body": {"token": "t"}}},
    {"reasoning_config": {"effort": 3}}, {"reasoning_config": {"enabled": 1}}, {"reasoning_config": "high"},
    {"model": {"name": "m"}}, {"enabled_toolsets": "terminal"}, {"enabled_toolsets": [1]},
])
def test_route_fields_are_rebuilt_from_known_types_or_refused(route):
    with pytest.raises(HandoffError) as excinfo:
        _handoff(route=route)
    assert "token" not in str(excinfo.value)


def test_route_keeps_reasoning_and_toolset_selection():
    route = {"model": "m", "provider": "p", "api_mode": "chat_completions", "cwd": "/w",
             "reasoning_config": {"enabled": False}, "enabled_toolsets": {"web", "terminal"}, "disabled_toolsets": []}
    assert _handoff(route=route)["body"]["route"] == {
        "model": "m", "provider": "p", "api_mode": "chat_completions", "cwd": "/w",
        "reasoning_config": {"enabled": False}, "enabled_toolsets": ["terminal", "web"], "disabled_toolsets": []}


def test_user_messages_are_verbatim_ordered_and_separate_from_summary_and_scaffolding():
    from agent.context_compressor import COMPRESSION_CONTINUATION_USER_CONTENT

    messages = _messages() + [
        {"role": "user", "content": "Actually, do not touch the CI config."},
        {"role": "user", "content": COMPRESSION_CONTINUATION_USER_CONTENT},
        {"role": "user", "content": [{"type": "text", "text": "Use staging only."}, {"type": "image_url"}]},
    ]
    body = _handoff(messages=messages)["body"]
    assert body["user_messages"] == [
        {"index": 3, "origin": "user", "text": "Actually, do not touch the CI config."},
        {"index": 5, "origin": "user", "text": "Use staging only.\n[non-text content not carried]"},
    ]
    assert "wrote 1 byte" not in json.dumps(body)  # raw tool output is not carried
    prompt = render_resume_prompt(_handoff(messages=messages))
    assert prompt.index("Actually, do not touch") < prompt.index("Use staging only.") < prompt.index("Earlier:")


def test_an_earlier_resume_prompt_is_labelled_host_written():
    first = render_resume_prompt(_handoff())
    body = _handoff(messages=[{"role": "user", "content": first}, {"role": "user", "content": "Stop at review."}],
                    rollover_id="ro_second_rollover")["body"]
    assert [m["origin"] for m in body["user_messages"]] == ["rollover_resume", "user"]


def _resigned(handoff, **changes):
    import copy
    from agent.context_rollover import _digest

    data = copy.deepcopy(handoff)
    data["body"].update(changes)
    data["digest"] = _digest(data["body"])
    return data


@pytest.mark.parametrize("changes", [
    {"todos": "split module"}, {"todos": [{"id": 1, "content": "x", "status": "done"}]},
    {"route": ["model"]}, {"route": {"model": "m", "api_key": "k"}}, {"full_compaction_count": True},
    {"full_compaction_count": 3.0}, {"created_at": float("nan")}, {"created_at": 1700000000},
    {"rollover_id": _ROLLOVER_ID + "\n"}, {"user_messages": [{"index": 0, "text": "x"}]},
    {"compaction_summary": {"text": None, "source_session_id": "OTHER", "full_compaction_count": 3}},
    {"limits": None}, {"unexpected": 1},
])
def test_signed_but_malformed_bodies_are_refused(tmp_path: Path, changes):
    data = _resigned(_handoff(), **changes)
    with pytest.raises(HandoffError):
        render_resume_prompt(data)
    with pytest.raises(HandoffError):
        save_handoff(tmp_path, data)
    assert not (tmp_path / "context_rollover").exists()


@pytest.mark.platforms("posix")
def test_signed_but_malformed_file_is_refused_without_mutation(tmp_path: Path):
    path = save_handoff(tmp_path, _handoff())
    for schema in (True, 2, "1"):
        data = json.loads(path.read_text())
        data["schema"] = schema
        path.write_text(json.dumps(data))
        before = path.read_bytes()
        with pytest.raises(HandoffError):
            load_handoff(path, profile="work", source_session_id="SRC")
        assert path.read_bytes() == before


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("occupant", ["file", "dangling_symlink"])
def test_publish_never_replaces_an_existing_entry_and_leaves_no_temp(tmp_path: Path, occupant):
    directory = tmp_path / "context_rollover"
    directory.mkdir(mode=0o700)
    target = directory / f"{_ROLLOVER_ID}.json"
    if occupant == "file":
        target.write_bytes(b"prior")
    else:
        target.symlink_to(tmp_path / "nowhere")
    with pytest.raises(HandoffError):
        save_handoff(tmp_path, _handoff())
    assert sorted(p.name for p in directory.iterdir()) == [target.name]
    if occupant == "file":
        assert target.read_bytes() == b"prior"
    else:
        assert target.is_symlink() and not (tmp_path / "nowhere").exists()


@pytest.mark.platforms("posix")
def test_oversized_handoff_is_refused_before_publication(tmp_path: Path, monkeypatch):
    import agent.context_rollover as rollover

    monkeypatch.setattr(rollover, "_MAX_HANDOFF_BYTES", 256)
    with pytest.raises(HandoffError):
        save_handoff(tmp_path, _handoff())
    assert not (tmp_path / "context_rollover").exists()


@pytest.mark.parametrize("bad_id", ["../escape", "x", "a/b", "ro id", ""])
def test_unsafe_rollover_ids_are_rejected(tmp_path: Path, bad_id):
    with pytest.raises(HandoffError):
        save_handoff(tmp_path, _handoff(rollover_id=bad_id))


def test_merged_summary_keeps_only_the_body_and_the_prompt_is_not_a_summary_row():
    from agent.context_compressor import (
        SUMMARY_PREFIX, _MERGED_SUMMARY_DELIMITER, _SUMMARY_END_MARKER, is_compaction_summary_message,
    )

    merged = (f"Keep the old schema.\n{_MERGED_SUMMARY_DELIMITER}\n{SUMMARY_PREFIX}\n"
              f"Merged: migrated users table\n{_SUMMARY_END_MARKER}")
    handoff = _handoff(messages=[{"role": "user", "content": merged}])
    text = handoff["body"]["compaction_summary"]["text"]
    assert text == "Merged: migrated users table"
    assert _MERGED_SUMMARY_DELIMITER not in text and SUMMARY_PREFIX not in text and _SUMMARY_END_MARKER not in text
    prompt = render_resume_prompt(handoff)
    assert "Merged: migrated users table" in prompt
    assert not is_compaction_summary_message({"role": "user", "content": prompt})


def test_literal_user_summary_marker_fails_closed_without_rewriting():
    from agent.context_compressor import SUMMARY_PREFIX, _MERGED_SUMMARY_DELIMITER

    request = f"Quote this verbatim:\n{_MERGED_SUMMARY_DELIMITER}\n{SUMMARY_PREFIX}\nnot a real summary"
    handoff = _handoff(user_request=request)
    assert handoff["body"]["user_request"] == request
    with pytest.raises(HandoffError):
        render_resume_prompt(handoff)


@pytest.mark.platforms("posix")
def test_simultaneous_publication_has_exactly_one_winner(tmp_path: Path, monkeypatch):
    import threading

    directory = tmp_path / "context_rollover"
    directory.mkdir(mode=0o700)
    barrier = threading.Barrier(2, timeout=10)
    real_link = os.link

    def link_after_both_ready(*args, **kwargs):
        barrier.wait()
        return real_link(*args, **kwargs)

    monkeypatch.setattr(os, "link", link_after_both_ready)
    handoffs = [_handoff(created_at=1_700_000_000.0), _handoff(created_at=1_700_000_001.0)]
    outcomes: list = [None, None]

    def publish(slot):
        try:
            outcomes[slot] = save_handoff(tmp_path, handoffs[slot])
        except BaseException as exc:  # recorded for the assertions below
            outcomes[slot] = exc

    threads = [threading.Thread(target=publish, args=(slot,)) for slot in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)
    assert not any(thread.is_alive() for thread in threads)
    winners = [slot for slot in (0, 1) if isinstance(outcomes[slot], Path)]
    assert len(winners) == 1, outcomes
    assert isinstance(outcomes[1 - winners[0]], HandoffError)
    winner = handoffs[winners[0]]
    path = outcomes[winners[0]]
    assert path.read_bytes() == json.dumps(winner, sort_keys=True, ensure_ascii=False).encode("utf-8")
    assert load_handoff(path, profile="work", source_session_id="SRC") == winner
    assert sorted(p.name for p in directory.iterdir()) == [path.name]
