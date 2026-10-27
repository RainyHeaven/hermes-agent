"""``session.active_list`` ``workspace_cwd``: scope the open sessions to one project (FB35-H2).

Same key as ``session.list`` (git main repo root, else the realpath; catch-all folders fall back to
the full list). A live session is kept when its cwd is at/under the key; the current tab always stays.
"""

import os
import threading

import pytest

import tui_gateway.server as srv
import tui_gateway.methods_session
from tui_gateway import git_probe


@pytest.fixture(autouse=True)
def _no_git(monkeypatch):
    # tmp folders are not repos; pin that so a tmp dir inside a checkout can't change the key.
    monkeypatch.setattr(git_probe, "common_repo_root", lambda cwd: "")
    monkeypatch.setattr(srv, "_resolve_model", lambda: "test-model")


@pytest.fixture
def work(tmp_path):
    # Resolved: on macOS tmp_path sits under /var -> /private/var, and the key is a realpath.
    return tmp_path.resolve() / "work"


@pytest.fixture
def live(request):
    """Add minimal live session records; pop them after the test."""
    added: list[str] = []
    prefix = f"active-{request.node.name}-"

    def add(name: str, cwd) -> str:
        sid = prefix + name
        srv._sessions[sid] = {
            "agent": None,
            "agent_ready": threading.Event(),
            "agent_error": None,
            "attached_images": [],
            "cwd": str(cwd) if cwd is not None else None,
            "history": [],
            "history_lock": threading.RLock(),
            "history_version": 0,
            "running": False,
            "session_key": sid,
            "transport": None,
        }
        added.append(sid)
        return sid

    yield add
    for sid in added:
        srv._sessions.pop(sid, None)


def _call(params: dict) -> dict:
    return srv._methods["session.active_list"](1, params)["result"]


def _ids(result: dict) -> set[str]:
    return {row["id"] for row in result["sessions"]}


def test_no_workspace_lists_all_without_workspace_key(live, work):
    ours = {live("a", work / "proj"), live("b", work / "other"), live("c", None)}
    for params in ({}, {"workspace_cwd": ""}):
        result = _call(params)
        assert ours <= _ids(result)
        assert "workspace" not in result


def test_plain_folder_keeps_its_cwd_and_below(live, work):
    proj = work / "proj"
    here = live("here", proj)
    below = live("below", proj / "sub" / "deep")
    live("elsewhere", work / "other")
    live("nocwd", None)
    result = _call({"workspace_cwd": str(proj)})
    assert _ids(result) == {here, below}
    assert result["workspace"] == str(proj)


def test_sibling_folder_with_shared_prefix_is_excluded(live, work):
    foo = live("foo", work / "foo")
    live("foobar", work / "foobar")
    result = _call({"workspace_cwd": str(work / "foo")})
    assert _ids(result) == {foo}


def test_current_session_stays_even_from_another_folder(live, work):
    here = live("here", work / "proj")
    current = live("current", work / "other")
    result = _call({"workspace_cwd": str(work / "proj"), "current_session_id": current})
    assert _ids(result) == {here, current}
    assert next(r for r in result["sessions"] if r["id"] == current)["current"] is True


def test_broad_folder_lists_all_with_null_workspace(live, work):
    ours = {live("a", work / "proj"), live("b", work / "other")}
    result = _call({"workspace_cwd": os.path.expanduser("~")})
    assert ours <= _ids(result)
    assert result["workspace"] is None
