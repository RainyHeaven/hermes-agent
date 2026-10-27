"""``session.list`` ``workspace_cwd``: scope the listing to one project (FB34-H).

The key is the git main repo root, else the folder's realpath; catch-all folders (home, ``/``,
HERMES_HOME) and relative paths fall back to the full listing. Compressed conversations are
judged by their root row's cwd (the root row is listed, the tip projected onto it).
"""

import time

import pytest

import tui_gateway.server as srv
import tui_gateway.methods_session
from hermes_state import SessionDB
from tui_gateway import git_probe


@pytest.fixture
def db(tmp_path, monkeypatch):
    database = SessionDB(tmp_path / "state.db")
    monkeypatch.setattr(srv, "_get_db", lambda: database)
    try:
        yield database
    finally:
        database.close()


@pytest.fixture(autouse=True)
def _no_git(monkeypatch):
    # tmp folders are not repos; pin that so a tmp dir inside a checkout can't change the key.
    monkeypatch.setattr(git_probe, "common_repo_root", lambda cwd: "")


@pytest.fixture
def work(tmp_path):
    # Resolved: on macOS tmp_path sits under /var -> /private/var, and the key is a realpath.
    return tmp_path.resolve() / "work"


def _call(method: str, params: dict) -> dict:
    return srv._methods[method](1, params)


def _seed(db, sid: str, cwd) -> None:
    db.create_session(sid, "cli")
    db.append_message(sid, "user", f"hello from {sid}")
    db._conn.execute("UPDATE sessions SET cwd=? WHERE id=?", (str(cwd), sid))
    db._conn.commit()


def _ids(result: dict) -> set[str]:
    return {row["id"] for row in result["sessions"]}


def test_no_workspace_lists_all_without_workspace_key(db, work):
    _seed(db, "a", work / "proj")
    _seed(db, "b", work / "other")
    for params in ({}, {"workspace_cwd": ""}):
        result = _call("session.list", params)["result"]
        assert _ids(result) == {"a", "b"}
        assert "workspace" not in result


def test_plain_folder_keeps_its_cwd_and_below(db, work):
    proj = work / "proj"
    _seed(db, "here", proj)
    _seed(db, "below", proj / "sub" / "deep")
    _seed(db, "elsewhere", work / "other")
    result = _call("session.list", {"workspace_cwd": str(proj)})["result"]
    assert _ids(result) == {"here", "below"}
    assert result["workspace"] == str(proj)


def test_sibling_folder_with_shared_prefix_is_excluded(db, work):
    _seed(db, "foo", work / "foo")
    _seed(db, "foobar", work / "foobar")
    result = _call("session.list", {"workspace_cwd": str(work / "foo")})["result"]
    assert _ids(result) == {"foo"}


def test_home_folder_lists_all(db, work, monkeypatch):
    home = work / "home"
    monkeypatch.setenv("HOME", str(home))
    _seed(db, "in_home", home / "proj")
    _seed(db, "outside", work / "other")
    result = _call("session.list", {"workspace_cwd": str(home)})["result"]
    assert _ids(result) == {"in_home", "outside"}
    assert result["workspace"] is None


def test_relative_path_lists_all(db, work):
    _seed(db, "a", work / "proj")
    _seed(db, "b", work / "other")
    result = _call("session.list", {"workspace_cwd": "work/proj"})["result"]
    assert _ids(result) == {"a", "b"}
    assert result["workspace"] is None


def test_compressed_conversation_is_judged_by_root_cwd(db, work):
    """``root1`` (cwd=proj, compression-ended) -> ``tip1`` (cwd elsewhere): listed under proj."""
    proj, other = work / "proj", work / "other"
    t0 = time.time() - 3600
    _seed(db, "root1", proj)
    db._conn.execute(
        "UPDATE sessions SET started_at=?, ended_at=?, end_reason=? WHERE id=?",
        (t0, t0 + 1800, "compression", "root1"),
    )
    db.create_session("tip1", "cli", parent_session_id="root1")
    db.append_message("tip1", "user", "continuing")
    db._conn.execute("UPDATE sessions SET started_at=?, cwd=? WHERE id=?", (t0 + 1801, str(other), "tip1"))
    db._conn.commit()
    assert "tip1" in _ids(_call("session.list", {"workspace_cwd": str(proj)})["result"])
    assert "tip1" not in _ids(_call("session.list", {"workspace_cwd": str(other)})["result"])


def test_git_subfolder_scopes_to_repo_root(db, work, monkeypatch):
    repo = work / "repo"
    monkeypatch.setattr(
        git_probe, "common_repo_root",
        lambda cwd: str(repo) if cwd == str(repo) or cwd.startswith(f"{repo}/") else "",
    )
    _seed(db, "sub_a", repo / "a")
    _seed(db, "sub_b", repo / "b" / "c")
    _seed(db, "outside", work / "other")
    result = _call("session.list", {"workspace_cwd": str(repo / "a")})["result"]
    assert _ids(result) == {"sub_a", "sub_b"}
    assert result["workspace"] == str(repo)
