"""Resumed dashboard chat + ``?cwd=`` (FB35-H): the tab's folder reaches the TUI as ``HERMES_TUI_CWD`` only.

A resumed session keeps its stored cwd (no ``HERMES_CWD``); sessions started inside the tab land in the folder.
A dead/relative ``cwd`` on a resume is ignored instead of failing the resume.
"""

import pytest

from hermes_cli import web_server_chat
from hermes_cli.web_routers import chat_ws


@pytest.fixture
def fake_tui(monkeypatch, tmp_path):
    import hermes_cli.main_tui_launch as tui_launch

    monkeypatch.setattr(
        tui_launch, "_make_tui_argv", lambda *_a, **_k: (["node", "fake-tui.js"], tmp_path))
    monkeypatch.setattr(tui_launch, "_apply_tui_python_env", lambda _env: None)
    monkeypatch.delenv("HERMES_CWD", raising=False)
    monkeypatch.delenv("HERMES_TUI_CWD", raising=False)


def test_startup_cwd_sets_tui_cwd_only(fake_tui, tmp_path):
    _argv, _cwd, env = web_server_chat._resolve_chat_argv(startup_cwd=str(tmp_path))
    assert env["HERMES_TUI_CWD"] == str(tmp_path)
    assert "HERMES_CWD" not in env


def test_no_startup_or_workspace_cwd_is_unchanged(fake_tui):
    _argv, _cwd, env = web_server_chat._resolve_chat_argv()
    assert "HERMES_TUI_CWD" not in env
    assert "HERMES_CWD" not in env


def test_resume_startup_cwd_ignores_bad_paths(tmp_path):
    assert chat_ws._resume_startup_cwd(str(tmp_path)) == str(tmp_path)
    assert chat_ws._resume_startup_cwd(str(tmp_path / "gone")) is None
    assert chat_ws._resume_startup_cwd("relative/dir") is None
    assert chat_ws._resume_startup_cwd(None) is None
