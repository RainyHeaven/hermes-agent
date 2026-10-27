"""FB35-H3: the FB34/35 keys pass the wire contract on the real dispatch path.

Params models are extra="forbid"; an undeclared key is refused with 4000 by ``validate_params``
before the handler runs, which calling ``_methods[...]`` directly never exercised.
"""

import pytest

import tui_gateway.server as srv
from hermes_state import SessionDB
from tui_gateway import git_probe
from tui_gateway.contracts.registry import METHODS


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
    monkeypatch.setattr(git_probe, "common_repo_root", lambda cwd: "")


def _request(method: str, params: dict) -> dict:
    resp = srv.handle_request({"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
    assert resp is not None
    return resp


def _not_refused(resp: dict) -> None:
    assert (resp.get("error") or {}).get("code") != 4000, resp


def test_session_list_workspace_cwd_passes_dispatch(db, tmp_path):
    resp = _request("session.list", {"workspace_cwd": str(tmp_path.resolve())})
    _not_refused(resp)
    assert "workspace" in resp["result"]


def test_session_active_list_workspace_cwd_passes_dispatch(db, tmp_path):
    _not_refused(_request("session.active_list", {"workspace_cwd": str(tmp_path.resolve())}))


def test_session_resume_cwd_passes_dispatch(db, tmp_path):
    _not_refused(_request("session.resume", {"session_id": "no-such-session", "cwd": str(tmp_path.resolve())}))


def test_contract_models_accept_new_keys():
    METHODS["session.list"].params.model_validate({"workspace_cwd": "/p"})
    METHODS["session.list"].result.model_validate({"sessions": [], "workspace": "/p"})
    METHODS["session.list"].result.model_validate({"sessions": [], "workspace": None})
    METHODS["session.active_list"].params.model_validate({"workspace_cwd": "/p"})
    METHODS["session.active_list"].result.model_validate({"sessions": [], "workspace": "/p"})
    METHODS["session.resume"].params.model_validate({"session_id": "x", "cwd": "/p"})
