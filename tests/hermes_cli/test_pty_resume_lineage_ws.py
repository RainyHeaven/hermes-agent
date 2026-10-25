import pytest

from hermes_cli import web_server
import hermes_cli.web_routers.chat_ws as _chat_ws
import hermes_cli.web_server_chat as _web_server_chat


class FakeBridge:
    def __init__(self):
        self.alive = True
        self.accept_input = True
        self.written = bytearray()

    def read(self, timeout):
        return b""        # idle forever

    async def write(self, data):
        if not self.accept_input:
            return False
        self.written.extend(data)
        return True

    def resize(self, cols, rows):
        pass

    def is_alive(self):
        return self.alive

    def close(self):
        self.alive = False


@pytest.fixture
def lineage_harness(monkeypatch):
    class Spawned(list):
        pass

    spawned = Spawned()
    spawned.bridges = []
    spawned.lineage = {"A": "A", "C": "C"}

    def fake_spawn(argv, cwd=None, env=None):
        b = FakeBridge()
        spawned.append(argv)
        spawned.bridges.append(b)
        return b

    monkeypatch.setattr(_web_server_chat.PtyBridge, "spawn", staticmethod(fake_spawn))
    monkeypatch.setattr(_web_server_chat, "_ws_auth_reason", lambda ws: (None, "test"))
    monkeypatch.setattr(_web_server_chat, "_ws_host_origin_reason", lambda ws: None)
    monkeypatch.setattr(_web_server_chat, "_ws_client_reason", lambda ws: None)

    async def fake_argv(**kw):
        raw = kw.get("resume")
        resume = spawned.lineage.get(raw, raw) if raw else None
        env = {"HERMES_TUI_RESUME": resume} if resume else {}
        return (["x", resume or "fresh"], "/tmp", env)

    monkeypatch.setattr(_web_server_chat, "_resolve_chat_argv_async", fake_argv)
    # raising=False: the seam is absent before the fix, and cases 3-5 must still run there.
    monkeypatch.setattr(
        _chat_ws, "_latest_descendant_for",
        lambda resume, profile: spawned.lineage.get(resume, resume), raising=False,
    )

    try:
        yield spawned
    finally:
        _web_server_chat.PTY_REGISTRY._sessions.clear()


@pytest.mark.asyncio
async def test_rollover_reattach_keeps_the_live_terminal(lineage_harness):
    from starlette.testclient import TestClient

    client = TestClient(web_server.app)
    with client.websocket_connect("/api/pty?attach=TOK&resume=A") as ws1:
        ws1.send_bytes(b"hi")
    lineage_harness.lineage["A"] = "B"
    with client.websocket_connect("/api/pty?attach=TOK&resume=A") as ws2:
        ws2.send_bytes(b"again")

    assert len(lineage_harness) == 1
    bridge = lineage_harness.bridges[0]
    assert bridge.alive is True
    assert bytes(bridge.written) == b"hi\x0cagain"
    assert list(web_server.PTY_REGISTRY._sessions) == ["TOK\0\0A"]


@pytest.mark.asyncio
async def test_rollover_reattach_while_attached_supersedes_socket_not_pty(lineage_harness):
    from starlette.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    client = TestClient(web_server.app)
    with client.websocket_connect("/api/pty?attach=TOK&resume=A") as ws1:
        ws1.send_bytes(b"hi")
        lineage_harness.lineage["A"] = "B"
        with client.websocket_connect("/api/pty?attach=TOK&resume=A") as ws2:
            with pytest.raises(WebSocketDisconnect) as exc_info:
                ws1.receive_bytes()
            assert exc_info.value.code == 4409
            ws2.send_bytes(b"again")

    assert len(lineage_harness) == 1
    bridge = lineage_harness.bridges[0]
    assert bridge.alive is True
    assert b"again" in bytes(bridge.written)


@pytest.mark.asyncio
async def test_unrelated_lineage_spawns_fresh_terminal(lineage_harness):
    from starlette.testclient import TestClient

    client = TestClient(web_server.app)
    with client.websocket_connect("/api/pty?attach=TOK&resume=A") as ws1:
        ws1.send_bytes(b"hi")
    with client.websocket_connect("/api/pty?attach=TOK&resume=C") as ws2:
        ws2.send_bytes(b"again")

    assert lineage_harness == [["x", "A"], ["x", "C"]]
    assert lineage_harness.bridges[0].alive is False
    assert lineage_harness.bridges[1].alive is True
    assert bytes(lineage_harness.bridges[1].written) == b"again"


@pytest.mark.asyncio
async def test_lineage_lookup_failure_falls_back_to_fresh_spawn(lineage_harness, monkeypatch):
    from starlette.testclient import TestClient

    def broken_lookup(resume, profile):
        raise RuntimeError("session store unavailable")

    monkeypatch.setattr(_chat_ws, "_latest_descendant_for", broken_lookup, raising=False)
    client = TestClient(web_server.app)
    with client.websocket_connect("/api/pty?attach=TOK&resume=A") as ws1:
        ws1.send_bytes(b"hi")
    lineage_harness.lineage["A"] = "B"
    with client.websocket_connect("/api/pty?attach=TOK&resume=A") as ws2:
        ws2.send_bytes(b"again")

    assert len(lineage_harness) == 2
    assert bytes(lineage_harness.bridges[1].written) == b"again"


@pytest.mark.asyncio
async def test_lineage_match_is_scoped_to_profile(lineage_harness):
    from starlette.testclient import TestClient

    client = TestClient(web_server.app)
    with client.websocket_connect("/api/pty?attach=TOK&resume=A&profile=p1") as ws1:
        ws1.send_bytes(b"hi")
    lineage_harness.lineage["A"] = "B"
    with client.websocket_connect("/api/pty?attach=TOK&resume=A&profile=p2") as ws2:
        ws2.send_bytes(b"again")

    assert len(lineage_harness) == 2
    assert bytes(lineage_harness.bridges[1].written) == b"again"
    assert list(web_server.PTY_REGISTRY._sessions) == ["TOK\0p2\0B"]


@pytest.mark.asyncio
async def test_explicit_successor_resume_after_rollover_keeps_the_live_terminal(lineage_harness):
    from starlette.testclient import TestClient

    client = TestClient(web_server.app)
    with client.websocket_connect("/api/pty?attach=TOK&resume=A") as ws1:
        ws1.send_bytes(b"hi")
    lineage_harness.lineage["A"] = "B"
    lineage_harness.lineage["B"] = "B"
    with client.websocket_connect("/api/pty?attach=TOK&resume=B") as ws2:
        ws2.send_bytes(b"again")

    assert len(lineage_harness) == 1
    bridge = lineage_harness.bridges[0]
    assert bridge.alive is True
    assert bytes(bridge.written) == b"hi\x0cagain"
    assert list(web_server.PTY_REGISTRY._sessions) == ["TOK\0\0A"]


@pytest.mark.asyncio
async def test_multi_step_lineage_reattach_keeps_the_live_terminal(lineage_harness):
    from starlette.testclient import TestClient

    client = TestClient(web_server.app)
    with client.websocket_connect("/api/pty?attach=TOK&resume=A") as ws1:
        ws1.send_bytes(b"hi")
    # A -> B -> C inside the live terminal; the lookup reports the newest leaf.
    lineage_harness.lineage["A"] = "C"
    lineage_harness.lineage["B"] = "C"
    with client.websocket_connect("/api/pty?attach=TOK&resume=A") as ws2:
        ws2.send_bytes(b"again")

    assert len(lineage_harness) == 1
    bridge = lineage_harness.bridges[0]
    assert bridge.alive is True
    assert bytes(bridge.written) == b"hi\x0cagain"
    assert list(web_server.PTY_REGISTRY._sessions) == ["TOK\0\0A"]


def _real_store_ids(*names):
    import uuid

    # Unique ids: the isolated HERMES_HOME may be shared across tests in one run.
    tag = uuid.uuid4().hex[:8]
    return [f"fb20-{tag}-{n}" for n in names]


def _assert_isolated_home():
    from hermes_constants import get_hermes_home

    home = str(get_hermes_home())
    assert "/.hermes" not in home or "pytest" in home or "tmp" in home.lower(), home


def test_latest_descendant_for_reads_real_default_store():
    from hermes_state import SessionDB

    _assert_isolated_home()
    a, b, missing = _real_store_ids("A", "B", "missing")
    db = SessionDB()
    try:
        db.create_session(a, source="tui")
        db.create_session(b, source="tui", parent_session_id=a)
    finally:
        db.close()

    for profile in (None, "", "current"):
        assert _chat_ws._latest_descendant_for(a, profile) == b, profile
        assert _chat_ws._latest_descendant_for(b, profile) == b, profile
        assert _chat_ws._latest_descendant_for(missing, profile) is None, profile


def test_latest_descendant_for_follows_multi_step_lineage_in_real_store():
    from hermes_state import SessionDB

    _assert_isolated_home()
    a, b, c = _real_store_ids("A", "B", "C")
    db = SessionDB()
    try:
        db.create_session(a, source="tui")
        db.create_session(b, source="tui", parent_session_id=a)
        db.create_session(c, source="tui", parent_session_id=b)
    finally:
        db.close()

    for profile in (None, "", "current"):
        assert _chat_ws._latest_descendant_for(a, profile) == c, profile
        assert _chat_ws._latest_descendant_for(b, profile) == c, profile
