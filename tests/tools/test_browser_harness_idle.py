"""Idle stop for browser_exec harness daemons (tools/browser_harness_idle.py).

A long-lived process (the gateway) only stopped its harness daemons at exit, and a crash skipped even
that, so every distinct ``BU_NAME`` left a ``browser_harness.daemon`` behind for good. Each use is now
recorded under HERMES_HOME, and any live Hermes process's cleanup thread stops names idle past the
timeout.
"""
import json
import os
import stat
import time

import pytest

import tools.browser_use_cli as bu_cli
from tools import browser_harness_idle as idle
from tools import browser_tool
from tools import browser_tool_lifecycle as lifecycle


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.delenv("BU_NAME", raising=False)
    monkeypatch.delenv("BU_AUTOSPAWN", raising=False)
    monkeypatch.delenv("HERMES_BROWSER_HARNESS_IDLE_TIMEOUT", raising=False)
    monkeypatch.setattr("hermes_cli.config.read_raw_config", dict)
    # Never let browser_exec start the real janitor: its startup reap scans the host tmp dir.
    monkeypatch.setattr(lifecycle, "_start_browser_cleanup_thread", lambda: None)


def _fake_cli(tmp_path, body):
    script = tmp_path / "browser-use"
    script.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script)


def _record(name, last_used, owner_pid=None):
    idle.activity_dir().mkdir(parents=True, exist_ok=True)
    path = idle.activity_dir() / f"{name}.json"
    path.write_text(json.dumps({"name": name, "owner_pid": owner_pid or os.getpid(), "last_used": last_used}),
                    encoding="utf-8")
    return path


class TestTouch:
    def test_touch_records_last_use_and_owner(self):
        before = time.time()
        idle.touch("iwt-desk-v2")
        data = json.loads((idle.activity_dir() / "iwt-desk-v2.json").read_text(encoding="utf-8"))
        assert data["name"] == "iwt-desk-v2"
        assert data["owner_pid"] == os.getpid()
        assert before <= data["last_used"] <= time.time()

    def test_touch_accepts_sidecar_names(self):
        idle.touch("_local_research")
        assert (idle.activity_dir() / "_local_research.json").exists()

    @pytest.mark.parametrize("name", ["", "../escape", "a/b", "has space", "x" * 65, "-lead"])
    def test_invalid_name_writes_nothing(self, name):
        idle.touch(name)
        assert not idle.activity_dir().exists() or not any(idle.activity_dir().iterdir())


class TestReapIdle:
    def test_only_names_past_the_timeout_are_stopped(self):
        now = time.time()
        stale = _record("stale", now - 1801)
        fresh = _record("fresh", now - 60)
        stopped = []

        assert idle.reap_idle(stopped.append) == ["stale"]
        assert stopped == ["stale"]
        assert not stale.exists()
        assert fresh.exists()

    def test_record_of_a_dead_owner_is_reaped_by_this_process(self):
        stale = _record("orphan", time.time() - 3600, owner_pid=2**22 + 12345)
        stopped = []
        idle.reap_idle(stopped.append)
        assert stopped == ["orphan"]
        assert not stale.exists()

    def test_stop_failure_does_not_block_the_rest(self):
        now = time.time()
        first, second = _record("first", now - 4000), _record("second", now - 4000)
        stopped = []

        def stop(name):
            stopped.append(name)
            if name == "first":
                raise RuntimeError("harness gone")

        assert idle.reap_idle(stop) == ["second"]
        assert sorted(stopped) == ["first", "second"]
        assert not first.exists() and not second.exists()

    def test_corrupt_record_is_removed_without_a_stop(self):
        idle.activity_dir().mkdir(parents=True, exist_ok=True)
        corrupt = idle.activity_dir() / "broken.json"
        corrupt.write_text("{not json", encoding="utf-8")
        stopped = []

        assert idle.reap_idle(stopped.append) == []
        assert stopped == []
        assert not corrupt.exists()

    def test_zero_timeout_disables_the_reaper(self, monkeypatch):
        monkeypatch.setenv("HERMES_BROWSER_HARNESS_IDLE_TIMEOUT", "0")
        stale = _record("stale", time.time() - 10**6)
        stopped = []
        assert idle.reap_idle(stopped.append) == []
        assert stopped == []
        assert stale.exists()

    def test_missing_directory_is_a_no_op(self):
        assert idle.reap_idle(lambda name: pytest.fail("nothing to stop")) == []


class TestIdleTimeout:
    def test_default(self):
        assert idle.idle_timeout_s() == 1800

    def test_config_overrides_default(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.config.read_raw_config",
                            lambda: {"browser": {"harness_idle_timeout": 120}})
        assert idle.idle_timeout_s() == 120

    def test_env_overrides_config(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.config.read_raw_config",
                            lambda: {"browser": {"harness_idle_timeout": 120}})
        monkeypatch.setenv("HERMES_BROWSER_HARNESS_IDLE_TIMEOUT", "45")
        assert idle.idle_timeout_s() == 45

    def test_config_timeout_applies_to_reap(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.config.read_raw_config",
                            lambda: {"browser": {"harness_idle_timeout": 120}})
        _record("two-minutes", time.time() - 300)
        assert idle.reap_idle(lambda name: None) == ["two-minutes"]


class TestBrowserExecWiring:
    @pytest.fixture
    def exec_env(self, tmp_path, monkeypatch):
        from tools import browser_supervisor

        class _Registry:
            def get_or_start(self, task_id, cdp_url, **kw):
                pass

        monkeypatch.setattr(browser_supervisor, "SUPERVISOR_REGISTRY", _Registry())
        monkeypatch.setattr("tools.browser_tool_cdp._resolve_cdp_override", lambda url: url)
        monkeypatch.setenv("BROWSER_CDP_URL", "http://127.0.0.1:9400")
        monkeypatch.setattr(bu_cli, "_driven_daemons", set())
        return tmp_path

    def test_dispatch_records_use_before_and_after_the_call(self, exec_env, monkeypatch):
        ran = exec_env / "cli-ran"
        cli = _fake_cli(exec_env, f'cat > /dev/null\ntouch "{ran}"\necho ok\n')
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: [cli])
        real_touch = idle.touch
        touches = []

        def spy(name):
            touches.append((name, ran.exists()))
            real_touch(name)

        monkeypatch.setattr(idle, "touch", spy)

        result = json.loads(bu_cli.browser_exec("print(1)", session="idle-probe", task_id="t-idle"))

        assert result["success"] is True
        assert touches == [("idle-probe", False), ("idle-probe", True)]
        data = json.loads((idle.activity_dir() / "idle-probe.json").read_text(encoding="utf-8"))
        assert data["owner_pid"] == os.getpid()

    def test_dispatch_starts_the_cleanup_thread(self, exec_env, monkeypatch):
        cli = _fake_cli(exec_env, 'cat > /dev/null\necho ok\n')
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: [cli])
        started = []
        monkeypatch.setattr(lifecycle, "_start_browser_cleanup_thread", lambda: started.append(True))

        bu_cli.browser_exec("print(1)", task_id="t-idle")

        assert started

    def test_stop_harness_daemon_reloads_only_that_name(self, tmp_path, monkeypatch):
        log = tmp_path / "calls"
        cli = _fake_cli(tmp_path, f'echo "$1 $BU_NAME" >> "{log}"\n')
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: [cli])
        monkeypatch.setattr(bu_cli, "_driven_daemons", {"a", "b"})

        bu_cli.stop_harness_daemon("a")

        assert log.read_text(encoding="utf-8").splitlines() == ["--reload a"]
        assert bu_cli._driven_daemons == {"b"}


class TestCleanupThreadReaps:
    def test_orphan_cycle_reaps_idle_harness_daemons(self, monkeypatch):
        monkeypatch.setattr(lifecycle, "_reap_orphaned_browser_sessions", lambda: None)
        # One cycle only: the inactivity sweep runs after the reap cycle and ends the loop.
        monkeypatch.setattr(lifecycle, "_cleanup_inactive_browser_sessions",
                            lambda: setattr(browser_tool, "_cleanup_running", False))
        monkeypatch.setattr(browser_tool, "_cleanup_running", True)
        calls = []
        monkeypatch.setattr(idle, "reap_idle", lambda stop: calls.append(stop) or [])

        lifecycle._browser_cleanup_thread_worker()

        assert calls == [bu_cli.stop_harness_daemon]
