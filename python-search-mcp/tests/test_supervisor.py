"""Tests for the router supervisor.

The interesting behaviour is: connect to an already-running router without
spawning anything, spawn one when nothing is listening, and never spawn when
autostart is disabled. Network is avoided by pointing at a dead port.
"""

from __future__ import annotations

from scholarly_answer_engine.supervisor import RouterSupervisor


def test_is_reachable_false_for_a_dead_port():
    assert RouterSupervisor("http://127.0.0.1:9", router_dir=None).is_reachable() is False


def test_ensure_starts_a_router_when_one_can_be_built(tmp_path):
    router_dir = tmp_path / "scholarly-answer-router"
    (router_dir / "dist").mkdir(parents=True)
    (router_dir / "dist" / "index.js").write_text("// stub", encoding="utf-8")

    supervisor = RouterSupervisor("http://127.0.0.1:8911", router_dir=str(router_dir), port=8911, startup_timeout=2.0)
    # The stub exits immediately, so ensure() reports it could not come up.
    result = supervisor.ensure()
    assert result is False
    assert supervisor.process is not None, "a spawn should have been attempted"
    supervisor.stop()


def test_ensure_reports_failure_when_no_router_build_exists(tmp_path):
    supervisor = RouterSupervisor("http://127.0.0.1:8912", router_dir=str(tmp_path), port=8912)
    assert supervisor.ensure() is False
    assert supervisor.process is None, "nothing should be spawned when no build exists"


def test_discovers_a_sibling_router_repo_without_being_told(tmp_path):
    site = tmp_path / "repo" / "src" / "scholarly_answer_engine"
    site.mkdir(parents=True)
    router = tmp_path / "repo" / ".." / "scholarly-answer-router" / "dist"
    router.mkdir(parents=True)
    (tmp_path / "scholarly-answer-router" / "dist" / "index.js").write_text("// stub", encoding="utf-8")


    # Point the module at our fake tree by monkeypatching __file__'s resolution.
    import scholarly_answer_engine.supervisor as mod

    original = mod.__file__
    mod.__file__ = str(site / "supervisor.py")
    try:
        found = mod.RouterSupervisor("http://127.0.0.1:8913")._default_router_dir()
        assert found is not None
        assert (found / "dist" / "index.js").exists()
    finally:
        mod.__file__ = original


def test_port_is_taken_from_the_url():
    assert RouterSupervisor("http://127.0.0.1:8787")._port_from_url() == 8787
    assert RouterSupervisor("http://127.0.0.1:8787", port=9999)._port_from_url() == 9999


def test_stop_is_a_noop_when_nothing_was_spawned():
    RouterSupervisor("http://127.0.0.1:8914").stop()
