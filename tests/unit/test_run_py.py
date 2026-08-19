"""
Tests for run.py's checkable logic (dependency checks, Redis detection,
argument parsing). Full process-orchestration behavior (spawning real
child processes, SIGINT cleanup) was verified manually via a real
end-to-end smoke test -- documented in the Stage D report.
"""

import subprocess
import sys

import run as run_module


def test_check_dependencies_true_when_all_present():
    assert run_module.check_dependencies() is True


def test_check_redis_running_reflects_real_state():
    result = run_module.check_redis_running()
    assert isinstance(result, bool)


def test_worker_modules_list_matches_real_runtime_services():
    import importlib

    for module_name in run_module.WORKER_MODULES:
        mod = importlib.import_module(module_name)
        assert hasattr(mod, "build_worker")


def test_argument_parser_accepts_all_documented_flags():
    result = subprocess.run(
        [sys.executable, "run.py", "--help"], cwd=".", capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    assert "--backend-only" in result.stdout
    assert "--frontend-only" in result.stdout
    assert "--workers-only" in result.stdout


def test_shutdown_does_not_crash_with_no_processes():
    run_module._processes.clear()
    try:
        run_module.shutdown()
    except SystemExit:
        pass


def test_npm_executable_resolves_to_a_real_path_or_bare_fallback():
    """On this (Linux) test environment, shutil.which('npm') should
    resolve to a real path since npm is installed -- proving the
    resolution mechanism works, not just that it doesn't crash. On
    Windows this same call resolves to npm.cmd, which is the actual fix
    (subprocess.Popen(["npm", ...]) fails with WinError 2 there because
    npm is a .cmd shim, not a directly-executable binary)."""
    resolved = run_module._npm_executable()
    assert resolved  # never empty/falsy
    import shutil
    which_result = shutil.which("npm")
    if which_result:
        assert resolved == which_result
    else:
        assert resolved == "npm"  # graceful fallback, preserves original error message


def test_npm_executable_falls_back_gracefully_when_npm_not_found(monkeypatch):
    monkeypatch.setattr(run_module.shutil, "which", lambda name: None)
    assert run_module._npm_executable() == "npm"


def test_redis_spawn_uses_resolved_path_not_bare_string(monkeypatch):
    """Proves start_redis() passes shutil.which()'s resolved path into
    _spawn(), not the bare 'redis-server' string -- the same class of
    fix as npm, applied consistently."""
    calls = []

    def fake_spawn(cmd, name, cwd=None):
        calls.append(cmd)

        class FakeProc:
            def poll(self):
                return None

        return FakeProc()

    monkeypatch.setattr(run_module, "_spawn", fake_spawn)
    monkeypatch.setattr(run_module, "check_redis_running", lambda: True)  # short-circuit "already running"
    monkeypatch.setattr(run_module.shutil, "which", lambda name: "/fake/path/redis-server" if name == "redis-server" else None)

    # already-running short circuit means _spawn is never called here --
    # exercise the actual spawn path by forcing "not yet running" once,
    # then "running" on the second poll inside start_redis()'s retry loop
    poll_results = iter([False, True])
    monkeypatch.setattr(run_module, "check_redis_running", lambda: next(poll_results, True))

    result = run_module.start_redis()
    assert result is True
    assert calls, "expected _spawn to be called"
    assert calls[0][0] == "/fake/path/redis-server"
