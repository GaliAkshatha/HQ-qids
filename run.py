#!/usr/bin/env python3
"""
run.py

Single-command startup for the complete system: Redis -> distributed
workers (Phase 7, for architectural completeness) -> API backend
-> frontend.

    python run.py                  # everything
    python run.py --backend-only   # Redis + workers + API, no frontend
    python run.py --frontend-only  # frontend dev server only
    python run.py --workers-only   # Redis + the 6 distributed workers only

Redis: if a local `redis-server` binary is found on PATH, it is started
as a child process. If not found, this prints a clear message and exits
rather than doing anything destructive.
"""

from __future__ import annotations

import argparse
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent

load_dotenv(REPO_ROOT / ".env")

WORKER_MODULES = [
    "src.runtime.services.detection_worker",
    "src.runtime.services.quantum_worker",
    "src.runtime.services.risk_worker",
    "src.runtime.services.defense_worker",
    "src.runtime.services.incident_worker",
]

API_PORT = 8080
FRONTEND_PORT = 5173

_processes: list = []


def _npm_executable() -> str:
    """
    On Windows, npm is installed as npm.cmd (a shell shim), not a
    directly-executable PE binary -- subprocess.Popen(["npm", ...])
    fails with WinError 2 (file not found) even though `npm --version`
    works fine interactively in PowerShell/cmd, because the shell
    resolves .cmd/.bat shims via PATHEXT but CreateProcess does not.
    shutil.which("npm") correctly resolves to npm.cmd on Windows (it
    also checks PATHEXT) and to the plain "npm" binary on Linux/macOS,
    so this is a single, portable fix -- no OS-specific branching needed
    beyond letting shutil.which do its job. Falls back to the literal
    "npm" if which() can't find it, preserving the original behavior
    (and original error message) rather than silently doing nothing.
    """
    return shutil.which("npm") or "npm"


def _spawn(cmd, name, cwd=None):
    print(f"[run.py] starting {name}: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=str(cwd or REPO_ROOT))
    _processes.append(proc)
    return proc


def check_redis_binary() -> bool:
    return shutil.which("redis-server") is not None


def check_redis_running() -> bool:
    """
    Check the configured Redis instance.

    If REDIS_URL is set, use it. Otherwise fall back to the
    existing REDIS_HOST/REDIS_PORT configuration (localhost by default).
    """
    try:
        import os
        import redis

        redis_url = os.getenv("REDIS_URL", "").strip()

        if redis_url:
            client = redis.from_url(
                redis_url,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
        else:
            host = os.getenv("REDIS_HOST", "localhost")
            port = int(os.getenv("REDIS_PORT", "6379"))

            client = redis.Redis(
                host=host,
                port=port,
                socket_connect_timeout=2,
                socket_timeout=2,
            )

        return bool(client.ping())

    except Exception:
        return False


def start_redis() -> bool:
    """
    Reuse an already-configured Redis instance when available.

    If REDIS_URL is configured and reachable, no local Redis process
    is started.

    If REDIS_URL is not configured, preserve the existing local
    redis-server fallback behavior.
    """
    import os

    redis_url = os.getenv("REDIS_URL", "").strip()

    # Managed/external Redis
    if redis_url:
        if check_redis_running():
            print("[run.py] Configured Redis is reachable -- reusing it.")
            return True

        print(
            "[run.py] ERROR: REDIS_URL is configured, but the configured "
            "Redis instance is not reachable."
        )
        return False

    # Existing local Redis behavior
    if check_redis_running():
        print("[run.py] Redis already running -- reusing it.")
        return True

    if not check_redis_binary():
        print(
            "[run.py] ERROR: redis-server not found on PATH and no Redis instance "
            "is reachable at localhost:6379. Install Redis or start it yourself, "
            "then re-run this script."
        )
        return False

    redis_path = shutil.which("redis-server")
    _spawn(
        [redis_path, "--daemonize", "no", "--port", "6379"],
        "redis",
    )

    for _ in range(20):
        if check_redis_running():
            print("[run.py] Redis is up.")
            return True
        time.sleep(0.5)

    print(
        "[run.py] ERROR: redis-server started but did not become "
        "reachable in time."
    )
    return False


def start_workers() -> None:
    for module in WORKER_MODULES:
        _spawn([sys.executable, "-m", module], module.rsplit(".", 1)[-1])


def start_backend() -> None:
    _spawn([sys.executable, "-m", "src.api.app"], "api-backend")


def start_frontend() -> None:
    frontend_dir = REPO_ROOT / "frontend"
    if not (frontend_dir / "node_modules").exists():
        print("[run.py] frontend/node_modules not found -- run `npm install` in frontend/ first.")
        return
    _spawn([_npm_executable(), "run", "dev", "--", "--port", str(FRONTEND_PORT)], "frontend", cwd=frontend_dir)


def check_dependencies() -> bool:
    try:
        import flask  # noqa: F401
        import redis  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError as e:
        print(f"[run.py] ERROR: missing Python dependency: {e}. Run `pip install -r requirements.txt`.")
        return False
    return True


def shutdown(*_args):
    print("\n[run.py] shutting down child processes...")
    for proc in _processes:
        if proc.poll() is None:
            proc.terminate()
    for proc in _processes:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("[run.py] done.")
    sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-command startup for the Quantum-Assisted IDS platform")
    parser.add_argument("--backend-only", action="store_true")
    parser.add_argument("--frontend-only", action="store_true")
    parser.add_argument("--workers-only", action="store_true")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    if args.frontend_only:
        start_frontend()
        print(f"[run.py] frontend: http://localhost:{FRONTEND_PORT}")
    else:
        if not check_dependencies():
            sys.exit(1)
        if not start_redis():
            sys.exit(1)

        if args.workers_only:
            start_workers()
        elif args.backend_only:
            start_workers()
            start_backend()
            time.sleep(2)
            print(f"[run.py] API: http://localhost:{API_PORT}/api/health")
        else:
            start_workers()
            start_backend()
            time.sleep(2)
            start_frontend()
            print(f"[run.py] API:      http://localhost:{API_PORT}/api/health")
            print(f"[run.py] Frontend: http://localhost:{FRONTEND_PORT}")

    print("[run.py] press Ctrl+C to stop.")
    while True:
        time.sleep(1)
        for proc in _processes:
            if proc.poll() is not None:
                print(f"[run.py] WARNING: a child process exited unexpectedly (code {proc.returncode}).")


if __name__ == "__main__":
    main()
