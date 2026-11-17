"""Local background launcher used by start.sh (macOS/Linux)."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

from dotenv import dotenv_values
from redis import Redis
from redis.exceptions import RedisError

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"


def environment(root: Path) -> dict[str, str]:
    if not (root / ".env").is_file():
        raise RuntimeError("Missing root .env; copy .env.example and configure your keys and Supabase DATABASE_URL.")
    env = {**os.environ, **{k: v for k, v in dotenv_values(root / ".env").items() if v is not None}}
    # Login/startup launchers often omit Homebrew and the user's Node/pnpm paths.
    paths = ["/opt/homebrew/opt/ffmpeg-full/bin", "/usr/local/opt/ffmpeg-full/bin",
             str(Path.home() / "Library/pnpm"), "/opt/homebrew/bin", "/usr/local/bin"]
    env["PATH"] = os.pathsep.join([*paths, env.get("PATH", os.defpath)])
    env.setdefault("FRONTEND_PORT", "3107")
    env.setdefault("BACKEND_PORT", "8000")
    for key in ("FRONTEND_PORT", "BACKEND_PORT", "REDIS_PORT"):
        value = env.get(key, "6379")
        if not value.isdigit() or not 1 <= int(value) <= 65535:
            raise RuntimeError(f"{key} must be a port between 1 and 65535.")
    env.setdefault("REDIS_HOST", "127.0.0.1")
    env.setdefault("REDIS_PORT", "6379")
    env.setdefault("NEXT_PUBLIC_API_URL", f"http://127.0.0.1:{env['BACKEND_PORT']}")
    env.setdefault("BACKEND_INTERNAL_URL", env["NEXT_PUBLIC_API_URL"])
    if not env.get("DATABASE_URL"):
        raise RuntimeError("Set DATABASE_URL in .env to your Supabase Postgres connection string.")
    return env


def process_identity(pid: int) -> str:
    result = subprocess.run(["ps", "-p", str(pid), "-o", "stat=,lstart=,command="], capture_output=True, text=True)
    output = result.stdout.strip()
    if result.returncode or not output or output.split()[0].startswith("Z"):
        return ""
    return " ".join(output.split()[1:])


def saved_process(name: str) -> int | None:
    try:
        saved = json.loads((LOGS / f"local-{name}.pid.json").read_text())
        if saved["identity"] and process_identity(saved["pid"]) == saved["identity"]:
            return saved["pid"]
    except (OSError, ValueError, KeyError):
        pass
    return None


def service_processes(name: str, port: int | None = None) -> list[tuple[int, bool]]:
    """Return PIDs and whether they belong to this checkout (never guess by port)."""
    managed = saved_process(name)
    if managed:
        return [(managed, True)]
    if port is not None:
        result = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"], capture_output=True, text=True)
        candidates = {int(pid) for pid in result.stdout.split() if pid.isdigit()}
    else:
        result = subprocess.check_output(["ps", "-axo", "pid=,command="], text=True)
        candidates = {int(line.split()[0]) for line in result.splitlines()
                      if "src.workers.tasks.WorkerSettings" in line and "arq" in line}
    expected = ROOT / ("frontend" if name == "frontend" else "backend")
    found = []
    for pid in sorted(candidates):
        identity = process_identity(pid)
        if not identity:
            continue
        result = subprocess.run(["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"], capture_output=True, text=True)
        cwd = next((line[1:] for line in result.stdout.splitlines() if line.startswith("n")), "")
        marker = {"api": "uvicorn src.main:app", "worker": "arq src.workers.tasks.WorkerSettings",
                  "frontend": "next", "redis": "redis-server"}[name]
        owned = bool(cwd and os.path.samefile(cwd, expected) and marker in identity)
        if port is not None or owned:
            found.append((pid, owned))
    return found


def remember_process(name: str, pid: int):
    identity = process_identity(pid)
    if not identity:
        raise RuntimeError(f"{name} exited before it could be recorded.")
    (LOGS / f"local-{name}.pid.json").write_text(json.dumps({"pid": pid, "identity": identity}))


def stop_process(name: str, pid: int):
    identity = process_identity(pid)
    if not identity:
        return
    # Only our recorded session leaders may take their child processes with them.
    group = saved_process(name) == pid and os.getpgid(pid) == pid
    print(f"Stopping {name} (PID {pid}) gracefully...", flush=True)
    try:
        if process_identity(pid) != identity:
            raise RuntimeError(f"PID {pid} changed; refusing to stop a different process.")
        (os.killpg if group else os.kill)(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process_identity(pid) != identity:
            (LOGS / f"local-{name}.pid.json").unlink(missing_ok=True)
            print(f"{name}: stopped", flush=True)
            return
        time.sleep(.25)
    if sys.stdin.isatty():
        answer = input(f"{name} is still stopping (it may be finishing a job). Type KILL {pid} to force-stop, or Enter to leave it running: ")
        if answer.strip() == f"KILL {pid}" and process_identity(pid) == identity:
            (os.killpg if group else os.kill)(pid, signal.SIGKILL)
            return
    raise RuntimeError(f"{name} (PID {pid}) is still stopping. No force-kill was sent.")


def show_status(env):
    ports = {"redis": int(env["REDIS_PORT"]), "api": int(env["BACKEND_PORT"]),
             "worker": None, "frontend": int(env["FRONTEND_PORT"])}
    print("\nSERVICE    PID       PORT    PROCESS")
    for name, port in ports.items():
        found = service_processes(name, port)
        for pid, owned in found or [(None, False)]:
            state = ("managed" if saved_process(name) == pid else "this checkout" if owned else "external/shared") if pid else "stopped"
            print(f"{name:<10} {str(pid or '-'):<9} {str(port or '-'):<7} {state}")
    print(f"\nLogs: {LOGS}/local-*.log\nStop a service: ./start.sh stop worker  (or api, frontend, redis, all)\n")


def http_ready(url: str) -> bool:
    try:
        with urlopen(url, timeout=2) as response:
            return response.status == 200
    except (OSError, URLError):
        return False


def port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def startup_error(name: str, log_offset: int) -> str:
    if name == "api":
        try:
            with (LOGS / "local-api.log").open("rb") as log:
                log.seek(log_offset)
                recent = log.read().lower()
            if b"tenant" in recent and b"not found" in recent:
                return (
                    "Supabase rejected DATABASE_URL: tenant/user not found. "
                    "Check the project in the Supabase dashboard: resume it if paused; "
                    "otherwise copy Connect > Session pooler into the root .env DATABASE_URL. "
                    "Then rerun ./start.sh."
                )
        except OSError:
            pass
    return f"{name} is not ready; see logs/local-{name}.log. Fix the error and rerun ./start.sh; other started services remain running."


def start(name, command, cwd, env, ready, port=None):
    log_path = LOGS / f"local-{name}.log"
    log_offset = log_path.stat().st_size if log_path.exists() else 0
    pid = saved_process(name)
    if pid:
        print(f"{name}: already running (PID {pid})", flush=True)
    else:
        found = service_processes(name, port)
        if found:
            print(f"{name}: found " + ", ".join(f"PID {p} ({'this checkout' if owned else 'external/shared'})" for p, owned in found), flush=True)
            if len(found) != 1 or not found[0][1]:
                raise RuntimeError(f"{name}: unmanaged conflict. Inspect ./start.sh status; no unrelated process was stopped.")
            pid = found[0][0]
            choice = input(f"{name} PID {pid}: [r] reuse, [s] stop and replace, [q] cancel [r]: ").strip().lower() if sys.stdin.isatty() else "r"
            if choice in {"", "r"}:
                remember_process(name, pid)
                print(f"{name}: reusing PID {pid}", flush=True)
            elif choice == "s":
                stop_process(name, pid)
                pid = None
            else:
                raise RuntimeError("Startup cancelled; existing processes were left running.")
        elif port and port_open(port):
            raise RuntimeError(f"Port {port} has an unmanaged listener that could not be identified.")
    if not pid:
        with (LOGS / f"local-{name}.log").open("ab") as log:
            process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        pid = process.pid
        # Wait for shell/CLI wrappers to exec before recording their process identity.
        time.sleep(1)
        identity = process_identity(pid)
        if process.poll() is not None or not identity:
            raise RuntimeError(startup_error(name, log_offset))
        remember_process(name, pid)
        print(f"{name}: started (PID {pid})", flush=True)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if not process_identity(pid):
            break
        if ready():
            return
        time.sleep(.5)
    raise RuntimeError(startup_error(name, log_offset))


def main():
    parser = argparse.ArgumentParser(description="Run and manage the local Clipper.io services.")
    parser.add_argument("action", nargs="?", default="start", choices=("start", "status", "stop"))
    parser.add_argument("service", nargs="?", choices=("api", "worker", "frontend", "redis", "all"))
    args = parser.parse_args()
    if args.action == "stop" and not args.service:
        parser.error("choose a service, for example: ./start.sh stop worker")
    if args.action != "stop" and args.service:
        parser.error("a service name is only supported with stop")
    env = environment(ROOT)
    show_status(env)
    if args.action == "status":
        return
    if args.action == "stop":
        ports = {"worker": None, "frontend": int(env["FRONTEND_PORT"]), "api": int(env["BACKEND_PORT"]), "redis": int(env["REDIS_PORT"])}
        for name in ports if args.service == "all" else [args.service]:
            found = service_processes(name, ports[name])
            for pid, owned in found:
                if owned:
                    stop_process(name, pid)
                else:
                    print(f"{name}: leaving external/shared PID {pid} running")
            if not found:
                print(f"{name}: already stopped")
        return
    backend = ROOT / "backend"
    python = str(backend / ".venv/bin/python")
    node = shutil.which("node", path=env["PATH"])
    ffmpeg = shutil.which("ffmpeg", path=env["PATH"])
    if not node or not ffmpeg or not shutil.which("ffprobe", path=env["PATH"]):
        raise RuntimeError("Install Node.js and FFmpeg with libass support (macOS: brew install node ffmpeg-full).")
    filters = subprocess.run([ffmpeg, "-hide_banner", "-filters"], capture_output=True, text=True, env=env, check=True)
    if " subtitles " not in filters.stdout:
        raise RuntimeError("FFmpeg lacks the subtitles filter. Install ffmpeg-full or a build with libass.")
    if not (ROOT / "frontend/node_modules/next").is_dir():
        raise RuntimeError("Install frontend dependencies: cd frontend && pnpm install --frozen-lockfile")
    redis = Redis(host=env["REDIS_HOST"], port=int(env["REDIS_PORT"]), password=env.get("REDIS_PASSWORD") or None,
                  socket_connect_timeout=1, socket_timeout=1)

    def redis_ready():
        try:
            return redis.ping()
        except RedisError:
            return False

    if not redis_ready():
        if env["REDIS_HOST"] not in {"localhost", "127.0.0.1"} or env.get("REDIS_PASSWORD"):
            raise RuntimeError("Configured Redis is unavailable; start it with its configured credentials first.")
        redis_server = shutil.which("redis-server", path=env["PATH"])
        if not redis_server:
            raise RuntimeError("Install Redis (macOS: brew install redis).")
        start("redis", [redis_server, "--bind", "127.0.0.1", "--port", env["REDIS_PORT"],
                        "--dir", str(LOGS)], ROOT, env, redis_ready, int(env["REDIS_PORT"]))
    else:
        print("redis: ready (reusing existing instance)", flush=True)

    api_url = f"http://127.0.0.1:{env['BACKEND_PORT']}"
    frontend_url = f"http://127.0.0.1:{env['FRONTEND_PORT']}"
    # API startup applies the repository's pending SQL migrations before health succeeds.
    start("api", [python, "-m", "uvicorn", "src.main:app", "--host", "127.0.0.1", "--port", env["BACKEND_PORT"]],
          backend, env, lambda: http_ready(api_url + "/health"), int(env["BACKEND_PORT"]))
    start("worker", [python, "-m", "arq", "src.workers.tasks.WorkerSettings"], backend, env,
          lambda: bool(redis.exists("supoclip_tasks:health-check")))
    # Run the installed Next CLI directly; startup must not trigger package installs.
    start("frontend", [node, str(ROOT / "frontend/node_modules/next/dist/bin/next"), "dev", "--turbopack", "--port", env["FRONTEND_PORT"], "--hostname", "127.0.0.1"],
          ROOT / "frontend", env, lambda: http_ready(frontend_url), int(env["FRONTEND_PORT"]))
    print(f"\nClipper.io ready: {frontend_url}\nAPI: {api_url}\nLogs: {LOGS}/local-*.log\n"
          "Services run in the background. Re-running start.sh reuses them; it does not restart active jobs.", flush=True)


if __name__ == "__main__":
    LOGS.mkdir(exist_ok=True)
    with (LOGS / "local-start.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            main()
        except (OSError, RuntimeError, RedisError, subprocess.SubprocessError) as error:
            # Provider/connection errors can contain credentials; keep those in service logs.
            message = str(error) if isinstance(error, RuntimeError) else type(error).__name__
            print(f"Startup failed: {message}", file=sys.stderr)
            sys.exit(1)
