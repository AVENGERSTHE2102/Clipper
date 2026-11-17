"""Run with backend/.venv/bin/python -m unittest discover -s scripts -p test_start_local.py."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location("start_local", Path(__file__).with_name("start-local.py"))
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LocalStartupTests(unittest.TestCase):
    def test_root_env_preserves_secrets_as_data_and_local_access(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / ".env").write_text("DATABASE_URL='postgresql://example'\nANTHROPIC_API_KEY='literal!$(echo nope)`nope`'\nNEXT_PUBLIC_LOCAL_APP_ACCESS=true\n")
            with patch.dict(launcher.os.environ, {"PATH": "/bin"}, clear=True):
                env = launcher.environment(root)
            self.assertEqual(env["ANTHROPIC_API_KEY"], 'literal!$(echo nope)`nope`')
            self.assertEqual(env["NEXT_PUBLIC_LOCAL_APP_ACCESS"], "true")
            self.assertEqual(env["FRONTEND_PORT"], "3107")
            self.assertEqual(env["BACKEND_INTERNAL_URL"], "http://127.0.0.1:8000")

    def test_repeated_start_does_not_spawn_another_service(self):
        with patch.object(launcher, "saved_process", return_value=42), patch.object(launcher, "process_identity", return_value="running"), patch.object(launcher.subprocess, "Popen") as spawn:
            launcher.start("api", [], Path("."), {}, lambda: True, 8000)
            spawn.assert_not_called()

    def test_unmanaged_port_is_not_reused_or_killed(self):
        with patch.object(launcher, "saved_process", return_value=None), patch.object(launcher, "port_open", return_value=True), patch.object(launcher, "service_processes", return_value=[]), patch.object(launcher.subprocess, "Popen") as spawn:
            with self.assertRaisesRegex(RuntimeError, "unmanaged"):
                launcher.start("api", [], Path("."), {}, lambda: True, 8000)
            spawn.assert_not_called()

    def test_discovered_worker_can_be_reused_without_duplicate(self):
        with patch.object(launcher, "saved_process", return_value=None), patch.object(launcher, "service_processes", return_value=[(42, True)]), patch.object(launcher, "remember_process") as remember, patch.object(launcher.sys.stdin, "isatty", return_value=False), patch.object(launcher, "process_identity", return_value="worker"), patch.object(launcher.subprocess, "Popen") as spawn:
            launcher.start("worker", [], Path("."), {}, lambda: True)
            remember.assert_called_once_with("worker", 42)
            spawn.assert_not_called()

    def test_foreign_port_owner_is_never_offered_for_termination(self):
        with patch.object(launcher, "saved_process", return_value=None), patch.object(launcher, "service_processes", return_value=[(42, False)]), patch.object(launcher, "stop_process") as stop:
            with self.assertRaisesRegex(RuntimeError, "unmanaged conflict"):
                launcher.start("api", [], Path("."), {}, lambda: True, 8000)
            stop.assert_not_called()

    def test_stop_checks_process_identity_before_signalling(self):
        with patch.object(launcher, "process_identity", side_effect=["original", "recycled"]), patch.object(launcher, "saved_process", return_value=None), patch.object(launcher.os, "kill") as kill:
            with self.assertRaisesRegex(RuntimeError, "PID 42 changed"):
                launcher.stop_process("worker", 42)
            kill.assert_not_called()

    def test_graceful_stop_does_not_force_kill(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(launcher, "LOGS", Path(folder)), patch.object(launcher, "process_identity", side_effect=["original", "original", ""]), patch.object(launcher, "saved_process", return_value=None), patch.object(launcher.os, "kill") as kill:
                launcher.stop_process("worker", 42)
                kill.assert_called_once_with(42, launcher.signal.SIGTERM)

    def test_exited_child_is_not_reported_as_running(self):
        with patch.object(launcher.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="Z Mon Nov 16 defunct")):
            self.assertEqual(launcher.process_identity(42), "")

    def test_database_failure_message_does_not_expose_log_secrets(self):
        with tempfile.TemporaryDirectory() as folder:
            logs = Path(folder)
            (logs / "local-api.log").write_text("secret-password tenant/user postgres.project not found")
            with patch.object(launcher, "LOGS", logs):
                message = launcher.startup_error("api", 0)
            self.assertIn("Session pooler", message)
            self.assertNotIn("secret-password", message)
            self.assertNotIn("postgres.project", message)

    def test_previous_database_failure_is_not_reported_for_new_attempt(self):
        with tempfile.TemporaryDirectory() as folder:
            logs = Path(folder)
            path = logs / "local-api.log"
            path.write_text("tenant/user not found")
            offset = path.stat().st_size
            with path.open("a") as log:
                log.write("different failure")
            with patch.object(launcher, "LOGS", logs):
                self.assertNotIn("Supabase rejected", launcher.startup_error("api", offset))

    def test_recycled_pid_is_not_treated_as_our_service(self):
        with tempfile.TemporaryDirectory() as folder:
            logs = Path(folder)
            (logs / "local-api.pid.json").write_text(json.dumps({"pid": 42, "identity": "old process"}))
            with patch.object(launcher, "LOGS", logs), patch.object(launcher, "process_identity", return_value="unrelated process"):
                self.assertIsNone(launcher.saved_process("api"))


if __name__ == "__main__":
    unittest.main()
