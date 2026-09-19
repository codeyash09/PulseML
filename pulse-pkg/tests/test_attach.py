"""Watching a run that was not started under Pulse.

These test the parsing and the reporting, which are the parts that can be wrong quietly.
Actually reading another process needs py-spy and root, so the one end-to-end test skips
itself when either is missing.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
sys.path.insert(0, SRC)

from pulse import pulse_attach as attach     # noqa: E402

# One py-spy dump: a training loop inside a function, called from module level.
DUMP = [{
    "pid": 4242,
    "frames": [
        {"name": "train", "filename": "/home/me/proj/train.py", "short_filename": "train.py",
         "line": 9, "locals": [
             {"name": "loss", "repr": "1.9221563184394899"},
             {"name": "grad_norm", "repr": "0.768862527375796"},
             {"name": "step", "repr": "80"},
             {"name": "model", "repr": "<Net object at 0x7f...>"},
         ]},
        {"name": "<module>", "filename": "/home/me/proj/train.py", "short_filename": "train.py",
         "line": 11, "locals": []},
    ],
}]

LIBRARY_DUMP = [{
    "pid": 4242,
    "frames": [
        {"name": "_worker", "filename": "/usr/lib/python3.11/threading.py",
         "locals": [{"name": "loss", "repr": "99.0"}]},
        {"name": "train", "filename": "/home/me/proj/train.py",
         "locals": [{"name": "loss", "repr": "0.5"}]},
    ],
}]


class ParsingTest(unittest.TestCase):
    def test_reads_the_numbers_a_loop_is_holding(self):
        self.assertEqual(attach.values_from(DUMP),
                         {"loss": 1.9221563184394899, "grad_norm": 0.768862527375796,
                          "step": 80.0})

    def test_ignores_values_that_are_not_numbers(self):
        self.assertNotIn("model", attach.values_from(DUMP))

    def test_ignores_frames_from_libraries(self):
        # A `loss` inside threading.py is not the user's loss.
        self.assertEqual(attach.values_from(LIBRARY_DUMP), {"loss": 0.5})

    def test_finds_the_script(self):
        self.assertEqual(attach.script_of(DUMP), "/home/me/proj/train.py")

    def test_a_module_level_loop_reports_nothing(self):
        module_only = [{"frames": [{"name": "<module>", "filename": "/home/me/t.py",
                                    "locals": []}]}]
        self.assertEqual(attach.values_from(module_only), {})

    def test_empty_and_malformed_dumps(self):
        self.assertEqual(attach.values_from([]), {})
        self.assertEqual(attach.values_from([{"frames": None}]), {})
        self.assertIsNone(attach.script_of([]))


class ReadinessTest(unittest.TestCase):
    def test_missing_pyspy_is_reported_as_such(self):
        with mock.patch.object(attach, "pyspy_path", return_value=None):
            report = attach.describe_readiness(1234)
        self.assertFalse(report["ok"])
        self.assertIn("py-spy", report["reason"])

    def test_a_permission_failure_is_reported_verbatim(self):
        with mock.patch.object(attach, "pyspy_path", return_value="/usr/bin/py-spy"), \
             mock.patch.object(attach, "read_frames",
                               return_value=(None, "Permission Denied: Try running again")):
            report = attach.describe_readiness(1234)
        self.assertFalse(report["ok"])
        self.assertIn("Permission Denied", report["reason"])

    def test_readable_but_nothing_visible_says_why(self):
        module_only = [{"frames": [{"name": "<module>", "filename": "/home/me/t.py",
                                    "locals": []}]}]
        with mock.patch.object(attach, "pyspy_path", return_value="/usr/bin/py-spy"), \
             mock.patch.object(attach, "read_frames", return_value=(module_only, "")):
            report = attach.describe_readiness(1234)
        self.assertTrue(report["ok"])
        self.assertFalse(report["locals_visible"])
        self.assertIn("module globals", report["reason"])

    def test_root_is_not_needed_when_ptrace_is_open(self):
        with mock.patch.object(attach, "ptrace_scope", return_value=0), \
             mock.patch.object(os, "geteuid", return_value=1000):
            self.assertFalse(attach.needs_root())

    def test_root_is_needed_when_ptrace_is_restricted(self):
        with mock.patch.object(attach, "ptrace_scope", return_value=1), \
             mock.patch.object(os, "geteuid", return_value=1000):
            self.assertTrue(attach.needs_root())

    def test_sudo_is_used_in_the_command_when_root_is_needed(self):
        command = attach._command(77, "/usr/bin/py-spy", use_sudo=True, non_interactive=True)
        self.assertEqual(command[:3], ["sudo", "-n", "/usr/bin/py-spy"])
        self.assertIn("--locals", command)
        plain = attach._command(77, "/usr/bin/py-spy", use_sudo=False, non_interactive=True)
        self.assertEqual(plain[0], "/usr/bin/py-spy")


class StreamingTest(unittest.TestCase):
    """An attached run has to look like any other run to the rest of Pulse."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pulse-attach-")
        os.environ["PULSE_HOME"] = os.path.join(self.tmp, ".pulse")
        self.addCleanup(os.environ.pop, "PULSE_HOME", None)

    def _monitor(self, dump=DUMP):
        with mock.patch.object(attach, "read_frames", return_value=(dump, "")), \
             mock.patch.object(attach, "pyspy_path", return_value="/usr/bin/py-spy"):
            monitor = attach.AttachedMonitor(4242, script=os.path.join(self.tmp, "train.py"),
                                             directory=os.path.join(self.tmp, "spool"))
            monitor.sample_once()
            monitor.snapshot()
        return monitor

    def test_the_brain_reads_an_attached_run_like_any_other(self):
        from pulse.pulse_brain import Brain
        monitor = self._monitor()
        monitor.writer.close()
        brain = Brain(monitor.directory)
        brain.poll_once()
        self.assertEqual(brain.histories["loss"], [1.9221563184394899])
        self.assertEqual(brain.step, 80)

    def test_a_run_that_reports_nothing_leaves_no_trace(self):
        empty = [{"frames": [{"name": "<module>", "filename": "/home/me/t.py", "locals": []}]}]
        from pulse import pulse_stream as stream
        monitor = self._monitor(empty)
        self.assertEqual([s["session_id"] for s in stream.registered_sessions()], [],
                         "an attach that read nothing registered itself anyway")
        monitor.discard()
        self.assertFalse(os.path.isdir(monitor.directory))

    def test_a_run_that_reports_something_is_registered(self):
        from pulse import pulse_stream as stream
        monitor = self._monitor()
        self.assertIn(monitor.session_id,
                      [s["session_id"] for s in stream.registered_sessions()])
        monitor.stop()
        self.assertNotIn(monitor.session_id,
                         [s["session_id"] for s in stream.registered_sessions()],
                         "the pointer outlived the run")

    def test_a_nonfinite_value_raises_an_urgent_event(self):
        nan_dump = [{"frames": [{"name": "train", "filename": "/home/me/proj/train.py",
                                 "locals": [{"name": "loss", "repr": "nan"}]}]}]
        monitor = self._monitor(nan_dump)
        monitor.writer.close()
        from pulse import pulse_stream as stream
        frames = stream.StreamReader(monitor.directory).poll()
        urgent = [f for f in frames if f.get("event") == "nonfinite"]
        self.assertEqual(len(urgent), 1)
        self.assertTrue(urgent[0]["urgent"])


class LiveAttachTest(unittest.TestCase):
    """The real thing, against a process this test starts. Skipped unless it can run."""

    def setUp(self):
        if not attach.pyspy_path():
            self.skipTest("py-spy is not installed")
        if attach.needs_root() and not attach.sudo_is_passwordless():
            self.skipTest("reading another process needs root here, and sudo would prompt")
        self.tmp = tempfile.mkdtemp(prefix="pulse-live-attach-")

    def test_reads_a_running_process_it_did_not_start(self):
        script = os.path.join(self.tmp, "train.py")
        with open(script, "w", encoding="utf-8") as handle:
            handle.write(textwrap.dedent("""\
                import time

                def train():
                    loss = 2.0
                    for step in range(600):
                        loss = loss * 0.99 + 0.001
                        time.sleep(0.05)

                train()
            """))
        process = subprocess.Popen([sys.executable, script], cwd=self.tmp,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   start_new_session=True)
        self.addCleanup(process.kill)
        time.sleep(2)

        values = {}
        for _ in range(10):
            threads, error = attach.read_frames(process.pid)
            if threads:
                values = attach.values_from(threads)
                if values:
                    break
            time.sleep(0.5)
        self.assertIn("loss", values, f"nothing read from a live process: {values}")
        self.assertLess(values["loss"], 2.0)
        self.assertEqual(os.path.basename(attach.script_of(threads) or ""), "train.py")


if __name__ == "__main__":
    unittest.main(verbosity=2)
