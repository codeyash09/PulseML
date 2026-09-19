"""Repairing a syntax error before the run starts.

`pulse run train.py` is the only place this can be done. Pulse normally attaches from
inside the script, and a syntax error in the script itself happens before any of it
runs: Python compiles the whole file first, so the process dies at compile time and
`import pulse` never executes. cli.py handles that by repairing a copy, instrumenting
it and running it.

Three things had to be true for that to work, and each of them is a way it silently
did not: the agent has to be reachable without a terminal, a rate limit must not be
mistaken for a failed repair, and applying the fix must not hand control to Pulse's
restart machinery -- which re-ran the temporary copy and skipped everything after it,
leaving the user's own file broken with the fix stranded in a temp file.

No agent is called here.
"""
import os
import sys
import tempfile
import textwrap
import unittest
from types import SimpleNamespace
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from pulse import cli  # noqa: E402

BROKEN = "def train():\n    total = 0\n    for i in range(3):\n        total += 1 / (i + 1\n    return total\n"
FIXED = BROKEN.replace("(i + 1", "(i + 1)")


class FakePulseCLI:
    """Enough of PulseCLI to watch how the repair path drives it."""

    instances = []

    def __init__(self):
        self.agent_provider = "test"
        self.agent_key = "key"
        self.non_interactive = False
        self._suppress_auto_restart = False
        self._fix_applied_this_turn = False
        self._last_call_failed_transiently = False
        self.script_path = None
        self.asked = []
        self.transient_for = 0
        FakePulseCLI.instances.append(self)

    def set_code_text(self, source, script_path=None):
        self.script_path = script_path

    def _load_config(self):
        pass

    def _select_agent_provider_and_key(self, initial=False):
        return True

    def ask_agent(self, question, include_code=False):
        self.asked.append(question)
        if self.transient_for >= len(self.asked):
            self._last_call_failed_transiently = True
            return "rate limited"
        self._last_call_failed_transiently = False
        self._fix_applied_this_turn = True
        with open(self.script_path, "w", encoding="utf-8") as handle:
            handle.write(FIXED)
        return "fixed"


def run_repair(work, script_name="train.py", **cli_attrs):
    """Drive cli.main() over a broken script with the agent faked out."""
    path = os.path.join(work, script_name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(BROKEN)
    FakePulseCLI.instances = []

    def make():
        fake = FakePulseCLI()
        for name, value in cli_attrs.items():
            setattr(fake, name, value)
        return fake

    fake_module = mock.MagicMock()
    fake_module.PulseCLI = make
    with mock.patch.dict(sys.modules, {"pulse.pulse_cli": fake_module}), \
            mock.patch.object(cli, "_run_training_script",
                              return_value=SimpleNamespace(returncode=0)) as ran, \
            mock.patch.object(cli, "time") as clock, \
            mock.patch.object(sys, "argv", ["pulse", "run", path]):
        clock.sleep.return_value = None
        try:
            cli.main()
            error = None
        except SystemExit as exit_request:
            # A clean exit is how a finished run leaves; only a non-zero one is news.
            error = None if exit_request.code in (0, None) else exit_request
        except BaseException as exc:      # cli.main reports and re-raises
            error = exc
    return path, ran, error


class SyntaxRepair(unittest.TestCase):
    def test_the_users_own_file_is_repaired_not_just_a_temp_copy(self):
        with tempfile.TemporaryDirectory() as work:
            path, ran, error = run_repair(work)
            with open(path) as handle:
                repaired = handle.read()
            compile(repaired, path, "exec")          # raises if it is still broken
            self.assertIn("(i + 1)", repaired)
            self.assertTrue(ran.called, "the repaired script was never run")

    def test_the_restart_machinery_does_not_take_over_the_repair(self):
        """Applying a fix normally restarts the run. There is no run yet."""
        with tempfile.TemporaryDirectory() as work:
            run_repair(work)
            self.assertTrue(FakePulseCLI.instances, "no PulseCLI was built for the repair")
            self.assertTrue(FakePulseCLI.instances[0]._suppress_auto_restart,
                            "the repair let Pulse restart into a half-repaired temp file")

    def test_a_rate_limit_is_retried_rather_than_called_a_failed_repair(self):
        with tempfile.TemporaryDirectory() as work:
            path, ran, error = run_repair(work, transient_for=2)
            self.assertIsNone(error, "a transient provider error aborted the repair")
            self.assertGreaterEqual(len(FakePulseCLI.instances[0].asked), 3)
            with open(path) as handle:
                compile(handle.read(), path, "exec")

    def test_a_provider_that_stays_down_is_reported_as_such(self):
        with tempfile.TemporaryDirectory() as work:
            _, ran, error = run_repair(work, transient_for=99)
            self.assertIsNotNone(error)
            self.assertIn("could not reach", str(error).lower())
            self.assertFalse(ran.called, "a file that does not compile was run anyway")

    def test_no_terminal_still_reaches_the_agent(self):
        """A CI job has no tty; it must use PULSE_PROVIDER rather than stop at a prompt."""
        with tempfile.TemporaryDirectory() as work:
            with mock.patch.object(cli, "_bootstrap_pulse_config") as bootstrap:
                bootstrap.return_value = False       # would raise "setup was not completed"
                run_repair(work, agent_provider=None, agent_key=None)
            self.assertTrue(FakePulseCLI.instances[0].non_interactive,
                            "the repair did not switch to non-interactive without a tty")


class EntryPoint(unittest.TestCase):
    def test_python_m_pulse_is_the_same_entry_point(self):
        from pulse import __main__ as module
        self.assertIs(module.main, cli.main)


if __name__ == "__main__":
    unittest.main()
