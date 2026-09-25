"""Repairing a syntax error before the run starts.

`pulse run train.py` is the only place this can be done. Pulse normally attaches from
inside the script, and a syntax error in the script itself happens before any of it
runs: Python compiles the whole file first, so the process dies at compile time and
`import pulse` never executes.

cli.py handles that by handing the error to the same machinery that handles a crash
during training -- PulseCLI's excepthook, which asks the agent, writes the fix to the
file and restarts the run. Four things have to be true for that to be right, and each
of them is a way it can silently not be:

  * A file that does not compile must never be run anyway.
  * The fix has to reach the user's own file, not a copy of it.
  * The restart has to come back under `pulse run`, or the restarted run is untracked.
  * A run Pulse itself restarted must not start a second repair -- the parent owns the
    retry loop, and two of them fight over the same file.

No agent is called here: the CLI the repair drives is faked, and the fake writes the
fix the way a real one would.
"""
import os
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from pulse import cli  # noqa: E402

BROKEN = ("def train():\n"
          "    total = 0\n"
          "    for i in range(3):\n"
          "        total += 1 / (i + 1\n"
          "    return total\n")
FIXED = BROKEN.replace("(i + 1", "(i + 1)")


class FakeRepairCLI:
    """Enough of PulseCLI to watch how cli.py drives the repair."""

    def __init__(self):
        self.source = None
        self.script_path = None
        self.banner = False
        self.setup_calls = 0
        self.setup_raises = None

    def set_code_text(self, source, script_path=None):
        self.source, self.script_path = source, script_path

    def print_banner(self):
        self.banner = True

    def interactive_setup(self):
        self.setup_calls += 1
        if self.setup_raises is not None:
            raise self.setup_raises


def drive_repair(work, *, outcome="fix", setup_raises=None, script_name="train.py"):
    """Run `pulse run <broken script>` with the agent side faked out.

    `outcome` is what the excepthook does: "fix" writes the repair and raises SystemExit
    the way a successful restart does, "nothing" returns without fixing anything.
    Returns (path, status, fake_cli, executed).
    """
    path = os.path.join(work, script_name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(BROKEN)

    fake = FakeRepairCLI()
    fake.setup_raises = setup_raises

    def install_hook(driven_cli):
        def hook(exc_type, exc_value, tb):
            if outcome == "fix":
                # What PulseCLI does with a fix: write it to the file it was given, then
                # restart. The restart replaces this process, so it leaves as SystemExit.
                with open(driven_cli.script_path, "w", encoding="utf-8") as handle:
                    handle.write(FIXED)
                raise SystemExit(0)

        sys.excepthook = hook

    original_hook = sys.excepthook
    executed = []
    try:
        # _set_process_view rewrites sys.argv and prepends the script's directory to
        # sys.path, permanently. Left alone, each test leaves sys.argv[0] pointing at a
        # file in a deleted temp directory -- which other code here reads.
        with mock.patch.object(sys, "argv", list(sys.argv)), \
             mock.patch.object(sys, "path", list(sys.path)), \
             mock.patch.object(cli, "_make_syntax_repair_cli", return_value=fake), \
             mock.patch("pulse.pulse._install_cli_excepthook", install_hook), \
             mock.patch.object(cli, "_execute",
                               side_effect=lambda *a, **k: executed.append(a) or 0):
            status = cli.run_script(path, [])
    finally:
        sys.excepthook = original_hook
    return path, status, fake, executed


class RepairRouting(unittest.TestCase):
    """Which way a script that does not compile goes."""

    def test_a_file_that_does_not_compile_is_never_run(self):
        with tempfile.TemporaryDirectory() as work:
            _, _, _, executed = drive_repair(work, outcome="nothing")
            self.assertEqual(executed, [], "a file with a syntax error was executed anyway")

    def test_a_script_that_compiles_is_run_without_any_repair(self):
        with tempfile.TemporaryDirectory() as work:
            path = os.path.join(work, "fine.py")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("loss = 1.0\nfor step in range(3):\n    loss *= 0.9\n")
            with mock.patch.object(cli, "_repair_and_restart",
                                   side_effect=AssertionError("repaired a valid file")), \
                 mock.patch.object(cli, "_execute", return_value=0):
                self.assertEqual(cli.run_script(path, []), 0)

    def test_a_run_pulse_restarted_does_not_start_a_second_repair(self):
        # The parent process owns the retry loop and feeds the output back to the agent.
        # A child that repairs as well means two of them writing the same file.
        from pulse import pulse_cli
        with tempfile.TemporaryDirectory() as work:
            path = os.path.join(work, "train.py")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(BROKEN)
            with mock.patch.dict(os.environ, {pulse_cli._RESTART_CHILD_ENV: "1"}), \
                 mock.patch.object(cli, "_repair_and_restart",
                                   side_effect=AssertionError("the child repaired too")):
                self.assertEqual(cli.run_script(path, []), 1)


class RepairReachesTheUsersFile(unittest.TestCase):
    """The failure this was written for: the fix landed somewhere the user never sees."""

    def test_the_repair_is_given_the_users_own_path(self):
        with tempfile.TemporaryDirectory() as work:
            path, _, fake, _ = drive_repair(work)
            self.assertEqual(fake.script_path, path,
                             "the repair was pointed at a copy, so the fix cannot reach "
                             "the file the user runs")
            self.assertEqual(fake.source, BROKEN)

    def test_a_successful_repair_reports_the_restarted_runs_status(self):
        # Not "the file ends up fixed": the fix is written by this test's own fake hook,
        # so asserting on the content would only check the test's own constant. What is
        # worth pinning is that the SystemExit a restart leaves by becomes the exit
        # status of `pulse run`, rather than being swallowed into a failure.
        with tempfile.TemporaryDirectory() as work:
            _, status, _, executed = drive_repair(work)
            self.assertEqual(status, 0)
            self.assertEqual(executed, [],
                             "it ran the script itself as well as restarting it")

    def test_the_restart_hook_is_installed_before_anything_can_use_it(self):
        # Pulse asks this hook how to restart. Unset, a fix restarts with plain
        # `python script.py` and the restarted run is untracked -- under `pulse run` the
        # auto_track call is added in memory, never written to the file.
        from pulse import pulse_cli
        with tempfile.TemporaryDirectory() as work:
            seen = {}

            def capture(*a, **k):
                seen["hook"] = pulse_cli._RESTART_ARGV_HOOK
                raise SystemExit(0)

            path = os.path.join(work, "train.py")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(BROKEN)
            with mock.patch.object(sys, "argv", list(sys.argv)), \
                 mock.patch.object(sys, "path", list(sys.path)), \
                 mock.patch.object(cli, "_repair_and_restart", capture):
                try:
                    cli.run_script(path, [])
                except SystemExit:
                    pass
        self.assertIs(seen.get("hook"), cli._restart_argv,
                      "the restart hook was not wired up, so a fix would restart untracked")

    def test_the_restart_comes_back_under_pulse_run_on_the_real_script(self):
        # A plain `python train.py` restart would apply the fix and lose the tracking:
        # under `pulse run` the auto_track call is added in memory, not written to disk.
        argv = cli._restart_argv("/usr/bin/python3", "/home/me/proj/train.py", ["--epochs", "2"])
        self.assertEqual(argv[0], "/usr/bin/python3")
        self.assertIn("/home/me/proj/train.py", argv)
        self.assertEqual(argv[-3:], ["/home/me/proj/train.py", "--epochs", "2"])
        self.assertIn("'run'", " ".join(argv), "the restart does not go back through pulse run")


class RepairGivesUpHonestly(unittest.TestCase):
    def test_no_fix_means_the_script_is_not_started(self):
        with tempfile.TemporaryDirectory() as work:
            path, status, _, executed = drive_repair(work, outcome="nothing")
            self.assertEqual(status, 1)
            self.assertEqual(executed, [])
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), BROKEN, "the file was changed anyway")

    def test_cancelling_setup_is_not_a_crash(self):
        for interruption in (EOFError(), KeyboardInterrupt()):
            with self.subTest(interruption=type(interruption).__name__):
                with tempfile.TemporaryDirectory() as work:
                    _, status, fake, executed = drive_repair(work, setup_raises=interruption)
                    self.assertEqual(status, 1)
                    self.assertEqual(executed, [])
                    self.assertEqual(fake.setup_calls, 1)

    def test_a_missing_script_is_reported_before_any_of_this(self):
        with tempfile.TemporaryDirectory() as work:
            self.assertEqual(cli.run_script(os.path.join(work, "nope.py"), []), 1)


class EntryPoint(unittest.TestCase):
    def test_python_m_pulse_is_the_same_entry_point(self):
        from pulse import __main__ as module
        self.assertIs(module.main, cli.main)


if __name__ == "__main__":
    unittest.main(verbosity=2)
