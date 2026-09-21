"""Tests for the interactive console: finding runs, choosing one, and the views."""
import glob
import os
import shutil
import site
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
sys.path.insert(0, SRC)

from pulse import pulse_console as console      # noqa: E402
from pulse import pulse_stream as stream        # noqa: E402
from pulse.pulse_monitor import Monitor         # noqa: E402

TRAIN = textwrap.dedent("""\
    import time
    loss = 2.0
    for step in range({steps}):
        loss = loss * {factor} + 0.001
        grad_norm = loss * 0.5
        time.sleep(0.02)
    print("done")
""")


def _remove_tree(path):
    """rmtree, retried: a child being killed can recreate files under the directory
    between the walk and the unlink -- its spool, or a font cache in the fake HOME --
    which left a directory behind on every run. Patient, because under a full-suite
    load the child takes longer to go than it does on its own."""
    for attempt in range(6):
        shutil.rmtree(path, ignore_errors=True)
        if not os.path.exists(path):
            return
        time.sleep(0.25 * (attempt + 1))
    shutil.rmtree(path, ignore_errors=True)


def _kill_group(process):
    """Kill the launcher and anything it started."""
    import signal
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (OSError, AttributeError):
        try:
            process.kill()
        except OSError:
            pass
    try:
        process.wait(timeout=5)
    except Exception:
        pass


def child_env(home):
    os.makedirs(home, exist_ok=True)
    return dict(os.environ,
                PYTHONPATH=os.pathsep.join(
                    p for p in (SRC, site.getusersitepackages(),
                                os.environ.get("PYTHONPATH", "")) if p),
                HOME=home, PULSE_HOME=os.path.join(home, ".pulse"),
                PULSE_LOGGING="0", NO_COLOR="1")


class DiscoveryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pulse-console-")
        # Restore the cwd BEFORE the directory goes: a test that chdirs into its
        # own temp dir otherwise leaves the whole process in a deleted one.
        self.addCleanup(_remove_tree, self.tmp)
        self.addCleanup(os.chdir, os.getcwd())
        self.home = os.path.join(self.tmp, "home")
        os.environ["PULSE_HOME"] = os.path.join(self.home, ".pulse")
        self.addCleanup(os.environ.pop, "PULSE_HOME", None)

    def _session(self, name, *, finished=False, step=10, loss=0.5, pid=None):
        directory = os.path.join(self.tmp, "runs", ".pulse_stream", name)
        monitor = Monitor(directory=directory, session_id=name,
                          script_path=os.path.join(self.tmp, f"{name}.py"),
                          interval=0.0, tensor_interval=0.0)
        monitor.observe_locals({"loss": loss, "step": step})
        monitor.snapshot_state({"finished": finished})
        if finished:
            monitor.close()
        else:
            monitor.writer.close()
        if pid is not None:                      # pretend it belongs to another process
            stream._atomic_write_json(os.path.join(directory, "session.json"),
                                      {"session_id": name, "pid": pid,
                                       "script": os.path.join(self.tmp, f"{name}.py"),
                                       "started": time.time()})
        return directory

    def test_finds_a_run_from_anywhere_via_the_registry(self):
        self._session("alpha")
        os.chdir(self.tmp)                        # not the run's directory
        found = console.discover()
        self.assertEqual([s["session_id"] for s in found], ["alpha"])
        self.assertEqual(found[0]["step"], 10)
        self.assertAlmostEqual(found[0]["loss"], 0.5)

    def test_finds_a_run_by_scanning_when_the_registry_is_empty(self):
        self._session("beta")
        stream.unregister_session("beta")
        os.chdir(os.path.join(self.tmp, "runs"))
        found = console.discover()
        self.assertEqual([s["session_id"] for s in found], ["beta"])

    def test_a_finished_run_is_not_live(self):
        self._session("gamma", finished=True)
        os.chdir(self.tmp)
        self.assertEqual(console.discover()[0]["status"], "finished")

    def test_a_run_whose_process_is_gone_is_not_live(self):
        self._session("delta", pid=999999)        # a pid that cannot exist
        os.chdir(self.tmp)
        self.assertEqual(console.discover()[0]["status"], "ended")

    def test_scan_skips_heavy_directories(self):
        junk = os.path.join(self.tmp, "proj", "node_modules", ".pulse_stream", "nope")
        os.makedirs(junk, exist_ok=True)
        open(os.path.join(junk, "events.jsonl"), "w").close()
        found = console._scan_for_spools(os.path.join(self.tmp, "proj"))
        self.assertEqual(found, [])

    def test_the_order_does_not_change_between_calls(self):
        # Two live runs take turns being the most recent writer. Ordering by activity
        # means the list reorders between reading a number and typing it, so
        # `pulse watch 2` attaches to the wrong run.
        for name in ("one", "two", "three"):
            self._session(name)
            time.sleep(0.01)
        os.chdir(self.tmp)
        first = [s["session_id"] for s in console.discover()]
        for _ in range(3):
            time.sleep(0.05)
            self.assertEqual([s["session_id"] for s in console.discover()], first)

    def test_a_directory_vanishing_mid_scan_is_not_a_crash(self):
        # A cleanup job, or somebody deleting .pulse_stream while `pulse` is looking:
        # os.walk has already reported the directory when the read of it fails.
        import unittest.mock as mock
        os.makedirs(os.path.join(self.tmp, "p", ".pulse_stream", "s1"))
        open(os.path.join(self.tmp, "p", ".pulse_stream", "s1", "events.jsonl"), "w").close()
        real_listdir = os.listdir

        def racy_listdir(path, *a, **k):
            if str(path).endswith(".pulse_stream"):
                raise FileNotFoundError(2, "No such file or directory", str(path))
            return real_listdir(path, *a, **k)

        with mock.patch.object(os, "listdir", racy_listdir):
            self.assertEqual(console._scan_for_spools(os.path.join(self.tmp, "p")), [])

    def test_a_real_spool_is_still_found(self):
        os.makedirs(os.path.join(self.tmp, "q", ".pulse_stream", "s2"))
        open(os.path.join(self.tmp, "q", ".pulse_stream", "s2", "events.jsonl"), "w").close()
        found = console._scan_for_spools(os.path.join(self.tmp, "q"))
        self.assertEqual([os.path.basename(p) for p in found], ["s2"])

    def test_an_unreadable_registry_is_not_a_crash(self):
        import unittest.mock as mock
        with mock.patch.object(os, "listdir", side_effect=PermissionError("nope")):
            self.assertEqual(stream.registered_sessions(), [])

    def test_duplicates_from_both_sources_collapse(self):
        self._session("epsilon")
        os.chdir(os.path.join(self.tmp, "runs"))
        self.assertEqual(len([s for s in console.discover() if s["session_id"] == "epsilon"]), 1)


class LivenessProbeTest(unittest.TestCase):
    def test_windows_never_reaches_os_kill(self):
        # os.kill(pid, 0) on Windows is TerminateProcess: listing runs would kill them.
        import unittest.mock as mock
        with mock.patch.object(os, "name", "nt"), \
             mock.patch.object(os, "kill", side_effect=AssertionError("os.kill on Windows")):
            self.assertFalse(console._pid_alive(999999))   # no ctypes here: answers False

    def test_posix_probe_says_yes_for_this_process(self):
        if os.name == "nt":
            self.skipTest("POSIX branch")
        self.assertTrue(console._pid_alive(os.getpid()))
        self.assertFalse(console._pid_alive(999999))

    def test_nonsense_pids_are_not_alive(self):
        for value in (None, 0, "", "abc", -1):
            self.assertFalse(console._pid_alive(value), value)


class PickTest(unittest.TestCase):
    SESSIONS = [
        {"session_id": "20260918-1000-aaa", "status": "live", "script": "/x/train.py"},
        {"session_id": "20260918-0900-bbb", "status": "finished", "script": "/x/tune.py"},
    ]

    def test_a_single_live_run_is_chosen_without_asking(self):
        self.assertEqual(console.pick_session(self.SESSIONS, None)["session_id"],
                         "20260918-1000-aaa")

    def test_two_live_runs_need_an_answer(self):
        both = [dict(s, status="live") for s in self.SESSIONS]
        self.assertIsNone(console.pick_session(both, None))

    def test_pick_by_index_id_and_script_name(self):
        self.assertEqual(console.pick_session(self.SESSIONS, "2")["session_id"],
                         "20260918-0900-bbb")
        self.assertEqual(console.pick_session(self.SESSIONS, "20260918-0900-bbb")["script"],
                         "/x/tune.py")
        self.assertEqual(console.pick_session(self.SESSIONS, "tune.py")["script"], "/x/tune.py")

    def test_an_unknown_name_picks_nothing(self):
        self.assertIsNone(console.pick_session(self.SESSIONS, "nosuchthing"))

    def test_the_only_run_is_chosen_even_when_it_has_ended(self):
        one = [dict(self.SESSIONS[1])]
        self.assertEqual(console.pick_session(one, None)["session_id"], "20260918-0900-bbb")


class SparklineTest(unittest.TestCase):
    def test_flat_series(self):
        self.assertEqual(console.sparkline([1.0] * 5), "▁" * 5)

    def test_rising_series_ends_higher_than_it_starts(self):
        line = console.sparkline([float(i) for i in range(20)])
        self.assertLess(console.SPARK.index(line[0]), console.SPARK.index(line[-1]))

    def test_long_series_is_compressed_to_the_width(self):
        self.assertEqual(len(console.sparkline([float(i) for i in range(5000)], width=40)), 40)

    def test_empty_series(self):
        self.assertEqual(console.sparkline([]), "")


class EndToEndTest(unittest.TestCase):
    """A real streaming run, discovered and driven from a different directory."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pulse-console-e2e-")
        # Restore the cwd BEFORE the directory goes: a test that chdirs into its
        # own temp dir otherwise leaves the whole process in a deleted one.
        self.addCleanup(_remove_tree, self.tmp)
        self.addCleanup(os.chdir, os.getcwd())
        self.proj = os.path.join(self.tmp, "proj")
        os.makedirs(self.proj)
        self.home = os.path.join(self.tmp, "home")

    def _start_run(self, steps=4000, factor=0.999):
        script = os.path.join(self.proj, "train.py")
        with open(script, "w", encoding="utf-8") as handle:
            handle.write(TRAIN.format(steps=steps, factor=factor))
        # Its own process group: `pulse run` spawns the instrumented copy as a CHILD, so
        # killing only the parent leaves a training process running on the machine after
        # the test has finished. Several were still going after an earlier run.
        process = subprocess.Popen([sys.executable, "-m", "pulse", "run", "--stream", "train.py"],
                                   cwd=self.proj, env=child_env(self.home),
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   start_new_session=True)
        self.addCleanup(_kill_group, process)
        for _ in range(100):                     # wait for the stream to appear
            if glob.glob(os.path.join(self.proj, ".pulse_stream", "*", "state.json")):
                return process, script
            time.sleep(0.2)
        self.fail("the run never produced a stream: " + (process.stdout.read() or "")[-800:])

    def test_console_attaches_from_another_directory_and_reports(self):
        self._start_run()
        time.sleep(2)
        result = subprocess.run([sys.executable, "-m", "pulse"],
                                cwd=self.tmp,                 # NOT the project directory
                                env=child_env(self.home), input="/status\n/vars\n/cd\n/quit\n",
                                capture_output=True, text=True, timeout=180)
        out = result.stdout
        self.assertIn("train.py", out)
        self.assertIn("Attached to", out)
        self.assertIn("loss", out)
        self.assertIn(self.proj, out, "the console did not anchor to the script's directory")

    def test_run_command_records_the_users_script_not_the_temp_copy(self):
        _, script = self._start_run()
        time.sleep(1.5)
        os.chdir(self.tmp)
        os.environ["PULSE_HOME"] = os.path.join(self.home, ".pulse")
        self.addCleanup(os.environ.pop, "PULSE_HOME", None)
        sessions = console.discover()
        self.assertTrue(sessions, "no session discovered")
        self.assertEqual(sessions[0]["script"], script)
        self.assertNotIn("instrumented", sessions[0]["script"])

    def test_a_live_run_reports_its_progress_without_being_closed(self):
        self._start_run()
        time.sleep(3)
        os.chdir(self.tmp)
        os.environ["PULSE_HOME"] = os.path.join(self.home, ".pulse")
        self.addCleanup(os.environ.pop, "PULSE_HOME", None)
        session = console.discover()[0]
        self.assertEqual(session["status"], "live")
        self.assertGreater(session["step"] or 0, 0, "a live run reported no steps")

    def test_sessions_command_lists_the_run(self):
        self._start_run()
        time.sleep(2)
        result = subprocess.run([sys.executable, "-m", "pulse", "sessions"],
                                cwd=self.tmp, env=child_env(self.home),
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("train.py", result.stdout)
        self.assertIn("live", result.stdout)

    def test_with_no_runs_it_says_how_to_start_one(self):
        empty_home = os.path.join(self.tmp, "empty-home")
        result = subprocess.run([sys.executable, "-m", "pulse"],
                                cwd=self.tmp, env=child_env(empty_home),
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 1)
        self.assertIn("pulse run --stream train.py", result.stdout)
        self.assertIn('auto_track(mode="stream")', result.stdout)


class DispatchTest(unittest.TestCase):
    """What `pulse ...` does with each shape of command line.

    The rule: `run` starts a run, and without it a script name means the run of that
    script that is already going. `pulse --stream train.py` -- a launch option with no
    `run` -- used to open the console, which ignored both the flag and the script and
    reported nothing to watch while the script sat there unstarted.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pulse-dispatch-")
        # Restore the cwd BEFORE the directory goes: a test that chdirs into its
        # own temp dir otherwise leaves the whole process in a deleted one.
        self.addCleanup(_remove_tree, self.tmp)
        self.addCleanup(os.chdir, os.getcwd())
        with open(os.path.join(self.tmp, "train.py"), "w", encoding="utf-8") as handle:
            handle.write("loss = 1.0\n")
        self.calls = []
        os.chdir(self.tmp)

    def _dispatch(self, argv):
        import unittest.mock as mock
        from pulse import cli

        def fake_run_script(script, script_args, stream=False, cwd=None, again=False):
            self.calls.append({"run": script, "args": script_args, "stream": stream,
                               "again": again})
            return 0

        def fake_console(argv):
            self.calls.append({"console": list(argv)})
            return 0

        with mock.patch.object(cli, "run_script", fake_run_script), \
             mock.patch("pulse.pulse_console.main", fake_console):
            return cli.main(argv)

    def test_run_starts_a_run(self):
        self._dispatch(["run", "--stream", "train.py"])
        self.assertEqual(self.calls, [{"run": "train.py", "args": [], "stream": True,
                                       "again": False}])

    def test_run_passes_the_scripts_own_arguments_through(self):
        self._dispatch(["run", "train.py", "--epochs", "3"])
        self.assertEqual(self.calls[0]["args"], ["--epochs", "3"])

    def test_a_bare_script_name_watches_rather_than_starts(self):
        self._dispatch(["train.py"])
        self.assertEqual(self.calls, [{"console": ["train.py"]}])

    def test_a_launch_option_without_run_is_explained_not_guessed(self):
        status = self._dispatch(["--stream", "train.py"])
        self.assertEqual(status, 1)
        self.assertEqual(self.calls, [], "it started or watched something instead of asking")

    def test_again_reaches_run_script(self):
        self._dispatch(["run", "--again", "--stream", "train.py"])
        self.assertTrue(self.calls[0]["again"])

    def test_bare_pulse_is_the_console(self):
        self._dispatch([])
        self.assertIn("console", self.calls[0])

    def test_sessions_is_the_console(self):
        self._dispatch(["sessions"])
        self.assertIn("console", self.calls[0])

    def test_model_flag_alone_is_the_console(self):
        self._dispatch(["--model", "some/model"])
        self.assertIn("console", self.calls[0])

    def test_watch_by_script_name_is_the_console(self):
        self._dispatch(["watch", "train.py"])
        self.assertIn("console", self.calls[0])


class WatchByNameTest(unittest.TestCase):
    """`pulse train.py` when there is no Pulse run of that name, only the process.

    Looking the pid up by hand is work the tool can do, so `sudo pulse train.py` has to
    be the whole command.
    """

    def _process(self, pid, script="/x/train.py", started=0.0):
        return {"pid": pid, "script": script, "cmdline": f"python3 {script}",
                "started": started}

    def _run(self, processes, readiness, attached=None):
        import unittest.mock as mock
        calls = {}

        def fake_attach(pid, model=""):
            calls["pid"] = pid
            return 0

        with mock.patch.object(console, "unmonitored_python_processes", return_value=processes), \
             mock.patch("pulse.pulse_attach.describe_readiness", return_value=readiness), \
             mock.patch.object(console, "attach_to_pid", fake_attach):
            status = console.watch_by_name("train.py")
        return status, calls

    def test_one_readable_process_is_attached_without_a_pid(self):
        status, calls = self._run(
            [self._process(4242)],
            {"ok": True, "pyspy": "/usr/bin/py-spy", "needs_root": False,
             "ptrace_scope": 0, "locals_visible": True, "reason": ""})
        self.assertEqual(status, 0)
        self.assertEqual(calls.get("pid"), 4242, "it did not attach to the one process")

    def test_needing_root_says_sudo_pulse_script_not_a_pid(self):
        import io, contextlib
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            status, calls = self._run(
                [self._process(4242)],
                {"ok": False, "pyspy": "/usr/bin/py-spy", "needs_root": True,
                 "ptrace_scope": 1, "passwordless_sudo": True, "reason": "Permission Denied"})
        output = buffer.getvalue()
        self.assertEqual(status, 1)
        self.assertEqual(calls, {}, "it attached anyway")
        self.assertIn("ptrace", output)
        self.assertIn("ptrace_scope is 1", output)
        # The suggested command has to name the script the user asked about. It used to
        # be built from sys.argv, so it repeated whatever arguments this process happened
        # to be started with -- under the test runner, "sudo pulse discover -s tests".
        suggestion = [line for line in output.splitlines() if "sudo" in line and "env" not in line
                      or line.strip().startswith("sudo env")]
        self.assertTrue(suggestion, f"no sudo command was printed:\n{output}")
        self.assertTrue(any(line.rstrip().endswith("train.py") for line in suggestion),
                        f"the sudo command did not name train.py:\n{output}")
        self.assertNotIn("--pid", output, "it fell back to asking for a pid")

    def test_missing_pyspy_says_how_to_get_it(self):
        import io, contextlib
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            status, _ = self._run([self._process(4242)],
                                  {"ok": False, "pyspy": None, "needs_root": True,
                                   "reason": "py-spy is not installed"})
        self.assertEqual(status, 1)
        self.assertIn("pip install py-spy", buffer.getvalue())

    def test_several_processes_ask_which(self):
        import io, contextlib
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            status, calls = self._run([self._process(1), self._process(2)],
                                      {"ok": True, "pyspy": "/usr/bin/py-spy",
                                       "needs_root": False, "locals_visible": True})
        self.assertEqual(status, 1)
        self.assertEqual(calls, {}, "it guessed between two processes")
        self.assertIn("--pid 1", buffer.getvalue())
        self.assertIn("--pid 2", buffer.getvalue())

    def test_nothing_running_says_how_to_start_it(self):
        import io, contextlib
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            status, _ = self._run([], {"ok": False, "pyspy": None, "needs_root": True})
        self.assertEqual(status, 1)
        self.assertIn("pulse run --stream train.py", buffer.getvalue())


class SelectionTest(unittest.TestCase):
    """Choosing between runs when a name matches more than one."""

    def _sessions(self, *specs):
        return [{"session_id": f"id-{i}", "script": f"/x/{script}", "status": status,
                 "started": i, "directory": f"/d{i}"}
                for i, (script, status) in enumerate(specs, 1)]

    def test_a_script_name_finds_its_run(self):
        sessions = self._sessions(("train.py", "live"), ("other.py", "live"))
        self.assertEqual(console.pick_session(sessions, "train.py")["script"], "/x/train.py")

    def test_the_live_one_wins_over_a_finished_one_of_the_same_script(self):
        sessions = self._sessions(("train.py", "finished"), ("train.py", "live"))
        self.assertEqual(console.pick_session(sessions, "train.py")["status"], "live")

    def test_two_live_runs_of_one_script_are_ambiguous(self):
        sessions = self._sessions(("train.py", "live"), ("train.py", "live"))
        self.assertIsNone(console.pick_session(sessions, "train.py"))
        self.assertEqual(len(console.matching_sessions(sessions, "train.py")), 2)

    def test_a_name_that_matches_nothing(self):
        sessions = self._sessions(("train.py", "live"))
        self.assertIsNone(console.pick_session(sessions, "nope.py"))
        self.assertEqual(console.matching_sessions(sessions, "nope.py"), [])


class AlreadyRunningTest(unittest.TestCase):
    """Starting a second copy of a script that is already training is not what
    "watch my run" means: it competes for the same GPU and the console shows two."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pulse-already-")
        # Restore the cwd BEFORE the directory goes: a test that chdirs into its
        # own temp dir otherwise leaves the whole process in a deleted one.
        self.addCleanup(_remove_tree, self.tmp)
        self.addCleanup(os.chdir, os.getcwd())
        self.script = os.path.join(self.tmp, "train.py")
        with open(self.script, "w", encoding="utf-8") as handle:
            handle.write("import time\nfor i in range(600):\n    time.sleep(0.05)\n")

    def _start(self, cwd, argument):
        process = subprocess.Popen([sys.executable, argument], cwd=cwd,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   start_new_session=True)
        self.addCleanup(_kill_group, process)
        time.sleep(1.5)
        return process

    def test_finds_a_copy_started_with_a_relative_path(self):
        from pulse.cli import already_running
        if not os.path.isdir("/proc"):
            self.skipTest("/proc only")
        process = self._start(self.tmp, "train.py")
        found = already_running(self.script)
        self.assertIn(process.pid, [pid for pid, _ in found])

    def test_a_same_named_file_elsewhere_is_not_a_match(self):
        from pulse.cli import already_running
        if not os.path.isdir("/proc"):
            self.skipTest("/proc only")
        other = os.path.join(self.tmp, "other")
        os.makedirs(other)
        with open(os.path.join(other, "train.py"), "w", encoding="utf-8") as handle:
            handle.write("import time\nfor i in range(600):\n    time.sleep(0.05)\n")
        process = self._start(other, "train.py")
        found = already_running(self.script)          # asking about the FIRST train.py
        self.assertNotIn(process.pid, [pid for pid, _ in found],
                         "an unrelated file with the same name was treated as this run")

    def test_nothing_running_is_nothing_found(self):
        from pulse.cli import already_running
        self.assertEqual(already_running(self.script), [])


class ArgumentScopingTest(unittest.TestCase):
    """Pulse's own flags stop at the script path; after it, they are the script's."""

    def _parse(self, argv):
        import unittest.mock as mock
        from pulse import cli
        cli._TRACK_MODE = "cli"
        with mock.patch.object(cli.sys, "argv", ["pulse"] + argv), \
             mock.patch.object(cli, "_run_pulse_run", create=True):
            head = []
            inner = argv
            for index, argument in enumerate(inner[1:], start=1):
                if not argument.startswith("-"):
                    head = inner[1:index]
                    break
            else:
                head = inner[1:]
            return head

    def test_stream_before_the_script_is_pulses(self):
        self.assertIn("--stream", self._parse(["run", "--stream", "train.py"]))

    def test_stream_after_the_script_is_the_scripts(self):
        self.assertNotIn("--stream", self._parse(["run", "train.py", "--stream"]))

    def test_console_model_value_is_not_mistaken_for_a_run(self):
        import unittest.mock as mock
        seen = {}

        def fake_run_console(session, sessions, agent=None, sensitivity=0.3):
            seen["session"] = session
            return 0

        sessions = [
            {"session_id": "aaa", "status": "live", "script": "/x/one.py", "started": 2,
             "directory": "/d1"},
            {"session_id": "bbb", "status": "live", "script": "/x/two.py", "started": 1,
             "directory": "/d2"},
        ]
        with mock.patch.object(console, "discover", return_value=sessions), \
             mock.patch.object(console, "run_console", fake_run_console), \
             mock.patch.object(console, "build_litellm_agent", create=True, return_value=None):
            console.main(["watch", "--model", "some/model", "2"])
        self.assertEqual(seen["session"]["session_id"], "bbb",
                         "the model id was taken as the run to attach to")


class StreamFlagReachesTheScriptTest(unittest.TestCase):
    """--stream has to win even when the script calls auto_track() itself.

    `pulse run` only injects auto_track into a script that does not already have one, so
    a script written the way the README shows was run as it is, in the default cli mode,
    with --stream doing nothing: no spool, and `pulse` reporting no runs on the machine.
    """

    OWN_CALL = textwrap.dedent("""\
        from pulse import auto_track
        import time

        auto_track()                 # exactly as the README documents it

        loss = 2.0
        for step in range(200):
            loss = loss * 0.99 + 0.001
            time.sleep(0.02)
        print("done")
    """)

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pulse-streamflag-")
        # Restore the cwd BEFORE the directory goes: a test that chdirs into its
        # own temp dir otherwise leaves the whole process in a deleted one.
        self.addCleanup(_remove_tree, self.tmp)
        self.addCleanup(os.chdir, os.getcwd())
        self.proj = os.path.join(self.tmp, "proj")
        os.makedirs(self.proj)
        self.home = os.path.join(self.tmp, "home")

    def test_a_script_with_its_own_auto_track_still_streams(self):
        script = os.path.join(self.proj, "train.py")
        with open(script, "w", encoding="utf-8") as handle:
            handle.write(self.OWN_CALL)
        process = subprocess.Popen(
            [sys.executable, "-m", "pulse", "run", "--stream", "train.py"],
            cwd=self.proj, env=child_env(self.home), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, start_new_session=True)
        self.addCleanup(_kill_group, process)
        for _ in range(100):
            if glob.glob(os.path.join(self.proj, ".pulse_stream", "*", "events.jsonl")):
                break
            time.sleep(0.2)
        else:
            self.fail("--stream was ignored for a script that calls auto_track itself: "
                      + (process.stdout.read() or "")[-600:])

        os.chdir(self.tmp)
        os.environ["PULSE_HOME"] = os.path.join(self.home, ".pulse")
        self.addCleanup(os.environ.pop, "PULSE_HOME", None)
        sessions = console.discover()
        self.assertTrue(sessions, "the run streamed but was not discoverable")
        self.assertEqual(os.path.basename(sessions[0]["script"] or ""), "train.py")

    def test_without_the_flag_the_script_keeps_its_own_mode(self):
        # The old behaviour has to survive: no --stream, no PULSE_MODE forced.
        from pulse import cli
        import unittest.mock as mock
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PULSE_MODE", None)
            with mock.patch.object(cli, "_execute", return_value=0), \
                 mock.patch.object(cli, "_set_process_view"):
                script = os.path.join(self.proj, "plain.py")
                with open(script, "w", encoding="utf-8") as handle:
                    handle.write("loss = 1.0\n")
                cli.run_script(script, [], stream=False)
            self.assertNotEqual(os.environ.get("PULSE_MODE"), "stream")


class OldWaysStillWorkTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pulse-oldways-")
        # Restore the cwd BEFORE the directory goes: a test that chdirs into its
        # own temp dir otherwise leaves the whole process in a deleted one.
        self.addCleanup(_remove_tree, self.tmp)
        self.addCleanup(os.chdir, os.getcwd())
        self.home = os.path.join(self.tmp, "home")

    def test_pulse_run_without_stream_still_uses_the_original_mode(self):
        # Behavioural rather than signature-coupled: inject into a real script and read
        # the mode back out of the tree.
        from pulse.cli import PulseASTInjector
        import ast

        def injected_mode(mode):
            tree = ast.parse("import time\nloss = 1.0\nfor i in range(3):\n    loss *= 0.9\n")
            PulseASTInjector(mode=mode).visit(tree)
            source = ast.unparse(tree)
            self.assertIn("auto_track", source)
            return source

        self.assertIn("'cli'", injected_mode("cli"))
        self.assertIn("'stream'", injected_mode("stream"))

    def test_auto_track_stream_mode_is_still_reachable_directly(self):
        script = os.path.join(self.tmp, "direct.py")
        with open(script, "w", encoding="utf-8") as handle:
            handle.write(textwrap.dedent("""\
                from pulse import auto_track
                auto_track(mode="stream")
                loss = 1.0
                for step in range(20):
                    loss *= 0.9
                print("done")
            """))
        result = subprocess.run([sys.executable, script], cwd=self.tmp,
                                env=child_env(self.home), capture_output=True,
                                text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("done", result.stdout)
        self.assertTrue(glob.glob(os.path.join(self.tmp, ".pulse_stream", "*")))


class SudoCommandTest(unittest.TestCase):
    """The command Pulse prints for sudo has to be one that works when pasted.

    `sudo pulse ...` is the obvious suggestion and it fails on a pip install: sudo
    replaces PATH with secure_path, which has no ~/.local/bin, so the shell says
    "pulse: command not found" -- which is exactly what happened to the user. Spelling
    out the full path fails differently: root's Python does not read the user's
    site-packages, so it dies on numpy instead.
    """

    def setUp(self):
        import unittest.mock as mock
        self.mock = mock

    def test_short_form_when_a_launcher_is_where_sudo_looks(self):
        mock = self.mock
        with mock.patch.object(console, "pulse_on_secure_path", return_value=True):
            self.assertEqual(console.sudo_relaunch_command(["train.py"]),
                             ["sudo", "pulse", "train.py"])

    def test_carries_path_and_home_when_pulse_is_only_in_local_bin(self):
        mock = self.mock
        with mock.patch.object(console, "pulse_on_secure_path", return_value=False), \
             mock.patch.object(sys, "argv", ["/home/me/.local/bin/pulse", "train.py"]):
            command = console.sudo_relaunch_command(["train.py"])
        self.assertEqual(command[:2], ["sudo", "env"])
        self.assertTrue(any(part.startswith("PATH=") for part in command),
                        f"PATH is not carried across, so pulse stays unfindable: {command}")
        self.assertTrue(any(part.startswith("HOME=") for part in command),
                        f"HOME is not carried across, so numpy stays unfindable: {command}")
        self.assertEqual(command[-1], "train.py")
        self.assertIn("/home/me/.local/bin/pulse", command)

    def test_a_module_invocation_relaunches_as_a_module(self):
        mock = self.mock
        with mock.patch.object(console, "pulse_on_secure_path", return_value=False), \
             mock.patch.object(sys, "argv", ["/home/me/pulseml/pulse-pkg/src/pulse/cli.py", "t.py"]):
            command = console.sudo_relaunch_command(["t.py"])
        self.assertIn("-m", command)
        self.assertEqual(command[command.index("-m") + 1], "pulse")

    def test_it_does_not_repeat_the_hosts_own_arguments(self):
        mock = self.mock
        with mock.patch.object(sys, "argv", ["pulse", "discover", "-s", "tests"]):
            self.assertNotIn("discover", console.sudo_relaunch_command(["train.py"]))


class InstallSudoTest(unittest.TestCase):
    """`pulse install-sudo` puts a launcher where sudo will actually look.

    It has to run as the user, not under sudo -- if `sudo pulse` worked there would be
    nothing to install -- and it must never overwrite an unrelated /usr/local/bin/pulse.
    """

    def setUp(self):
        import unittest.mock as mock
        self.mock = mock
        self.tmp = tempfile.mkdtemp(prefix="pulse-install-")
        # Restore the cwd BEFORE the directory goes: a test that chdirs into its
        # own temp dir otherwise leaves the whole process in a deleted one.
        self.addCleanup(_remove_tree, self.tmp)
        self.addCleanup(os.chdir, os.getcwd())
        self.target = os.path.join(self.tmp, "usr-local-bin-pulse")
        self.home = os.path.join(self.tmp, "home")
        os.makedirs(os.path.join(self.home, ".local", "bin"))
        open(os.path.join(self.home, ".local", "bin", "pulse"), "w").close()

    def _install(self, returncode=0, which=None):
        mock = self.mock
        calls = {}
        user_pulse = os.path.join(self.home, ".local", "bin", "pulse")

        def fake_run(command, *a, **k):
            calls["command"] = command
            if returncode == 0:                      # do what `sudo install` would do
                shutil.copyfile(command[-2], command[-1])
            return SimpleNamespace(returncode=returncode)

        empty = os.path.join(self.tmp, "no-system-pulse")
        os.makedirs(empty, exist_ok=True)
        with mock.patch.object(os.path, "expanduser", return_value=self.home), \
             mock.patch.object(console.shutil, "which",
                               return_value=user_pulse if which is None else which), \
             mock.patch.object(console, "SECURE_PATH", (empty,)), \
             mock.patch.object(sys, "argv", ["pulse", "install-sudo"]), \
             mock.patch("subprocess.run", fake_run):
            status = console.install_sudo_launcher(self.target)
        return status, calls

    def test_it_writes_a_launcher_that_runs_the_users_own_install(self):
        status, calls = self._install()
        self.assertEqual(status, 0)
        self.assertEqual(calls["command"][:2], ["sudo", "install"],
                         "it tried to write to /usr/local/bin without sudo")
        with open(self.target, encoding="utf-8") as handle:
            script = handle.read()
        self.assertTrue(script.startswith("#!/bin/sh"))
        self.assertIn("SUDO_USER", script,
                      "the launcher cannot find the real user's home, so it runs as root's")
        self.assertIn(os.path.join(self.home, ".local", "bin", "pulse"), script)

    def test_the_launcher_points_at_an_absolute_path_not_a_lookup(self):
        # /usr/local/bin is on every PATH, so a launcher that looks `pulse` up on PATH
        # finds itself and execs itself until the process table gives out.
        self._install()
        with open(self.target, encoding="utf-8") as handle:
            script = handle.read()
        self.assertNotIn("exec env HOME=\"$REAL_HOME\" pulse", script)
        self.assertIn(os.path.join(self.home, ".local", "bin", "pulse"), script)

    def test_it_refuses_to_point_the_launcher_at_itself(self):
        mock = self.mock
        empty = os.path.join(self.tmp, "no-system-pulse")
        os.makedirs(empty, exist_ok=True)
        with mock.patch.object(os.path, "expanduser", return_value=self.home), \
             mock.patch.object(console.shutil, "which", return_value=self.target), \
             mock.patch.object(console, "SECURE_PATH", (empty,)), \
             mock.patch.object(sys, "argv", ["pulse"]), \
             mock.patch("subprocess.run",
                        side_effect=AssertionError("installed a launcher that execs itself")):
            self.assertEqual(console.install_sudo_launcher(self.target), 1)

    def test_a_venv_install_is_pointed_at_rather_than_assumed(self):
        venv_pulse = os.path.join(self.tmp, "venv", "bin", "pulse")
        os.makedirs(os.path.dirname(venv_pulse))
        open(venv_pulse, "w").close()
        os.chmod(venv_pulse, 0o755)
        status, _ = self._install(which=venv_pulse)
        self.assertEqual(status, 0)
        with open(self.target, encoding="utf-8") as handle:
            self.assertIn(venv_pulse, handle.read(),
                          "a venv install was ignored in favour of an assumed ~/.local/bin")

    def test_the_launcher_is_valid_shell(self):
        self._install()
        result = subprocess.run(["sh", "-n", self.target], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def _run_launcher(self, home, *args, env=None, path=None):
        """Run the generated launcher for real, against a stand-in pulse."""
        environment = dict(os.environ, HOME=home)
        environment.pop("SUDO_USER", None)
        environment.pop("REAL_HOME", None)
        if path is not None:
            environment["PATH"] = path
        environment.update(env or {})
        # /bin/sh by absolute path: a test that strips PATH to hide getent would
        # otherwise fail to find the shell rather than exercising the launcher.
        return subprocess.run(["/bin/sh", self.target, *args], capture_output=True,
                              text=True, env=environment, timeout=30)

    def _stub_pulse(self, home):
        """A stand-in `pulse` that reports the HOME and arguments it was given."""
        os.makedirs(os.path.join(home, ".local", "bin"), exist_ok=True)
        path = os.path.join(home, ".local", "bin", "pulse")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write('#!/bin/sh\necho "HOME=[$HOME] ARGS=[$*]"\n')
        os.chmod(path, 0o755)
        return path

    def test_the_launcher_passes_home_and_arguments_through(self):
        target_pulse = os.path.join(self.home, ".local", "bin", "pulse")
        with open(target_pulse, "w", encoding="utf-8") as handle:
            handle.write('#!/bin/sh\necho "HOME=[$HOME] ARGS=[$*]"\n')
        os.chmod(target_pulse, 0o755)
        self._install()
        result = self._run_launcher(self.home, "train.py", "--model", "x")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"HOME=[{self.home}]", result.stdout)
        self.assertIn("ARGS=[train.py --model x]", result.stdout,
                      "the launcher mangled the arguments")

    def test_a_home_with_spaces_survives(self):
        spaced = os.path.join(self.tmp, "home with spaces")
        os.makedirs(os.path.join(spaced, ".local", "bin"))
        target_pulse = os.path.join(spaced, ".local", "bin", "pulse")
        with open(target_pulse, "w", encoding="utf-8") as handle:
            handle.write('#!/bin/sh\necho "HOME=[$HOME] ARGS=[$*]"\n')
        os.chmod(target_pulse, 0o755)
        self._install(which=target_pulse)
        result = self._run_launcher(spaced, "a b")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"HOME=[{spaced}]", result.stdout)
        self.assertIn("ARGS=[a b]", result.stdout)

    def test_a_launcher_whose_pulse_is_gone_says_so_instead_of_looping(self):
        self._install()
        os.unlink(os.path.join(self.home, ".local", "bin", "pulse"))
        result = self._run_launcher(self.home, "--version")
        self.assertEqual(result.returncode, 127)
        self.assertIn("install-sudo", result.stderr,
                      "it failed without saying how to fix it")

    # ---- the sudo path itself: the reason the launcher exists ----

    def test_under_sudo_it_uses_the_invoking_users_home_not_roots(self):
        # The whole point. SUDO_USER is set and HOME is root's, as sudo leaves them.
        import pwd
        me = pwd.getpwuid(os.getuid())
        # Point the launcher at a stub, then ask it to resolve OUR home from SUDO_USER.
        stub = self._stub_pulse(self.home)
        self._install(which=stub)
        result = self._run_launcher("/root", env={"SUDO_USER": me.pw_name})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"HOME=[{me.pw_dir}]", result.stdout,
                      "the launcher did not resolve the real user's home from SUDO_USER")
        self.assertNotIn("HOME=[/root]", result.stdout)

    def test_an_inherited_REAL_HOME_is_ignored(self):
        # /usr/local/bin/pulse is on every user's plain PATH, where SUDO_USER is unset.
        # Reading whatever REAL_HOME the caller exported would let it choose the home.
        self._stub_pulse(self.home)
        self._install()
        result = self._run_launcher(self.home, env={"REAL_HOME": "/somewhere-else"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"HOME=[{self.home}]", result.stdout)
        self.assertNotIn("somewhere-else", result.stdout,
                         "an exported REAL_HOME decided where Pulse looked")

    def test_it_refuses_rather_than_silently_using_roots_home(self):
        # No getent and no dscl -- a container, or macOS with a stripped PATH. Falling
        # back to root's HOME is the exact failure the launcher exists to prevent, and
        # it fails silently: Pulse then cannot find the user's install.
        import pwd
        self._stub_pulse(self.home)
        self._install()
        bare = os.path.join(self.tmp, "empty-bin")
        os.makedirs(bare, exist_ok=True)
        result = self._run_launcher("/root", path=bare,
                                    env={"SUDO_USER": pwd.getpwuid(os.getuid()).pw_name})
        self.assertNotEqual(result.returncode, 0,
                            "it ran with root's HOME instead of saying it could not tell")
        self.assertIn("home directory", result.stderr.lower())

    def test_running_it_twice_is_not_an_error(self):
        self._install()
        status, calls = self._install()
        self.assertEqual(status, 0)
        self.assertEqual(calls, {}, "it reinstalled over its own launcher")

    def test_it_refuses_to_overwrite_someone_elses_pulse(self):
        with open(self.target, "w", encoding="utf-8") as handle:
            handle.write("#!/bin/sh\n# a different pulse entirely\n")
        status, calls = self._install()
        self.assertEqual(status, 1)
        self.assertEqual(calls, {}, "it overwrote an unrelated /usr/local/bin/pulse")
        with open(self.target, encoding="utf-8") as handle:
            self.assertIn("a different pulse entirely", handle.read())

    def test_a_missing_user_install_is_reported_rather_than_linked_to(self):
        os.unlink(os.path.join(self.home, ".local", "bin", "pulse"))
        mock = self.mock
        empty = os.path.join(self.tmp, "no-system-pulse")
        os.makedirs(empty, exist_ok=True)
        with mock.patch.object(os.path, "expanduser", return_value=self.home), \
             mock.patch.object(console.shutil, "which", return_value=None), \
             mock.patch.object(console, "SECURE_PATH", (empty,)), \
             mock.patch.object(sys, "argv", ["pulse"]):
            self.assertEqual(console.install_sudo_launcher(self.target), 1)
        self.assertFalse(os.path.exists(self.target),
                         "it installed a launcher pointing at a pulse that is not there")

    def test_an_out_of_date_launcher_is_repointed(self):
        # Reinstalling Pulse somewhere else leaves the launcher aimed at a path that is
        # gone. Its own launcher is ours to update; anything else is not.
        self._install()
        moved = os.path.join(self.tmp, "elsewhere", "bin", "pulse")
        os.makedirs(os.path.dirname(moved))
        open(moved, "w").close()
        status, calls = self._install(which=moved)
        self.assertEqual(status, 0)
        self.assertTrue(calls, "the stale launcher was left pointing at a dead path")
        with open(self.target, encoding="utf-8") as handle:
            self.assertIn(moved, handle.read())

    def test_sudo_refusing_is_reported_and_leaves_nothing(self):
        status, _ = self._install(returncode=1)
        self.assertEqual(status, 1)
        self.assertFalse(os.path.exists(self.target))

    def test_the_command_is_reachable_from_the_cli(self):
        mock = self.mock
        from pulse import cli
        with mock.patch.object(console, "install_sudo_launcher", return_value=0) as installer:
            self.assertEqual(cli.main(["install-sudo"]), 0)
        self.assertTrue(installer.called, "`pulse install-sudo` did not reach the installer")

    def test_it_does_not_shadow_a_pulse_sudo_can_already_find(self):
        # /usr/local/bin comes before /usr/bin on every PATH. Installing over a
        # system-wide pulse would redirect it for everybody on the machine.
        mock = self.mock
        system_bin = os.path.join(self.tmp, "usr-bin")
        os.makedirs(system_bin)
        system_pulse = os.path.join(system_bin, "pulse")
        open(system_pulse, "w").close()
        os.chmod(system_pulse, 0o755)
        with mock.patch.object(os.path, "expanduser", return_value=self.home), \
             mock.patch.object(console, "SECURE_PATH", (system_bin,)), \
             mock.patch.object(console.shutil, "which",
                               return_value=os.path.join(self.home, ".local", "bin", "pulse")), \
             mock.patch.object(sys, "argv", ["pulse"]), \
             mock.patch("subprocess.run",
                        side_effect=AssertionError("shadowed a system-wide pulse")):
            self.assertEqual(console.install_sudo_launcher(self.target), 0)
        self.assertFalse(os.path.exists(self.target))

    def test_an_unreadable_target_is_reported_not_a_traceback(self):
        mock = self.mock
        os.makedirs(self.target)               # a directory where the file should go
        empty = os.path.join(self.tmp, "no-system-pulse")
        os.makedirs(empty, exist_ok=True)
        with mock.patch.object(os.path, "expanduser", return_value=self.home), \
             mock.patch.object(console.shutil, "which",
                               return_value=os.path.join(self.home, ".local", "bin", "pulse")), \
             mock.patch.object(console, "SECURE_PATH", (empty,)), \
             mock.patch.object(sys, "argv", ["pulse"]), \
             mock.patch("subprocess.run",
                        side_effect=AssertionError("installed over a directory")):
            self.assertEqual(console.install_sudo_launcher(self.target), 1)

    def test_the_refusal_does_not_suggest_the_command_that_just_failed(self):
        # `sudo pulse` is what the blocking file would run. Telling the user to use it
        # is circular; the long form works without any launcher.
        import io, contextlib
        mock = self.mock
        with open(self.target, "w", encoding="utf-8") as handle:
            handle.write("#!/bin/sh\n# something else\n")
        os.chmod(self.target, 0o755)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self._install()
        printed = buffer.getvalue()
        self.assertIn("sudo env", printed,
                      f"it did not offer the command that works:\n{printed}")

    def test_it_refuses_when_anybody_could_rewrite_the_target_pulse(self):
        # The launcher makes root exec that file. If anyone can write it, anyone
        # decides what root runs.
        mock = self.mock
        exposed_dir = os.path.join(self.tmp, "exposed")
        os.makedirs(exposed_dir)
        exposed = os.path.join(exposed_dir, "pulse")
        open(exposed, "w").close()
        os.chmod(exposed, 0o777)
        empty = os.path.join(self.tmp, "no-system-pulse")
        os.makedirs(empty, exist_ok=True)
        with mock.patch.object(os.path, "expanduser", return_value=self.home), \
             mock.patch.object(console.shutil, "which", return_value=exposed), \
             mock.patch.object(console, "SECURE_PATH", (empty,)), \
             mock.patch.object(sys, "argv", ["pulse"]), \
             mock.patch("subprocess.run",
                        side_effect=AssertionError("installed a world-writable target")):
            self.assertEqual(console.install_sudo_launcher(self.target), 1)


class AskingBeforeSudoTest(unittest.TestCase):
    """offer_sudo must not ask a question nobody can see."""

    def test_it_does_not_prompt_when_stdout_is_redirected(self):
        # Under redirect_stdout the prompt lands in a buffer while input() blocks on a
        # terminal that was never told anything was wanted: a test run hangs with no
        # output, and answering it would exec sudo over the test runner.
        import io, contextlib, unittest.mock as mock
        buffer = io.StringIO()
        with mock.patch.object(sys.stdin, "isatty", return_value=True), \
             mock.patch.object(console, "confirm",
                               side_effect=AssertionError("prompted into a buffer")), \
             mock.patch.object(os, "execvp",
                               side_effect=AssertionError("replaced the test runner")), \
             contextlib.redirect_stdout(buffer):
            self.assertFalse(console.offer_sudo(["train.py"]))
        self.assertIn("sudo", buffer.getvalue())

    def test_it_does_not_prompt_without_a_terminal_on_stdin(self):
        import io, contextlib, unittest.mock as mock
        with mock.patch.object(console, "confirm",
                               side_effect=AssertionError("prompted with no tty")), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(console.offer_sudo(["train.py"]))



class ProcessStartTimeTest(unittest.TestCase):
    """How old a process is has to come from the process, not from /proc's mtime.

    Pulse skips processes younger than a few seconds, on the grounds that they are
    probably helpers rather than the run being asked about. It measured that age with
    the mtime of /proc/<pid>, which is not the start time: on some kernels it tracks
    the last change to the directory and reads as "just now" for a process that has
    been training for hours. Every run then looked one second old and was skipped, so
    `sudo pulse train.py` reported nothing while train.py was plainly running --
    intermittently, depending on when the directory was last touched.
    """

    def setUp(self):
        if not os.path.isdir("/proc"):
            self.skipTest("Linux /proc only")
        self.tmp = tempfile.mkdtemp(prefix="starttime-check-")
        self.addCleanup(_remove_tree, self.tmp)

    def test_it_reads_the_real_start_time_of_this_process(self):
        boot = console._boot_time()
        self.assertIsNotNone(boot, "could not read btime from /proc/stat")
        started = console._process_started(os.getpid(), boot)
        self.assertIsNotNone(started, "could not read the start time of this process")
        # This test process is certainly not from the future, nor older than the boot.
        self.assertLessEqual(started, time.time() + 1)
        self.assertGreaterEqual(started, boot - 1)

    def test_the_answer_does_not_come_from_the_directory_mtime(self):
        import unittest.mock as mock
        boot = console._boot_time()
        real = console._process_started(os.getpid(), boot)
        # If mtime were the source, pretending it is "now" would change the answer.
        with mock.patch.object(os.path, "getmtime", return_value=time.time()):
            again = console._process_started(os.getpid(), boot)
        self.assertEqual(real, again, "the start time still comes from /proc mtime")

    def test_a_long_running_script_is_found_even_when_mtime_says_it_is_new(self):
        import unittest.mock as mock
        script = os.path.join(self.tmp, "longrun.py")
        with open(script, "w", encoding="utf-8") as handle:
            handle.write("import time\nfor _ in range(600):\n    time.sleep(0.1)\n")
        process = subprocess.Popen([sys.executable, script], cwd=self.tmp,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   start_new_session=True)
        self.addCleanup(_kill_group, process)
        time.sleep(6)                       # older than the "probably a helper" window

        with mock.patch.object(console, "discover", return_value=[]), \
             mock.patch.object(os.path, "getmtime", return_value=time.time()):
            found = console.unmonitored_python_processes()
        self.assertIn(process.pid, [p["pid"] for p in found],
                      "a process running for 6s was dropped as 'too new'")

    def test_a_process_that_really_is_new_is_still_skipped(self):
        import unittest.mock as mock
        script = os.path.join(self.tmp, "brandnew.py")
        with open(script, "w", encoding="utf-8") as handle:
            handle.write("import time\nfor _ in range(600):\n    time.sleep(0.1)\n")
        process = subprocess.Popen([sys.executable, script], cwd=self.tmp,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   start_new_session=True)
        self.addCleanup(_kill_group, process)
        with mock.patch.object(console, "discover", return_value=[]):
            found = console.unmonitored_python_processes()
        self.assertNotIn(process.pid, [p["pid"] for p in found],
                         "a process started moments ago was offered as a run to watch")

    def test_an_unreadable_start_time_lists_the_process_rather_than_hiding_it(self):
        # Hiding the run somebody is asking about is the worse failure.
        self.assertIsNone(console._process_started(999999, console._boot_time()))


class PulsesOwnProcessesTest(unittest.TestCase):
    """Telling Pulse's processes from the user's, without swallowing the user's.

    The test was `"pulse" in script`, which matched the whole path: every script in
    ~/pulse-experiments, or in a checkout of this repo, was invisible to
    `pulse <script>` -- the tool could not watch runs in its own source tree.
    """

    def test_pulses_own_modules_are_recognised(self):
        for script in ("/usr/lib/python3/site-packages/pulse/cli.py",
                       "/home/me/.local/lib/python3.11/site-packages/pulse/pulse.py",
                       "pulse_cli.py", "/opt/x/pulse_console.py"):
            self.assertTrue(console._is_pulse_itself(script), script)

    def test_a_users_script_in_a_pulse_named_directory_is_not(self):
        for script in ("/home/me/pulse-experiments/train.py",
                       "/home/me/pulseml/benchmarks/run_case.py",
                       "/tmp/pulse-test-1234/longrun.py",
                       "/home/me/my-pulse-project/finetune.py"):
            self.assertFalse(console._is_pulse_itself(script),
                             f"{script} would be hidden from `pulse <script>`")


class StaleSessionsDoNotHideALiveRunTest(unittest.TestCase):
    """Sessions outlive their runs, so by the second day there are several dead ones.

    Asking for train.py while train.py is running has one right answer, and it is never
    "choose between yesterday's two finished sessions". The earlier fix only covered a
    single stale session; with two or more, pick_session returned None and the console
    printed a menu of corpses instead of looking at the process.
    """

    def _run(self, sessions, processes, wanted="train.py"):
        import unittest.mock as mock
        called = {}

        def fake_watch_by_name(name, model=""):
            called["watched"] = name
            return 0

        with mock.patch.object(console, "discover", return_value=sessions), \
             mock.patch.object(console, "unmonitored_python_processes",
                               return_value=processes), \
             mock.patch.object(console, "watch_by_name", fake_watch_by_name), \
             mock.patch.object(console, "run_console",
                               side_effect=lambda s, *a, **k: called.setdefault("console", s) and 0):
            import io, contextlib
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                status = console.main([wanted])
        return status, called, buffer.getvalue()

    @staticmethod
    def _session(session_id, status, script="/home/me/train.py"):
        return {"session_id": session_id, "status": status, "script": script,
                "directory": "/tmp/" + session_id, "step": 10, "loss": 1.0,
                "last_seen": time.time() - 86400}

    @staticmethod
    def _process(pid, script="/home/me/train.py"):
        return {"pid": pid, "script": script, "cmdline": "python3 " + script,
                "started": time.time() - 600}

    def test_two_finished_sessions_do_not_hide_the_running_process(self):
        status, called, output = self._run(
            [self._session("yesterday-a", "finished"), self._session("yesterday-b", "ended")],
            [self._process(4242)])
        self.assertEqual(called.get("watched"), "train.py",
                         f"it offered a menu of finished runs instead:\n{output}")
        self.assertEqual(status, 0)

    def test_one_finished_session_still_does_not_hide_it(self):
        _, called, _ = self._run([self._session("yesterday-a", "finished")],
                                 [self._process(4242)])
        self.assertEqual(called.get("watched"), "train.py")

    def test_a_live_session_is_preferred_over_attaching_from_outside(self):
        # Pulse is already inside that run: reading it from outside would be worse.
        _, called, _ = self._run(
            [self._session("today", "live"), self._session("yesterday", "finished")],
            [self._process(4242)])
        self.assertNotIn("watched", called,
                         "it attached from outside to a run Pulse is already tracking")
        self.assertEqual((called.get("console") or {}).get("session_id"), "today")

    def test_with_no_process_running_it_still_reports_the_finished_ones(self):
        status, called, output = self._run(
            [self._session("yesterday-a", "finished"), self._session("yesterday-b", "ended")],
            [])
        self.assertNotIn("watched", called)
        self.assertEqual(status, 1)
        self.assertIn("Which one?", output)

    def test_attaching_to_a_run_that_is_over_says_so(self):
        """"healthy" about numbers that stopped moving yesterday reads as live.

        The detectors are right -- nothing is wrong with the data, it is just finished
        -- so the header has to carry that, or the whole screen looks like a live run.
        """
        import io, contextlib, unittest.mock as mock
        from pulse.pulse_brain import Brain

        session = self._session("yesterday", "ended")
        session["last_seen"] = time.time() - 86400
        buffer = io.StringIO()

        class FakeConsole:
            workdir = "/home/me"
            brain = mock.MagicMock(spec=Brain)

            def __init__(self, *a, **k):
                pass

            def start(self):
                pass

            def stop(self, join=False):
                pass

            def status_line(self):
                return "step 1,531 . healthy"

        with mock.patch.object(console, "Console", FakeConsole), \
             mock.patch.object(console, "input", create=True, side_effect=EOFError), \
             contextlib.redirect_stdout(buffer):
            console.run_console(session, [session])
        header = next(l for l in buffer.getvalue().splitlines() if l.startswith("Attached to"))
        self.assertIn("ended", header, f"a finished run looked live: {header!r}")

    def test_attaching_to_a_live_run_is_not_labelled(self):
        import io, contextlib, unittest.mock as mock
        from pulse.pulse_brain import Brain

        session = self._session("today", "live")
        buffer = io.StringIO()

        class FakeConsole:
            workdir = "/home/me"
            brain = mock.MagicMock(spec=Brain)

            def __init__(self, *a, **k):
                pass

            def start(self):
                pass

            def stop(self, join=False):
                pass

            def status_line(self):
                return "step 10 . healthy"

        with mock.patch.object(console, "Console", FakeConsole), \
             mock.patch.object(console, "input", create=True, side_effect=EOFError), \
             contextlib.redirect_stdout(buffer):
            console.run_console(session, [session])
        header = next(l for l in buffer.getvalue().splitlines() if l.startswith("Attached to"))
        self.assertNotIn("live", header, f"a live run was labelled anyway: {header!r}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
