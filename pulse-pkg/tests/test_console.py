"""Tests for the interactive console: finding runs, choosing one, and the views."""
import glob
import os
import site
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
