"""Pulse has to import on a machine with no GUI toolkit.

tkinter is not a pip package: it comes with the interpreter on desktop installs and is
absent from most server and container images, including a plain `apt install python3`
box. Importing it at module scope made `import pulse` fail outright on exactly the
machines Pulse is written for -- a headless GPU box, a container, an SSH session --
even though those runs never open a window.
"""
import os
import shutil
import site
import subprocess
import sys
import tempfile
import textwrap
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")

# Refuse to import tkinter (and the PIL bridge that needs it), then import Pulse.
PRETEND_NO_TKINTER = textwrap.dedent("""\
    import sys

    class Blocker:
        BLOCKED = ("tkinter", "PIL.ImageTk")

        def find_spec(self, fullname, path=None, target=None):
            if fullname in self.BLOCKED or fullname.startswith("tkinter."):
                raise ImportError(f"No module named {fullname!r} (blocked by the test)")
            return None

    sys.meta_path.insert(0, Blocker())
    for name in list(sys.modules):
        if name == "tkinter" or name.startswith("tkinter."):
            del sys.modules[name]

    import pulse
    import pulse.pulse as core
    print("IMPORT_OK", pulse.__version__)
    print("HAS_TK", core.HAS_TK)
    print("MODE", core._determine_mode("auto"))
    print("AUTO_TRACK", callable(pulse.auto_track))
""")


def run(code):
    env = dict(os.environ,
               PYTHONPATH=os.pathsep.join(
                   p for p in (SRC, site.getusersitepackages(),
                               os.environ.get("PYTHONPATH", "")) if p),
               PULSE_LOGGING="0", NO_COLOR="1")
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=env, timeout=300)


class HeadlessImportTest(unittest.TestCase):
    def test_pulse_imports_without_tkinter(self):
        result = run(PRETEND_NO_TKINTER)
        self.assertIn("IMPORT_OK", result.stdout,
                      "import pulse failed without tkinter:\n" + result.stderr[-1500:])
        self.assertIn("HAS_TK False", result.stdout)
        self.assertIn("AUTO_TRACK True", result.stdout)

    def test_mode_detection_falls_back_to_cli_without_tkinter(self):
        result = run(PRETEND_NO_TKINTER)
        self.assertIn("MODE cli", result.stdout,
                      "auto mode picked a GUI it cannot draw:\n" + result.stdout)

    def test_tracking_a_run_works_without_tkinter(self):
        # The directory is made here rather than in the child, so it can be cleaned up:
        # a child that creates its own leaves it behind on every run.
        work = tempfile.mkdtemp(prefix="pulse-headless-")
        self.addCleanup(shutil.rmtree, work, ignore_errors=True)
        code = PRETEND_NO_TKINTER + textwrap.dedent(f"""\

            import glob, os, time
            work = {work!r}
            os.chdir(work)
            os.environ["PULSE_HOME"] = os.path.join(work, ".pulse")

            from pulse import pulse_monitor
            monitor = pulse_monitor.attach(script_path=os.path.join(work, "train.py"),
                                           interval=0.0)
            loss = 2.0
            for step in range(5):
                loss = loss * 0.9
                monitor.observe("loss", loss, step=step)
            pulse_monitor.detach()

            from pulse.pulse_brain import Brain
            brain = Brain(sorted(glob.glob(os.path.join(work, ".pulse_stream", "*")))[-1])
            brain.poll_once()
            print("READINGS", len(brain.histories.get("loss", [])))
        """)
        result = run(code)
        self.assertIn("READINGS 5", result.stdout,
                      "a run could not be tracked without tkinter:\n" + result.stderr[-1500:])

    def test_asking_for_the_dashboard_without_tkinter_says_why(self):
        code = PRETEND_NO_TKINTER + textwrap.dedent("""\

            import pulse.pulse as core
            try:
                core._chat_panel_class()
                print("NO_ERROR")
            except RuntimeError as exc:
                print("EXPLAINED", "tkinter" in str(exc), "python3-tk" in str(exc))
        """)
        result = run(code)
        self.assertIn("EXPLAINED True True", result.stdout,
                      "the GUI failure did not explain itself:\n" + result.stdout + result.stderr[-800:])


class QuietOutputTest(unittest.TestCase):
    """Importing Pulse must not put escape codes in the program's own output.

    pulse_cli replaces builtins.print for the whole process so a line Pulse is drawing
    in the background is not left half-overwritten. That is right on a terminal and
    wrong everywhere else: piped to a file, every line the user's script printed came
    out as "\\r\\033[KLINE".
    """

    def test_piped_output_has_no_escape_codes(self):
        result = run("import pulse\nprint('HELLO')\nprint('WORLD')\n")
        self.assertEqual(result.stdout, "HELLO\nWORLD\n",
                         "escape codes leaked into piped output: %r" % result.stdout)

    def test_output_to_a_file_has_no_escape_codes(self):
        work = tempfile.mkdtemp(prefix="pulse-quiet-")
        self.addCleanup(shutil.rmtree, work, ignore_errors=True)
        path = os.path.join(work, "out.txt")
        code = f"import pulse\nwith open({path!r}, 'w') as f:\n    print('LINE', file=f)\n"
        run(code)
        with open(path, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "LINE\n")

    def test_print_still_works_with_every_argument(self):
        result = run("import pulse\nprint('a', 'b', sep='-', end='!')\n")
        self.assertEqual(result.stdout, "a-b!")


if __name__ == "__main__":
    unittest.main(verbosity=2)
