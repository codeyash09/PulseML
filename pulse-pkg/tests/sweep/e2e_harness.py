"""Shared runner for the end-to-end sweep tests (tests/sweep/test_sweep_e2e_*.py).

Runs a real training script in a subprocess, the way a user would, and reports what
happened: exit code, stdout/stderr, wall time, every (fake) LLM call Pulse made, the
files left in the run directory, and whether the script itself was changed.

Modes:
  plain   python train.py                         (no Pulse at all: the baseline)
  direct  python train.py  (the script calls auto_track() itself)
  run     python -m pulse run train.py
  stream  python -m pulse run --stream train.py
"""
import json
import os
import signal
import site
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
FAKE_LLM_DIR = os.path.join(HERE, "e2e_fake_llm")


def _src_dir():
    env = os.environ.get("PULSE_SRC")
    if env:
        return env
    for candidate in (os.path.join(os.path.dirname(os.path.dirname(HERE)), "src"),   # tests/sweep/
                      os.path.join(os.path.dirname(HERE), "src")):                    # tests_e2e/
        if os.path.isdir(os.path.join(candidate, "pulse")):
            return candidate
    raise RuntimeError("cannot find Pulse's src/ -- set PULSE_SRC")


SRC = _src_dir()
_INSTALLED = {}


def installed_src():
    """A copy of the package under a directory named site-packages, the way a normal
    `pip install pulseml` lays it out (Pulse's tracer skips site-packages frames)."""
    if "dir" not in _INSTALLED:
        import shutil
        base = tempfile.mkdtemp(prefix="pulse-installed-")
        target = os.path.join(base, "lib", "python3", "site-packages")
        shutil.copytree(os.path.join(SRC, "pulse"), os.path.join(target, "pulse"),
                        ignore=shutil.ignore_patterns("__pycache__"))
        _INSTALLED["dir"] = target
    return _INSTALLED["dir"]


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    wall: float
    workdir: str
    script_path: str
    script_before: str
    script_after: str
    llm_calls: list = field(default_factory=list)
    files: list = field(default_factory=list)
    timed_out: bool = False

    @property
    def modified(self):
        return self.script_before != self.script_after

    def calls_about(self, text):
        return [c for c in self.llm_calls if text in c.get("all", c.get("last_user", ""))]

    def show(self, n=3000):
        return (f"rc={self.returncode} wall={self.wall:.1f}s timed_out={self.timed_out}\n"
                f"--- stdout ---\n{self.stdout[-n:]}\n--- stderr ---\n{self.stderr[-n:]}\n"
                f"--- files ---\n{self.files}\n--- llm calls ---\n"
                + "\n".join(c.get("last_user", "")[:200].replace("\n", " | ") for c in self.llm_calls))


def _listing(root):
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        if rel.startswith("home"):
            continue
        for d in dirnames:
            if os.path.join(rel, d).startswith("home"):
                continue
            out.append(os.path.normpath(os.path.join(rel, d)) + "/")
        for f in filenames:
            out.append(os.path.normpath(os.path.join(rel, f)))
    return sorted(out)


def _default_signals():
    # A test runner started in the background (nohup ... &) has SIGINT ignored, and a child
    # inherits that: Ctrl+C tests would then test nothing. Give the child a normal SIGINT.
    signal.signal(signal.SIGINT, signal.SIG_DFL)


def run(source, mode="direct", args=(), *, name="train.py", subdir="", extra_files=None,
        env=None, timeout=240, llm_rules=None, llm_reply=None, llm_delay=None,
        autofix=True, agent=True, stdin=None, module=None, workdir=None, pulse_opts=(),
        send_signal_after=None, installed=False):
    """Run `source` as a script. Returns a RunResult."""
    workdir = workdir or tempfile.mkdtemp(prefix="pulse-e2e-")
    script_dir = os.path.join(workdir, subdir) if subdir else workdir
    os.makedirs(script_dir, exist_ok=True)
    script_path = os.path.join(script_dir, name)
    source = textwrap.dedent(source)
    with open(script_path, "w", encoding="utf-8") as fh:
        fh.write(source)
    for rel, text in (extra_files or {}).items():
        p = os.path.join(workdir, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(textwrap.dedent(text))

    home = os.path.join(workdir, "home")
    os.makedirs(home, exist_ok=True)
    log = os.path.join(home, "llm_calls.jsonl")
    config = os.path.join(home, "pulse_config.json")
    with open(config, "w", encoding="utf-8") as fh:
        json.dump({"agent": "openrouter/deepseek/deepseek-v4.1-flash", "api_key": "sk-or-v1-fake",
                   "autofix": autofix, "telemetry": False, "pdfs": False}, fh)
    user_site = site.getusersitepackages()
    src = installed_src() if installed else SRC
    paths = [FAKE_LLM_DIR, src, user_site, os.environ.get("PYTHONPATH", "")]
    if mode == "plain":
        paths = [src, user_site, os.environ.get("PYTHONPATH", "")]
    e = dict(os.environ)
    for k in list(e):
        if k.startswith("PULSE_") and k != "PULSE_SRC":
            del e[k]
    e.update(PYTHONPATH=os.pathsep.join(p for p in paths if p), HOME=home,
             PULSE_TELEMETRY="off", PULSE_NONINTERACTIVE="1", NO_COLOR="1",
             CUDA_VISIBLE_DEVICES="", TF_CPP_MIN_LOG_LEVEL="3", PYTHONUNBUFFERED="1",
             OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    if agent:
        e["PULSE_CONFIG"] = config
    if mode != "plain":
        e["FAKE_LLM_LOG"] = log
    if llm_rules is not None:
        e["FAKE_LLM_RULES"] = json.dumps(llm_rules)
    if llm_reply is not None:
        e["FAKE_LLM_REPLY"] = llm_reply
    if llm_delay is not None:
        e["FAKE_LLM_DELAY"] = str(llm_delay)
    e.update(env or {})

    if module:
        target = ["-m", module]
    else:
        target = [os.path.relpath(script_path, workdir)]
    if mode in ("plain", "direct"):
        argv = [sys.executable] + target + list(args)
    elif mode == "run":
        argv = [sys.executable, "-m", "pulse", "run", *pulse_opts] + target + list(args)
    elif mode == "stream":
        argv = [sys.executable, "-m", "pulse", "run", "--stream", *pulse_opts] + target + list(args)
    else:
        raise ValueError(mode)

    t0 = time.monotonic()
    proc = subprocess.Popen(argv, cwd=workdir, env=e, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                            text=True, start_new_session=True, preexec_fn=_default_signals)
    timed_out = False
    try:
        if send_signal_after is not None:
            sig, after = send_signal_after
            try:
                out, err = proc.communicate(input=stdin, timeout=after)
            except subprocess.TimeoutExpired:
                proc.send_signal(sig)
                out, err = proc.communicate(timeout=timeout)
        else:
            out, err = proc.communicate(input=stdin, timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:
            proc.kill()
        out, err = proc.communicate()
    wall = time.monotonic() - t0
    calls = []
    if os.path.exists(log):
        with open(log, encoding="utf-8") as fh:
            calls = [json.loads(l) for l in fh if l.strip()]
    with open(script_path, encoding="utf-8") as fh:
        after = fh.read()
    return RunResult(proc.returncode, out, err, wall, workdir, script_path, source, after,
                     calls, _listing(workdir), timed_out)
