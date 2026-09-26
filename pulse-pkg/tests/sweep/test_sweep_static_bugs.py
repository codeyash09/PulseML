"""Static-analysis sweep: bugs found with pyflakes/pylint/mypy/ruff/vulture/bandit and
AST scans, each confirmed by reading the code and proven here with a failing test.

Every test_bug_* asserts the CORRECT behaviour, so it fails on the code as found.
test_ok_* tests are structural guards written generically so they keep protecting the
codebase (prompt placeholders, directive parsers, duplicate definitions, undefined names).

Run:  PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/sweep/test_sweep_static_bugs.py
"""
import ast
import glob
import importlib
import os
import re
import string
import subprocess
import sys
import threading
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(HERE)
if not os.path.isdir(os.path.join(_ROOT, "src", "pulse")):        # tests/sweep/ -> repo root
    _ROOT = os.path.dirname(_ROOT)
SRC = os.path.join(_ROOT, "src")
PKG = os.path.join(SRC, "pulse")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("PULSE_ASYNC_MODEL_CALLS", "0")
os.environ.setdefault("PULSE_LOGGING", "0")

from pulse import pulse_cli  # noqa: E402
from pulse.pulse_cli import PulseCLI  # noqa: E402

MODULE_FILES = sorted(p for p in glob.glob(os.path.join(PKG, "*.py"))
                      if not os.path.basename(p).startswith("test_"))


def _parse(path):
    with open(path, encoding="utf-8") as handle:
        return ast.parse(handle.read(), filename=path)


def _real_cli(tmp_path, **watch):
    cli = PulseCLI(watch_locals=dict(watch), pdf_dir=str(tmp_path / "pdfs"))
    cli.sensitivity = 0.3
    cli.epoch_scalar_histories = {}
    cli.batch_scalar_histories = {}
    return cli


# =====================================================================================
# 1. update() crashes with NameError whenever a tracked variable is None
# =====================================================================================

def test_bug_update_crashes_when_tracked_variable_is_none(tmp_path):
    """pulse_cli.py:11913 calls `_try_keras_history(...)` as a bare name, but the function
    is defined at pulse_cli.py:2804 as a *method* of PulseCLI (and without `self`, so even
    `self._try_keras_history` would be wrong). Any update() in which a tracked variable
    currently holds None -- `loss = None` before the first step, an optional metric, a
    Keras callback value not produced this epoch -- raises NameError out of Pulse into the
    user's training loop. Correct: the step is recorded, the None value is noted as
    unreadable, no exception."""
    cli = _real_cli(tmp_path, loss=None, acc=0.5)
    cli.tracked_vars = ["loss", "acc"]
    cli.continuous = True
    cli.update(step=1)          # NameError: name '_try_keras_history' is not defined
    assert cli.scalar_histories.get("loss") == [None]


def test_bug_try_keras_history_is_an_unbound_method_without_self():
    """PulseCLI._try_keras_history(var_name, watch_locals) is declared inside the class with
    no `self` and no @staticmethod: called as self._try_keras_history(name, locals) it would
    receive (self, name, locals) and raise TypeError. It must be a module-level function or a
    staticmethod so that the Keras-history fallback in update() can actually run."""
    raw = PulseCLI.__dict__.get("_try_keras_history")
    module_level = getattr(pulse_cli, "_try_keras_history", None)
    assert module_level is not None or isinstance(raw, staticmethod), (
        "_try_keras_history is neither a module function nor a staticmethod")


# =====================================================================================
# 2. CALC: is an unbounded, escapable eval() on model output
# =====================================================================================

def test_bug_calc_can_hang_the_training_thread():
    """_safe_eval_math (pulse_cli.py:1389, copy in pulse.py:315) evals the model's CALC:
    expression with no size/time bound. `CALC: 9**9**9` (a typo, or a model probing a
    magnitude) never returns: exponentiation of big ints runs for minutes and GBs, and the
    call sits on the training thread (_apply_directives) or the check-in worker. Correct:
    a bounded evaluator that rejects/limits huge powers and returns a calc error quickly."""
    code = ("import sys; sys.path.insert(0, %r)\n"
            "from pulse.pulse_cli import _safe_eval_math\n"
            "print(repr(_safe_eval_math('9**9**9'))[:80])\n") % SRC
    env = dict(os.environ, PULSE_LOGGING="0", CUDA_VISIBLE_DEVICES="")
    try:
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                              timeout=20, env=env)
    except subprocess.TimeoutExpired:
        pytest.fail("CALC: 9**9**9 was still evaluating after 20s")
    assert "calc error" in proc.stdout, proc.stdout + proc.stderr


def test_bug_calc_sandbox_escapes_to_shell(tmp_path):
    """_safe_eval_math's docstring says 'no builtins, no attribute access ... safe to
    eval() directly', but attribute access is not restricted at all: a CALC: line can walk
    ().__class__.__base__.__subclasses__() to os.system and run any command -- without the
    confirmation TERMINAL: applies to deleting files etc. The model reads user data and
    files, so prompt-injected text can do this. Correct: dunder attribute access (or any
    attribute access other than math.<fn>) is rejected."""
    marker = tmp_path / "pwned"
    expr = ("[c for c in ().__class__.__base__.__subclasses__() "
            "if c.__name__ == 'catch_warnings'][0]()._module.__builtins__['__import__']('os')"
            ".system('touch %s')" % marker)
    import warnings  # noqa: F401  (ensures catch_warnings is loaded, as it is in any real run)
    result = pulse_cli._safe_eval_math(expr)
    assert not marker.exists(), f"CALC ran a shell command (result {result!r})"
    assert "calc error" in str(result)


def test_ok_calc_still_does_arithmetic():
    assert pulse_cli._safe_eval_math("2 + 3 * 4") == 14
    assert abs(pulse_cli._safe_eval_math("math.log(10)") - 2.302585) < 1e-5
    assert abs(pulse_cli._safe_eval_math("sqrt(16)") - 4.0) < 1e-12


# =====================================================================================
# 3. Directive regexes run across line breaks
# =====================================================================================

def test_bug_bare_directive_swallows_the_next_line_of_the_diagnosis():
    """Every directive regex is `^\\s*NAME:\\s*(.*)$` with re.MULTILINE; `\\s*` matches a
    newline, so a bare flag directive the prompts tell the model to write on its own line
    ('DEPGRAPH:', 'CHANGELOG:', 'MLLINT:', 'SHAPETRACE:' ...) captures the NEXT line as its
    argument, and `rx.sub('', ...)` then deletes that line from the cleaned text. In PASS 2
    (pulse_cli.py:9193) the cleaned text *is* the diagnosis shown to the user and fed to
    the fix pass, so a sentence of the diagnosis silently disappears. Correct: the argument
    is '' and the next line survives."""
    answer = "DEPGRAPH:\nThe learning rate is 10x too high for Adam.\nFix: lower it."
    cleaned, requests = PulseCLI._extract_new_directives(answer)
    assert "The learning rate is 10x too high" in cleaned
    assert requests.get("depgraph") == [""]


def test_bug_bare_directive_turns_next_line_into_its_argument():
    """Same root cause: 'SHAPETRACE:' on its own line followed by the verdict makes Pulse run
    SHAPETRACE on a model called 'VERDICT: ok'; an empty 'GPUTRACK:' in the start-of-run
    reply makes Pulse try to GPU-track a variable named 'NEXTCHECK: 300'."""
    _clean, requests = PulseCLI._extract_new_directives("SHAPETRACE:\nVERDICT: ok")
    assert requests.get("shapetrace") == [""]
    parsed = PulseCLI._extract_directives("GPUTRACK:\nNEXTCHECK: 300")
    gputrack_names = parsed[3]
    assert "NEXTCHECK: 300" not in gputrack_names


# =====================================================================================
# 4. NORMAL_START: is silently a no-op with the default detector
# =====================================================================================

def test_bug_normal_start_baseline_never_reaches_detection_engine(tmp_path):
    """The prompts call NORMAL_START 'the ONLY way Pulse can catch a real explosion in the
    first few steps'. _apply_directives stores the estimate in cli._normal_start_baselines,
    but only _check_for_trouble_legacy (off unless PULSE_LEGACY_DETECTOR=1) reads it. The
    default path uses pulse_detect.DetectionEngine, which has set_baseline() and reads
    _baselines (pulse_detect.py:1107) -- and nothing ever calls set_baseline. So a loss that
    starts at ~100 when the code says it should start at ln(10)=2.3, then jumps to 400, is
    never flagged. Correct: the seeded baseline reaches the engine and the spike is raised."""
    cli = _real_cli(tmp_path)
    cli._apply_directives([], [], None, None, None, ["loss=2.3"])
    assert cli._normal_start_baselines == {"loss": 2.3}
    raised = []
    for epoch, value in enumerate([100, 101, 99, 100, 102, 100, 101, 400]):
        cli._record_keras_logs({"loss": value}, epoch=epoch)
        trouble = cli._check_for_trouble()
        if trouble:
            raised.append((epoch, trouble))
    assert cli._detection_engine()._baselines.get("loss") == 2.3
    assert raised, "the agent's NORMAL_START baseline had no effect on detection"


# =====================================================================================
# 5. Check-in prompt offers GPUTRACK/GPUUNTRACK as tools the check-in never services
# =====================================================================================

def _checkin_cli(replies):
    cli = PulseCLI.__new__(PulseCLI)
    cli.calls, cli.ran, cli.gputracked = [], [], []
    replies = list(replies)

    def call_model(message, max_tokens=0, *, system=None, history=None, purpose=None, timeout=None):
        cli.calls.append(message)
        return replies.pop(0)

    cli._call_model = call_model
    for name in ("_run_grep", "_run_view", "_run_terminal", "_run_trace", "_run_corr",
                 "_run_outlier", "_run_diffstats", "_run_histogram"):
        setattr(cli, name, (lambda n: lambda arg: cli.ran.append((n, arg)) or f"<{n} {arg}>")(name))
    cli._run_mllint = lambda: "<mllint>"
    cli._cmd_gputrack = lambda name, quiet=False: cli.gputracked.append(name) or name
    cli._cmd_gpuuntrack = lambda name, quiet=False: name
    cli.agent_history = []
    cli.auto_intervene = False
    cli.checkin_interval_steps = 500
    cli._checkin_note = ""
    return cli


def test_bug_checkin_gputrack_tool_line_is_dropped():
    """_CHECKIN_SYSTEM_PROMPT lists 'GPUTRACK: / GPUUNTRACK: <names>' under TOOLS ('put each
    on its own line. Results come back to you as the next message'), and tells the model its
    final answer must contain 'ONLY these lines, and no tool lines'. But
    _checkin_service_tools only services CALC/GREP/VIEW and _CHECKIN_TOOL_NAMES, and
    _finish_periodic_checkin reads GPUTRACK only from the final verdict answer. A GPUTRACK
    line sent as a tool (the way the prompt says) is silently discarded -- it is not even
    listed as 'refused'. Correct: the variable ends up GPU-tracked."""
    cli = _checkin_cli(["GPUTRACK: fc1_weight\nGREP: lr", "VERDICT: ok\nNEXTCHECK: 100\nCHECKNOTE: none"])
    answer, transcript = cli._run_checkin("the run")
    cli._finish_periodic_checkin(answer, None, "the run", transcript)
    assert "fc1_weight" in cli.gputracked


# =====================================================================================
# 6. Console holds the brain lock for a whole scheduled audit (a model call)
# =====================================================================================

def test_bug_console_freezes_while_a_scheduled_audit_runs(tmp_path):
    """Console._pump (pulse_console.py:437) wraps brain.poll_once() in self._brain_lock;
    poll_once runs brain.audit() when one is due -- a model call of up to 300s. Every view
    (/status via snapshot(), /findings, /vars, a question via ask(), /audit) takes the same
    lock, so the console is frozen for the whole audit, and Console.audit's 'an audit is
    already running' branch (brain._audit_lock, non-blocking) can never be reached: the
    person's /audit blocks behind the pump instead. Correct: views stay responsive while an
    audit's model call is in flight."""
    from pulse.pulse_console import Console

    entered, release = threading.Event(), threading.Event()

    def slow_agent(prompt):
        entered.set()
        release.wait(30)
        return '{"status": "ok", "next_check_minutes": 30, "reason": "fine"}'

    console = Console({"directory": str(tmp_path)}, agent=slow_agent)
    console.brain.schedule.next_at = 0.0          # an audit is due now
    console.start()
    try:
        assert entered.wait(10), "the scheduled audit never started"
        done = threading.Event()
        threading.Thread(target=lambda: (console.snapshot(), done.set()), daemon=True).start()
        responsive = done.wait(2.0)
    finally:
        release.set()
        console.stop(join=True)
    assert responsive, "/status blocked for the whole duration of a scheduled audit"


# =====================================================================================
# Structural guards (generic)
# =====================================================================================

def _format_calls():
    """(file, line, template_name, template, keywords) for every `NAME.format(k=...)` /
    `self.NAME.format(...)` whose template is a module- or class-level string."""
    out = []
    for path in MODULE_FILES:
        modname = "pulse." + os.path.basename(path)[:-3]
        module = importlib.import_module(modname)
        for node in ast.walk(_parse(path)):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "format"):
                continue
            target, template = node.func.value, None
            if isinstance(target, ast.Name):
                template = getattr(module, target.id, None)
            elif (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                  and target.value.id in ("self", "cls")):
                for obj in vars(module).values():
                    if isinstance(obj, type) and isinstance(getattr(obj, target.attr, None), str):
                        template = getattr(obj, target.attr)
                        break
            if not isinstance(template, str):
                continue
            keywords = {k.arg for k in node.keywords if k.arg}
            dynamic = bool(node.args) or any(k.arg is None for k in node.keywords)
            out.append((os.path.basename(path), node.lineno, ast.unparse(target), template,
                        keywords, dynamic))
    return out


def test_ok_every_prompt_format_call_supplies_exactly_its_placeholders():
    """Every .format() on a prompt template must supply each placeholder (KeyError at the
    worst moment otherwise), pass nothing the template ignores (a sign the template lost a
    field), and the template must format cleanly (a literal '{' in a JSON example must be
    doubled)."""
    calls = _format_calls()
    assert calls, "found no template.format() call sites -- the scan is broken"
    problems = []
    for fname, line, name, template, keywords, dynamic in calls:
        if dynamic:
            continue
        try:
            fields = {f.split(".")[0].split("[")[0]
                      for _, f, _, _ in string.Formatter().parse(template) if f is not None}
        except ValueError as exc:
            problems.append(f"{fname}:{line} {name}: unparsable template ({exc})")
            continue
        if fields - keywords:
            problems.append(f"{fname}:{line} {name}: missing {sorted(fields - keywords)}")
        if keywords - fields:
            problems.append(f"{fname}:{line} {name}: unused {sorted(keywords - fields)}")
        for dummy in ("x", 1.5):
            try:
                template.format(**{k: dummy for k in keywords})
                break
            except (ValueError, TypeError):
                continue
            except (KeyError, IndexError) as exc:
                problems.append(f"{fname}:{line} {name}: raises {type(exc).__name__} {exc}")
                break
        else:
            problems.append(f"{fname}:{line} {name}: cannot be formatted with any dummy value")
    assert not problems, "\n".join(problems)


def test_ok_no_class_or_module_defines_the_same_name_twice():
    """A second `def` with the same name silently replaces the first (the first becomes
    dead code). Properties' setter/deleter and @overload are exempt."""
    problems = []

    def scan(body, where, path):
        seen = {}
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                decorators = [ast.unparse(d) for d in getattr(node, "decorator_list", [])]
                if any(d.endswith((".setter", ".deleter")) or d.endswith("overload") for d in decorators):
                    continue
                if node.name in seen:
                    problems.append(f"{os.path.basename(path)}:{node.lineno} {where}.{node.name} "
                                    f"redefines line {seen[node.name]}")
                seen[node.name] = node.lineno
                if isinstance(node, ast.ClassDef):
                    scan(node.body, f"{where}.{node.name}", path)

    for path in MODULE_FILES:
        scan(_parse(path).body, os.path.basename(path)[:-3], path)
    assert not problems, "\n".join(problems)


def test_ok_every_method_takes_self_or_is_static():
    """A method whose first parameter is not self/cls (and that is not a @staticmethod) is
    unusable as a method -- see _try_keras_history."""
    problems = []
    for path in MODULE_FILES:
        for cls in [n for n in ast.walk(_parse(path)) if isinstance(n, ast.ClassDef)]:
            for fn in cls.body:
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                decorators = {ast.unparse(d) for d in fn.decorator_list}
                if "staticmethod" in decorators:
                    continue
                params = fn.args.posonlyargs + fn.args.args
                first = params[0].arg if params else (fn.args.vararg.arg if fn.args.vararg else None)
                if first not in ("self", "cls", "mcs", "args"):
                    problems.append(f"{os.path.basename(path)}:{fn.lineno} {cls.name}.{fn.name}"
                                    f"({first}, ...)")
    assert not problems, "\n".join(problems)


def test_ok_no_undefined_names():
    """pyflakes 'undefined name' is always a latent NameError. (Also covers pulse.py's
    `linecache`, whose NameError is swallowed by a bare except so its debug trace never
    logs anything.)"""
    pyflakes = pytest.importorskip("pyflakes.api")
    from pyflakes import reporter as _reporter
    import io
    out, err = io.StringIO(), io.StringIO()
    for path in MODULE_FILES:
        pyflakes.checkPath(path, _reporter.Reporter(out, err))
    undefined = [line for line in out.getvalue().splitlines() if "undefined name" in line]
    assert not undefined, "\n".join(undefined)


_TOOL_LINE = re.compile(r"(?m)^\s{2,}([A-Z][A-Z_]{2,}):")


def _advertised(prompt):
    names = set(_TOOL_LINE.findall(prompt))
    names |= set(re.findall(r"\b([A-Z][A-Z_]{2,}): /", prompt))           # 'GPUTRACK: / GPUUNTRACK:'
    names |= set(re.findall(r"/ ([A-Z][A-Z_]{2,}):", prompt))
    return names


def test_ok_every_tool_the_debugger_prompt_advertises_is_parsed():
    """Every NAME: tool in SYSTEM_PROMPT has a parser the pipeline uses."""
    parsed = {rx.pattern.split("*")[1].split(":")[0] for rx in PulseCLI._NEW_DIRECTIVE_RES.values()}
    parsed |= {"CALC", "PROMOTE", "GPUTRACK", "GPUUNTRACK", "SENSITIVITY", "NORMAL_START", "GREP", "VIEW"}
    missing = _advertised(pulse_cli.SYSTEM_PROMPT) - parsed
    assert not missing, f"advertised but never parsed: {sorted(missing)}"


def test_ok_every_tool_the_checkin_prompt_advertises_is_serviced():
    """Every tool _CHECKIN_SYSTEM_PROMPT offers must be run by _checkin_service_tools
    (results come back 'as the next message'). Currently GPUTRACK/GPUUNTRACK are not."""
    serviced = {"CALC", "GREP", "VIEW"} | {n.upper() for n in PulseCLI._CHECKIN_TOOL_NAMES}
    advertised = _advertised(PulseCLI._CHECKIN_SYSTEM_PROMPT) - {"VERDICT", "PROBLEM", "NEXTCHECK",
                                                                  "CHECKNOTE", "TOOLS"}
    missing = advertised - serviced
    assert not missing, f"offered to the check-in but never serviced: {sorted(missing)}"


def test_ok_every_tool_the_code_agent_prompt_advertises_is_allowed():
    from pulse import pulse_code
    allowed = {n.upper() for n in pulse_code._ALLOWED_TOOLS} | {"GREP", "VIEW"}
    missing = _advertised(pulse_code.CODE_SYSTEM_PROMPT) - allowed
    assert not missing, f"offered to Pulse Code but filtered out: {sorted(missing)}"


def test_ok_start_prime_reply_fields_are_all_applied():
    """Each field the start-of-run prompt asks for is read by _finish_start_prime."""
    prompt = PulseCLI._START_PRIME_PROMPT
    fields = set(re.findall(r"(?m)^([A-Z_]+): <", prompt))
    assert fields == {"SENSITIVITY", "NORMAL_START", "GPUTRACK", "NEXTCHECK", "CHECKNOTE"}
    answer = ("SENSITIVITY: 0.4\nNORMAL_START: loss=2.3\nGPUTRACK: none\n"
              "NEXTCHECK: 250\nCHECKNOTE: watch val_loss at train.py:40")
    parsed = PulseCLI._extract_directives(answer)
    assert parsed[5] == ["0.4"] and parsed[6] == ["loss=2.3"] and parsed[3] == []
    assert PulseCLI._parse_nextcheck_steps(answer) == 250
    assert PulseCLI._CHECKNOTE_RE.search(answer).group(1).strip() == "watch val_loss at train.py:40"
