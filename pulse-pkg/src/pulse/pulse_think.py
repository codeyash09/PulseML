"""
Think mode -- work until the error is fixed or the feature is done.

Off by default. Turn it on with `/think on`, `--think`, `PULSE_THINK=1`, or `"think": true` in
pulse_config.json. Both the debugger and Pulse Code use this one engine; each supplies a small
adapter for what differs (its context, its tools, its edit format and how an edit is applied).

What it does, in order:

  1. OUTLINE -- the agent's first pass is an action plan, before any code is written. It may
     read the project with the (read-only) tools while it plans.
  2. BUDGET -- the agent is then asked to size the work: how many steps, and for each step how
     many tool calls it wants BEFORE writing it and how many verifications it wants AFTER the
     step has been integrated (plus how many for the whole job at the end). Pulse holds it to
     those numbers -- within hard caps -- rather than to a fixed pipeline.
  3. EXECUTE -- step by step: the tool calls, then the edit (through the existing applier and
     its lint gate), then the verifications. A step whose edit or verification fails gets a
     bounded number of revisions; if it still fails the plan is revised (a replan).
  4. DONE -- only when a final check says the goal is achieved. If it says something remains,
     the agent replans the remainder and continues.

Nothing here writes a file itself: every edit goes through the adapter's applier, so the lint
gate, the history log and /undo work exactly as they do without think mode.

Because "until done" can mean many model calls, there are hard limits, all overridable:
PULSE_THINK_MAX_STEPS (30), PULSE_THINK_MAX_REPLANS (4), PULSE_THINK_MAX_MINUTES (45) and
PULSE_THINK_MAX_CALLS (250). The plan and an estimate of the calls it may take are shown up
front, and Ctrl+C stops it cleanly with whatever steps already landed left applied.
"""
import os
import re
import time

from . import pulse_ui as _ui
from .pulse_cli import (
    AgentRequestFailed,
    PulseCLI,
    _AGENT_MAX_TOKENS,
    _GREEN,
    _RED,
    _Spinner,
    _YELLOW,
    cprint,
)

# ---------------------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------------------

def _env_number(name, default, cast=int):
    try:
        return cast(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


MAX_STEPS = _env_number("PULSE_THINK_MAX_STEPS", 30)
MAX_REPLANS = _env_number("PULSE_THINK_MAX_REPLANS", 4)
MAX_MINUTES = _env_number("PULSE_THINK_MAX_MINUTES", 45, float)
MAX_CALLS = _env_number("PULSE_THINK_MAX_CALLS", 250)

MAX_TOOL_CALLS_PER_STEP = 8         # what the agent may ask for; Pulse clamps to this
MAX_VERIFICATIONS_PER_STEP = 4
STEP_REVISIONS = 2                  # corrections of one step before giving up on it
OUTLINE_TOOL_ROUNDS = 3


def is_on(cli):
    return bool(getattr(cli, "think", False))


# ---------------------------------------------------------------------------------------
# Prompts. Every call is self-contained: the model is sent the current code each time, because
# an earlier step has usually changed it and only the last few chat messages are kept.
# ---------------------------------------------------------------------------------------

_OUTLINE = (
    "ACTION PLAN. Before writing any code, outline your plan for this job:\n{goal}\n\n"
    "Say what you will look at, the sequence of changes you will make, and how you will know each "
    "one worked and that the whole job is done. You may read the project first with the tools "
    "(one directive per line; I will run them and return the results). Do not write code yet."
    "{extra}\n\nCurrent state:\n{context}"
)

_BUDGET = (
    "Now size the work. Given your plan, respond with ONLY this JSON object -- no prose, no fences:\n"
    '{{"steps": <how many steps>, "plan": [{{"title": "short name", "goal": "what is true when this '
    'step is done", "tool_calls": <0-{max_tools}: read-only tool calls you want BEFORE writing this '
    'step>, "verifications": <0-{max_verif}: checks to run AFTER this step is integrated, before '
    'moving on>}}, ...], "final_verifications": <0-{max_verif}: checks that the WHOLE job is done>}}\n'
    "Rules: 'steps' must equal the length of 'plan'. One step is one integrable change -- code that "
    "leaves the project working when it is done. Size the plan to the job: a one-line fix is one "
    "step, a feature across several files is many. Ask for tool calls where you genuinely need to "
    "look something up, and for more verifications where a step is risky or touches shared code."
)

_STEP_TOOLS = (
    "Step {i} of {n}: {title}\nGoal of this step: {step_goal}\n\n"
    "You may make up to {remaining} more read-only tool call(s) before writing this step. Request "
    "them now (one directive per line), or reply with only the word READY if you already have what "
    "you need.\n\nThe plan so far:\n{plan}\n\nProgress so far:\n{progress}\n\nNotes gathered for "
    "this step:\n{notes}\n\nCurrent state:\n{context}"
)

_STEP_IMPLEMENT = (
    "Step {i} of {n}: {title}\nGoal of this step: {step_goal}\n\n"
    "The overall job:\n{goal}\n\nThe plan:\n{plan}\n\nProgress so far:\n{progress}\n\n"
    "Notes gathered for this step:\n{notes}\n\n"
    "Implement ONLY this step -- the smallest change that makes its goal true and leaves everything "
    "working. Earlier steps are already in the code you are shown below.\n\n{rules}"
    "{revise}\n\nCurrent state:\n{context}"
)

_STEP_REVISE = (
    "\n\nYour previous attempt at this step did not work:\n{reason}\n"
    "Correct it. Respond with the complete corrected change for this step."
)

_FORMAT_REMINDER = (
    "\n\nYour last reply was not the change object described above. Respond with ONLY that object."
)

_LENSES = (
    "Correctness: does the change do what this step says, and is it right? Trace it against the code.",
    "Integration: with the change in place, do imports, names, signatures, callers and other files "
    "still line up? Look for anything the change should have updated elsewhere and did not.",
    "Scope: is the change as small as the step needs? Is anything unrelated touched, removed or "
    "left half-done?",
    "Completeness: is anything this step promised still missing, or any edge case it obviously misses?",
)

_STEP_VERIFY = (
    "Verification {j} of {m} for step {i}: {title}\nGoal of this step: {step_goal}\n\n"
    "The change that was just integrated:\n{changes}\n\n"
    "Check ({lens})\n\nRespond with ONLY "
    '{{"passes": true or false, "reason": "one sentence"}}.\n\nCurrent state (change included):\n{context}'
)

_FINAL_CHECK = (
    "Final check {j} of {m}. The job:\n{goal}\n\nThe plan:\n{plan}\n\nProgress:\n{progress}\n\n"
    "Looking at the current code, is the job fully done -- {criterion}? Respond with ONLY "
    '{{"done": true or false, "remaining": "what is still missing, or empty"}}.\n\nCurrent state:\n{context}'
)

_REPLAN = (
    "The plan needs revising.\nReason: {reason}\n\nThe job:\n{goal}\n\nThe plan so far:\n{plan}\n\n"
    "Progress (what has already landed):\n{progress}\n\nRespond with ONLY the same JSON object as "
    "before, but covering just the REMAINING work -- {{\"steps\": n, \"plan\": [...], "
    '"final_verifications": n}}. If nothing remains, use {{"steps": 0, "plan": [], "final_verifications": 0}}.'
    "\n\nCurrent state:\n{context}"
)

_CRITERIA = (
    "the error is actually gone and nothing else was broken getting there",
    "every part of the request is implemented and wired together",
)


# ---------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------

class Step:
    def __init__(self, title, goal, tool_calls, verifications):
        self.title = title
        self.goal = goal
        self.tool_calls = tool_calls
        self.verifications = verifications


class Applied:
    """What an adapter reports after trying to apply one edit."""
    def __init__(self, ok, message="", problems=""):
        self.ok = ok
        self.message = message
        self.problems = problems


class Outcome:
    def __init__(self):
        self.done = False
        self.declined = False
        self.request_failed = None
        self.stopped = ""
        self.steps_landed = 0
        self.steps_total = 0
        self.calls = 0
        self.last_fix = None
        self.summary = ""


class _Stop(Exception):
    """A hard limit was reached; the reason is the message."""


def _clamp(value, lo, hi, default):
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return default


def parse_budget(cli, text):
    """(steps, final_verifications) from the budget JSON, clamped, or None if unusable."""
    obj = cli._parse_json_obj(text)
    if not isinstance(obj, dict):
        return None
    raw_steps = obj.get("plan")
    if raw_steps is None and isinstance(obj.get("steps"), list):
        raw_steps = obj["steps"]
    if not isinstance(raw_steps, list):
        return None
    steps = []
    for index, item in enumerate(raw_steps, 1):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or item.get("goal") or f"Step {index}").strip()[:110]
        goal = str(item.get("goal") or title).strip()[:400]
        steps.append(Step(title, goal,
                          _clamp(item.get("tool_calls"), 0, MAX_TOOL_CALLS_PER_STEP, 2),
                          _clamp(item.get("verifications"), 0, MAX_VERIFICATIONS_PER_STEP, 1)))
    final = _clamp(obj.get("final_verifications"), 0, MAX_VERIFICATIONS_PER_STEP, 1)
    return steps, final


# ---------------------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------------------

_DIRECTIVE_NAMES = None


def _directive_re():
    global _DIRECTIVE_NAMES
    if _DIRECTIVE_NAMES is None:
        names = {"GREP", "VIEW"} | {k.upper() for k in PulseCLI._NEW_DIRECTIVE_RES}
        _DIRECTIVE_NAMES = re.compile(r"(?m)^[ \t]*(?:%s)[ \t]*:.*$" % "|".join(sorted(names)))
    return _DIRECTIVE_NAMES


class Engine:
    def __init__(self, cli, adapter):
        self.cli = cli
        self.adapter = adapter
        self.calls = 0
        self.started = time.monotonic()
        self.plan_text = ""
        self.log = []                  # ["1. title -- what changed", ...] for steps that landed
        self.outcome = Outcome()
        self.no_change = False
        self.pending = None            # (step number, title) of an applied edit not yet through verification

    # -- a model call, with the hard limits ----------------------------------------------
    def ask(self, prompt, label):
        if self.calls >= MAX_CALLS:
            raise _Stop(f"reached the model-call limit ({MAX_CALLS}; PULSE_THINK_MAX_CALLS)")
        if (time.monotonic() - self.started) / 60.0 > MAX_MINUTES:
            raise _Stop(f"reached the time limit ({MAX_MINUTES:g} min; PULSE_THINK_MAX_MINUTES)")
        self.calls += 1
        with _Spinner(label):
            return self.cli._call_model(prompt, max_tokens=_AGENT_MAX_TOKENS)

    def progress(self):
        return "\n".join(self.log) if self.log else "(nothing done yet)"

    # -- output --------------------------------------------------------------------------
    def _heading(self, text):
        if _ui.enabled():
            _ui.header(text)
        else:
            print(f"\n=== {text} ===")

    def _line(self, kind, text):
        if _ui.enabled():
            {"ok": lambda t: _ui.ok(t), "warn": _ui.warn, "fail": _ui.fail, "note": _ui.note}[kind](text)
        else:
            print({"ok": "✓ ", "warn": "⚠ ", "fail": "✕ ", "note": "  "}[kind] + text)

    def _show_plan(self, steps, final_v):
        self._heading("Plan")
        for i, s in enumerate(steps, 1):
            print(f"  ○ {i}. {s.title}")
            print(f"       tools ≤ {s.tool_calls}  ·  verify × {s.verifications}")
        est = 2 + sum(s.tool_calls + 1 + s.verifications for s in steps) + final_v
        print(f"\n  {len(steps)} step(s), final verification × {final_v}  ·  up to ~{est} model calls "
              f"(more if a step needs correcting)\n")

    # -- 1. outline ----------------------------------------------------------------------
    def outline(self):
        extra = self.adapter.outline_extra
        prompt = _OUTLINE.format(goal=self.adapter.goal, extra=extra, context=self.adapter.context())
        answer = self.ask(prompt, "Outlining the plan")
        notes_so_far = ""
        for _ in range(OUTLINE_TOOL_ROUNDS):
            lines = _directive_re().findall(answer)
            if not lines:
                break
            result = self.adapter.service_tools("\n".join(lines))
            if not result:
                break
            print(f"\n[tool results]\n{result}\n")
            notes_so_far += f"\n\nTool results so far:\n{result}"
            answer = self.ask(prompt + notes_so_far + "\n\nNow give the plan, or use more tools.", "Outlining the plan")
        text = self.adapter.plain_text(answer)
        self.plan_text = text
        marker = getattr(self.adapter, "no_change_re", None)
        self.no_change = bool(marker and marker.search(answer))
        self._heading("Answer" if self.no_change else "Action plan")
        print(f"{text}\n")

    # -- 2. budget -----------------------------------------------------------------------
    def budget(self, replan_reason=None):
        if replan_reason is None:
            prompt = _BUDGET.format(max_tools=MAX_TOOL_CALLS_PER_STEP, max_verif=MAX_VERIFICATIONS_PER_STEP)
            prompt = f"Your plan:\n{self.plan_text}\n\n{prompt}\n\nCurrent state:\n{self.adapter.context()}"
        else:
            prompt = _REPLAN.format(reason=replan_reason, goal=self.adapter.goal, plan=self.plan_text,
                                    progress=self.progress(), context=self.adapter.context())
        for attempt in range(2):
            parsed = parse_budget(self.cli, self.ask(prompt, "Sizing the work"))
            if parsed is not None:
                return parsed
            prompt += "\n\nThat was not the JSON object described. Respond with ONLY that object."
        return None

    # -- 3. one step ---------------------------------------------------------------------
    def tool_phase(self, i, n, step):
        remaining, notes = step.tool_calls, []
        while remaining > 0:
            prompt = _STEP_TOOLS.format(i=i, n=n, title=step.title, step_goal=step.goal, remaining=remaining,
                                        plan=self.plan_text, progress=self.progress(),
                                        notes="\n\n".join(notes) or "(none yet)", context=self.adapter.context())
            answer = self.ask(prompt, f"Step {i}: looking things up")
            lines = _directive_re().findall(answer)
            if not lines:
                break
            lines = lines[:remaining]
            result = self.adapter.service_tools("\n".join(lines))
            remaining -= len(lines)
            if result:
                print(f"\n[tool results]\n{result}\n")
                notes.append(result)
        return "\n\n".join(notes) or "(none)"

    def implement(self, i, n, step, notes, revise_reason=None):
        revise = _STEP_REVISE.format(reason=revise_reason) if revise_reason else ""
        prompt = _STEP_IMPLEMENT.format(
            i=i, n=n, title=step.title, step_goal=step.goal, goal=self.adapter.goal, plan=self.plan_text,
            progress=self.progress(), notes=notes, rules=self.adapter.format_rules, revise=revise,
            context=self.adapter.context())
        label = f"Step {i}: writing the change" if not revise_reason else f"Step {i}: correcting"
        fix = self.adapter.parse(self.ask(prompt, label))
        if fix is None:
            fix = self.adapter.parse(self.ask(prompt + _FORMAT_REMINDER, label))
        return fix

    def verify(self, i, n, step, fix):
        for j in range(1, step.verifications + 1):
            prompt = _STEP_VERIFY.format(j=j, m=step.verifications, i=i, title=step.title, step_goal=step.goal,
                                         changes=self.adapter.describe(fix), lens=_LENSES[(j - 1) % len(_LENSES)],
                                         context=self.adapter.context())
            verdict = self.cli._parse_json_obj(self.ask(prompt, f"Step {i}: verifying ({j}/{step.verifications})"))
            if isinstance(verdict, dict) and verdict.get("passes") is False:
                return False, str(verdict.get("reason") or "the check did not pass").strip()
        return True, ""

    def run_step(self, i, n, step):
        self._heading(f"Step {i} of {n}")
        print(f"  {step.title}\n")
        notes = self.tool_phase(i, n, step)
        fix = self.implement(i, n, step, notes)
        reason = ""
        for attempt in range(STEP_REVISIONS + 1):
            if fix is None:
                reason = "the model did not produce a usable change"
            else:
                applied = self.adapter.apply(fix, step)
                if applied.ok:
                    self.outcome.last_fix = fix
                    self.pending = (i, step.title)
                    ok, why = self.verify(i, n, step, fix)
                    if ok:
                        self.pending = None
                        self.log.append(f"{i}. {step.title} -- {fix.get('explanation') or 'done'}")
                        self._line("ok", f"Step {i} of {n}: {step.title}")
                        return True, ""
                    reason = f"the change was applied but failed verification: {why}"
                    self._line("warn", f"Step {i}: {reason}")
                else:
                    reason = applied.problems or "the change could not be applied"
                    self._line("warn", f"Step {i}: {reason}")
            if attempt == STEP_REVISIONS:
                break
            fix = self.implement(i, n, step, notes, revise_reason=reason)
        self._line("fail", f"Step {i} of {n}: {step.title} -- {reason}")
        return False, reason

    # -- 4. finish -----------------------------------------------------------------------
    def final_checks(self, count):
        criterion = self.adapter.criterion
        for j in range(1, count + 1):
            prompt = _FINAL_CHECK.format(j=j, m=count, goal=self.adapter.goal, plan=self.plan_text,
                                         progress=self.progress(), criterion=criterion,
                                         context=self.adapter.context())
            verdict = self.cli._parse_json_obj(self.ask(prompt, f"Final check ({j}/{count})"))
            if isinstance(verdict, dict) and verdict.get("done") is False:
                return False, str(verdict.get("remaining") or "something is still missing").strip()
        return True, ""

    def run(self):
        out = self.outcome
        try:
            self.outline()
            if self.no_change:
                out.done = True
                out.stopped = "no code change was needed"
                return self._finish()
            sized = self.budget()
            if sized is None:
                self._line("warn", "Could not get a sized plan from the model; running it as a single step.")
                sized = ([Step("Do the work", self.adapter.goal, 2, 1)], 1)
            steps, final_v = sized
            if not steps:
                out.done = True
                out.stopped = "the agent found nothing to change"
                return self._finish()
            # No slicing to the step limit here: a cap has to STOP the run and say so (see the loop
            # below), not quietly shrink the job until the final check has less to look at.
            self._show_plan(steps, final_v)
            if not self.adapter.confirm_plan(steps):
                out.declined = True
                out.stopped = "plan declined"
                return self._finish()

            queue, executed, replans = list(steps), 0, 0
            while True:
                failure = None
                while queue:
                    if executed >= MAX_STEPS:
                        raise _Stop(f"reached the step limit ({MAX_STEPS}; PULSE_THINK_MAX_STEPS)")
                    step = queue.pop(0)
                    executed += 1
                    total = executed + len(queue)
                    out.steps_total = total
                    ok, why = self.run_step(executed, total, step)
                    if ok:
                        out.steps_landed += 1
                    else:
                        failure = f"step {executed} ({step.title}) failed: {why}"
                        break
                if failure is None:
                    done, remaining = self.final_checks(final_v)
                    if done:
                        out.done = True
                        break
                    failure = f"the final check says the job is not finished: {remaining}"
                replans += 1
                if replans > MAX_REPLANS:
                    raise _Stop(f"could not finish after {MAX_REPLANS} replans -- {failure}")
                self._line("warn", f"Revising the plan ({replans}/{MAX_REPLANS}): {failure}")
                sized = self.budget(replan_reason=failure)
                if sized is None:
                    raise _Stop(f"could not get a revised plan -- {failure}")
                queue, final_v = list(sized[0]), sized[1]
                if not queue:
                    done, remaining = self.final_checks(max(1, final_v))
                    out.done = done
                    if not done:
                        out.stopped = f"the agent has no further steps but the job is unfinished: {remaining}"
                    break
                self._show_plan(queue, final_v)
        except _Stop as stop:
            out.stopped = str(stop)
        except AgentRequestFailed as exc:
            out.request_failed = exc
            out.stopped = f"the agent request failed: {exc}"
        except KeyboardInterrupt:
            out.stopped = "interrupted"
            self._finish()
            raise
        return self._finish()

    def _finish(self):
        out = self.outcome
        out.calls = self.calls
        state = "DONE" if out.done else "STOPPED"
        attempted = f", {out.steps_total} planned" if out.steps_total and out.steps_total != out.steps_landed else ""
        lines = [f"{state}: {out.steps_landed} step(s) landed{attempted}, {self.calls} model call(s)."]
        if out.stopped and not out.done:
            lines.append(f"Reason: {out.stopped}")
        if self.log:
            lines.append("What landed:\n" + "\n".join(f"  {entry}" for entry in self.log))
        if self.pending and not out.done:
            lines.append(f"Also applied, but not confirmed by its verification: step {self.pending[0]} "
                         f"({self.pending[1]}). It is on disk; /undo reverts it.")
        out.summary = (self.plan_text + "\n\n" if self.plan_text else "") + "\n".join(lines)
        self._heading("Plan complete" if out.done else "Plan stopped")
        print("\n".join(lines) + "\n")
        return out


# ---------------------------------------------------------------------------------------
# The debugger's adapter
# ---------------------------------------------------------------------------------------

_DEBUG_RULES = (
    "Respond with ONLY one JSON object -- no prose, no fences:\n"
    '{"old": ["exact snippet"], "new": ["replacement"], "files": ["file label"], "explanation": "one sentence"}\n'
    "- old[i] is an exact, verbatim snippet from the shown code (WITHOUT the line-number prefixes) "
    "that occurs exactly once in that file; include enough neighbouring lines to be unique. "
    "new[i] replaces it. files[i] is the label of the file old[i] is in, exactly as in the header "
    "(it may be omitted only when there is a single file).\n"
    "- To insert code, use a nearby existing line as old and repeat it in new with your addition.\n"
    "- If you delete a statement that is the only body of a block, remove the whole block in the same edit "
    "or leave `pass` -- the result must still compile.\n"
    "- Preserve indentation exactly. Change only what this step needs.\n"
)


class DebugAdapter:
    """Think mode inside the debugger: its context, its tools, its code-fix JSON, its applier."""
    outline_extra = (
        " For a bug: say what the root cause is (or how you will find it), what you will change to fix "
        "it, and how you will know it is fixed -- the script running through, or the tracked values "
        "behaving."
    )
    criterion = _CRITERIA[0]
    format_rules = _DEBUG_RULES

    def __init__(self, cli, question):
        self.cli = cli
        self.goal = question

    def context(self):
        return self.cli._build_agent_context(include_code=True)

    def service_tools(self, text):
        return self.cli._service_tool_requests(text)

    def plain_text(self, answer):
        cleaned, _ = self.cli._extract_new_directives(answer)
        return _directive_re().sub("", cleaned).strip()

    def parse(self, raw):
        return self.cli._parse_code_fix(raw)

    def describe(self, fix):
        return self.cli._describe_fix(fix)

    def confirm_plan(self, steps):
        return True                    # the debugger acts on its own; that is what it is for

    def apply(self, fix, step):
        """The same apply-and-recover sequence the fixed pipeline uses: apply, one corrected-snippet
        retry for a mismatch, bounded corrections when the lint gate refuses."""
        cli = self.cli
        was_applied = cli._fix_applied_this_turn
        cli._fix_applied_this_turn = False
        cli._last_apply_skipped = []
        result = cli._apply_code_fix(fix)
        landed = cli._fix_applied_this_turn
        if cli._last_apply_skipped and not landed:
            retry = cli._request_corrected_snippets(fix, cli._last_apply_skipped)
            if retry is not None:
                result = cli._apply_code_fix(retry)
                landed = cli._fix_applied_this_turn
        rounds = 0
        while (not landed and getattr(cli, "_last_apply_lint_failed", None) and not cli._last_apply_skipped
               and rounds < 2):
            rounds += 1
            revised = cli._request_lint_revision(fix, self.goal)
            if revised is None:
                break
            result = cli._apply_code_fix(revised)
            landed = cli._fix_applied_this_turn
            if landed:
                fix.clear()
                fix.update(revised)
        cli._fix_applied_this_turn = was_applied or landed
        if landed:
            return Applied(True, result)
        return Applied(False, result, "the change did not apply (see above)")


def run_debug(cli, question):
    """Think mode for a debugger question. Returns an Outcome."""
    return Engine(cli, DebugAdapter(cli, question)).run()