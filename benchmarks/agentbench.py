"""Can the agent catch what the detectors cannot, on the cadence it chooses for itself?

Pulse has two layers. The deterministic checks run on every reading and catch 73% of the
failure families in limits.py. The agent is the other layer: it wakes on a schedule, is
handed the whole run -- downsampled curves, the checks' findings, recent events -- and
asked what the checks could not see. Then it chooses when to look again.

That second part is the interesting one and the reason this is a separate benchmark. The
agent does not see every reading. It sees whatever the run looks like at the moments it
decided to wake up, and it decided those moments itself. A cheap way to score well would
be to wake every minute; the prompt asks it not to, and the schedule is clamped to 1-60
minutes either way. So this measures two things at once:

  * **can it tell**, when it does look, that something is wrong; and
  * **does it look often enough** to be there when it matters.

A run is replayed epoch by epoch into a real Brain against a simulated clock, so the
agent's own `next_check_minutes` decides how much of the run it ever sees. Urgent events
still pull the next look forward, exactly as they do live, because that is part of the
mechanism being tested.

    python3 agentbench.py --limit 20                  # a pilot
    python3 agentbench.py --families divergence_rate,nan_onset
    python3 agentbench.py --hardest --workers 12 --json agent.json

Scoring, deliberately in two parts:

  flagged     the agent said "problem" or "watch" on a broken run, or "ok" on a healthy
              one. This is the honest headline: it is what the user is shown.
  identified  its findings actually name the mechanism, judged against the family's
              own keywords. A run can be flagged for the wrong reason, and an agent
              that says "something looks off" about everything would score 100% on
              flagging alone.
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout

sys.path.insert(0, os.environ.get("PULSE_SRC")
                or os.path.expanduser("~/pulseml/pulse-pkg/src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pulse import pulse_brain as brain_mod          # noqa: E402
from pulse import pulse_stream as stream            # noqa: E402
from pulse import pulse_terminal as terminal_mod    # noqa: E402
import limits                                       # noqa: E402
import projects                                     # noqa: E402

DEFAULT_MODEL = "openrouter/deepseek/deepseek-v4.1-flash"

# How long one reading of the replayed run takes in the simulated world. The schedule is
# clamped to 1..60 minutes, so at a minute an epoch the agent's choice spans "look at
# every epoch" to "look once at the end" -- which is the range that makes the decision
# mean something. Shorter, and every interval covers the whole run; longer, and even the
# minimum interval skips most of it.
SECONDS_PER_EPOCH = 60.0


# --------------------------------------------------------------------------- the clock
# Schedule.set and bring_forward read time.time() directly, so the replay moves the
# module's clock rather than sleeping. Each worker thread gets its own.

class _Clock:
    def __init__(self):
        self._local = threading.local()

    def set(self, value):
        self._local.now = value

    def __call__(self):
        return getattr(self._local, "now", None) or time.time()


CLOCK = _Clock()


def install_clock():
    """Point pulse_brain's time.time at the simulated clock, once."""
    if getattr(brain_mod.time, "_pulse_bench_patched", False):
        return
    real = brain_mod.time

    class _Shim:
        def __getattr__(self, name):
            return getattr(real, name)

        @staticmethod
        def time():
            return CLOCK()

    shim = _Shim()
    shim._pulse_bench_patched = True
    brain_mod.time = shim


# --------------------------------------------------------------------------- scoring
# What counts as naming the mechanism. Deliberately generous on wording and strict on
# subject: "loss is increasing" is not an identification of overfitting.

KEYWORDS = {
    "divergence": ("diverg", "increas", "rising", "growing", "blow", "explod", "unstable"),
    "overfit": ("overfit", "memoris", "memoriz", "generalis", "generaliz", "val.*gap", "gap.*val"),
    "nonfinite": ("nan", "inf", "not a number", "non-finite", "nonfinite"),
    "gradient": ("gradient", "grad_norm", "grad norm", "vanish", "explod"),
    "stuck": ("stuck", "plateau", "stagnat", "not improving", "flat", "stopped improving",
              "no progress", "converg"),
    "lr": ("learning rate", "lr ", "schedule", "step size"),
    "throughput": ("throughput", "tokens", "slow", "speed", "stall", "hang", "idle",
                   "utilis", "utiliz"),
    "memory": ("memory", "oom", "leak", "allocat"),
    "leak": ("leak", "too good", "suspicious", "perfect", "label", "cheat"),
    "collapse": ("collaps", "entropy", "degenerat", "mode collapse", "one expert",
                 "single", "saturat"),
    "distributed": ("rank", "worker", "shard", "sync", "all-reduce", "allreduce",
                    "straggl", "barrier", "desync"),
    "config": ("epsilon", "eps", "momentum", "dropout", "weight decay", "config",
               "hyperparam", "out of range"),
    "data": ("data", "shuffl", "label", "batch", "loader", "duplicate", "token",
             "vocab", "padding", "truncat", "oov", "chance"),
    "eval": ("eval", "validation", "val ", "noisy", "too small", "train mode", "dropout"),
    "precision": ("precision", "fp16", "float16", "dtype", "quantis", "quantiz",
                  "round", "resolution", "scale"),
    "tensor": ("tensor", "weight", "layer", "zero", "dtype", "device", "cpu"),
    "objective": ("reward", "episode", "gaming", "shortcut", "degenerate", "proxy",
                  "objective", "kl", "component", "term"),
    "resource": ("gpu", "thermal", "throttl", "clock", "temperature", "checkpoint",
                 "disk", "saving"),
    "schedule": ("restart", "resume", "optimizer state", "regress", "lost", "worse than"),
}

# Which keyword groups count as naming a given family's mechanism.
FAMILY_TOPICS = {
    "divergence_rate": ("divergence",), "divergence_onset": ("divergence",),
    "divergence_under_noise": ("divergence",), "overflow_to_inf": ("divergence", "nonfinite"),
    "short_run_diverge": ("divergence",), "negative_loss": ("divergence", "data"),
    "overfit_gap": ("overfit",), "overfit_flat_val": ("overfit",),
    "val_accuracy_regression": ("overfit", "eval"), "val_equals_train": ("leak", "eval", "data"),
    "nan_onset": ("nonfinite",), "nan_intermittent": ("nonfinite",), "inf_onset": ("nonfinite",),
    "nan_in_gradients": ("nonfinite", "gradient"), "short_run_nan": ("nonfinite",),
    "aux_loss_nan_only": ("nonfinite", "objective"),
    "grad_vanish": ("gradient",), "grad_explode": ("gradient",), "grad_cliff": ("gradient",),
    "dead_relu": ("gradient", "collapse"), "momentum_dead": ("gradient", "config"),
    "grad_clip_always_active": ("gradient", "config"),
    "plateau_length": ("stuck",), "stagnation_slope": ("stuck",), "never_learned": ("stuck",),
    "frozen_metric": ("stuck",), "repeating_cycle": ("stuck", "data"),
    "lr_to_zero": ("lr", "stuck"), "lr_jump": ("lr",), "lr_not_restored": ("lr", "schedule"),
    "oscillation_amp": ("lr", "divergence"), "loss_spike": ("divergence", "schedule"),
    "sawtooth": ("lr", "divergence"), "gradient_accumulation_double": ("lr", "config"),
    "norm_explosion": ("gradient", "divergence"), "norm_collapse": ("gradient", "collapse"),
    "throughput_decay": ("throughput",), "throughput_to_zero": ("throughput",),
    "step_time_growth": ("throughput", "memory"), "memory_growth": ("memory",),
    "gpu_util_collapse": ("throughput", "resource"), "thermal_throttle": ("resource",),
    "straggler_rank": ("distributed", "throughput"), "rank_desync": ("distributed",),
    "gradient_not_synced": ("distributed",), "metrics_stop_updating": ("stuck", "throughput"),
    "checkpoint_not_saving": ("resource",),
    "suspiciously_perfect": ("leak",), "impossible_value": ("leak", "data"),
    "shuffled_labels": ("data",), "tokenizer_mismatch": ("data",),
    "class_imbalance_collapse": ("data", "collapse"), "duplicate_batches": ("data", "leak"),
    "padding_fraction_growth": ("data",), "oov_rate_growth": ("data",),
    "sequence_truncation": ("data",), "data_ordering_bias": ("data",),
    "loss_not_averaged": ("data", "config"),
    "moe_expert_collapse": ("collapse",), "moe_load_imbalance": ("collapse", "distributed"),
    "attention_entropy_collapse": ("collapse",), "softmax_saturation": ("collapse",),
    "logit_explosion": ("collapse", "divergence"), "temperature_collapse": ("collapse",),
    "kl_collapse": ("objective", "collapse"), "gan_mode_collapse": ("collapse",),
    "reward_hacking": ("objective",), "catastrophic_forgetting": ("objective", "overfit"),
    "distill_teacher_ignored": ("objective",), "aux_loss_dominates": ("objective",),
    "embedding_norm_drift": ("collapse", "objective"),
    "eval_in_train_mode": ("eval",), "eval_set_too_small": ("eval",),
    "bn_stats_frozen": ("eval", "config"), "metric_off_by_one_epoch": ("eval", "data"),
    "metric_wrong_axis": ("eval", "data"), "perplexity_loss_mismatch": ("eval", "data"),
    "accuracy_frozen_while_loss_moves": ("stuck", "eval"),
    "quantised_loss": ("precision",), "quantized_loss": ("precision",),
    "silent_dtype_downcast": ("precision",), "loss_scale_collapse": ("precision",),
    "tensor_nonfinite": ("nonfinite", "tensor"), "tensor_inf": ("nonfinite", "tensor"),
    "tensor_norm_blowup": ("tensor", "divergence"), "tensor_all_zeros": ("tensor",),
    "tensor_dtype_change": ("tensor", "precision"), "tensor_moved_to_cpu": ("tensor", "throughput"),
    "resume_regression": ("schedule",), "bad_weight_init": ("config", "divergence"),
    "adam_epsilon_too_large": ("config",), "label_smoothing_floor": ("stuck", "config"),
    "seed_not_fixed": ("distributed", "config"), "needle_in_haystack": ("divergence",),
    "loss_constant_zero": ("stuck", "data"), "ema_diverges": ("objective", "schedule"),
    "adam_eps": ("config",),
}


def identifies(family, text):
    """Do the agent's words name this family's mechanism?"""
    topics = FAMILY_TOPICS.get(family)
    if not topics:
        return None                      # no rubric: report flagging only
    lowered = (text or "").lower()
    for topic in topics:
        for pattern in KEYWORDS.get(topic, ()):
            if re.search(pattern, lowered):
                return True
    return False


# --------------------------------------------------------------------------- the replay

def replay_with_agent(case, agent, sensitivity=0.3, seconds_per_epoch=SECONDS_PER_EPOCH,
                      include_code=False, max_audits=12, shell=False, shell_rounds=4):
    """Run one scenario past a Brain whose agent decides when to look."""
    import shutil
    import tempfile

    limits.reset_rngs()
    histories, tensor_stats = case["builder"]()
    histories = {k: list(v) for k, v in histories.items() if v}
    if not histories:
        return None
    length = max(len(v) for v in histories.values())

    work = tempfile.mkdtemp(prefix="agentbench-")
    started = 1_700_000_000.0
    CLOCK.set(started)
    commands = []
    try:
        project = work
        if shell:
            project = os.path.join(work, "project")
            projects.write_project(project, case, histories)
            agent = with_shell(agent, project, rounds=shell_rounds, transcript=commands)
        brain = brain_mod.Brain(work, agent=agent, sensitivity=sensitivity)
        brain.session = {"session_id": case["name"], "script": "train.py",
                         "pid": 0, "started": started}
        audits = []
        capped = False
        for step in range(1, length + 1):
            CLOCK.set(started + step * seconds_per_epoch)
            values = {name: series[step - 1] for name, series in histories.items()
                      if step <= len(series)}
            frames = [{"kind": stream.KIND_SCALARS, "step": step, "values": values}]
            if tensor_stats:
                current = (tensor_stats[min(step - 1, len(tensor_stats) - 1)]
                           if isinstance(tensor_stats, list) else tensor_stats)
                for name, stats in (current or {}).items():
                    frames.append({"kind": stream.KIND_TENSOR, "name": name, "stats": stats})
            brain.ingest(frames)
            if brain.schedule.due(CLOCK()):
                if len(audits) >= max_audits:
                    # The agent sets its own cadence, so a cautious one asking for the
                    # minute minimum would audit 58 times on a 60-epoch run. Capped so a
                    # sweep terminates, and recorded, because a run that hit the cap was
                    # given *more* chances than this number shows, not fewer.
                    capped = True
                    brain.schedule.set(seconds_per_epoch * length, "benchmark cap")
                    continue
                record = brain.audit(include_code=include_code)
                audits.append({"step": step, "status": record.get("status"),
                               "risk": record.get("risk"),
                               "findings": record.get("findings") or [],
                               "next_check_minutes": record.get("next_check_minutes"),
                               "text": (record.get("text") or "")[:4000]})
        return {"audits": audits, "epochs": length, "capped": capped,
                "commands": commands,
                "detector_findings": sorted({f.check for f in brain.engine.current()})}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def score(case, result):
    audits = result["audits"]
    said = [a for a in audits if a["status"] in ("problem", "watch")]
    problem = [a for a in audits if a["status"] == "problem"]
    text = "\n".join((a["text"] or "") + " " + " ".join(a["findings"]) for a in audits)

    row = {"name": case["name"], "family": case["family"], "rung": case["rung"],
           "tag": case["tag"], "blind": case["family"] in limits.BLIND_FAMILIES,
           "audits": len(audits), "epochs": result["epochs"],
           "first_audit_step": audits[0]["step"] if audits else None,
           "intervals": [a["next_check_minutes"] for a in audits],
           "statuses": [a["status"] for a in audits],
           "detector_findings": result["detector_findings"], "capped": result.get("capped"),
           "commands": [c["command"] for c in result.get("commands") or []][:12],
           "findings": [f for a in audits for f in a["findings"]][:8]}

    if case["tag"] == "fault":
        row["flagged"] = bool(said)
        row["called_problem"] = bool(problem)
        row["identified"] = identifies(case["family"], text) if said else False
        row["flagged_step"] = said[0]["step"] if said else None
        row["verdict"] = "caught" if said else "MISSED"
    else:
        row["flagged"] = bool(problem)
        row["verdict"] = "FALSE ALARM" if problem else "quiet"
    return row


# What the audit prompt does not mention, because the audit has never had tools. Added
# here so the same scheduled check can look at the project it is auditing, using the
# executor the product ships.
TOOL_PREAMBLE = """\
You have a read-only shell in the run's working directory. To use it, write a line

    TERMINAL: <command>

and nothing else after it. You will be given that command's real stdout, stderr and exit
code, and may then ask for another. The directory holds the training script the run is
executing (train.py), its config, and metrics.csv with every reading at full precision.

Use it when the numbers alone cannot settle a question -- what the schedule actually
does, whether the eval path differs from training, what the loss is computed over. Up to
{rounds} commands, then answer. If the numbers are already enough, answer now: an
unnecessary command costs the run nothing but costs you the turn.

"""

TERMINAL_RE = re.compile(r"^\s*TERMINAL:\s*(.+?)\s*$", re.MULTILINE)


def with_shell(ask, workdir, rounds=4, transcript=None):
    """Wrap an agent callable so TERMINAL: directives are actually executed."""
    executor = terminal_mod.TerminalExecutor(default_cwd=workdir)

    def run(prompt):
        base = TOOL_PREAMBLE.format(rounds=rounds) + prompt
        answer = ask(base)
        history = []
        for used in range(rounds):
            match = TERMINAL_RE.search(answer or "")
            if not match:
                break
            command = match.group(1).strip()
            request = terminal_mod.TerminalRequest(command=command, working_directory=workdir,
                                                   timeout=20.0)
            result = executor.run(request)
            if transcript is not None:
                transcript.append({"command": command, "exit_code": result.exit_code,
                                   "stdout": (result.stdout or "")[:400]})
            # A compact transcript rather than the whole conversation replayed. Appending
            # every previous answer and every previous result grew the prompt past 11k
            # characters in two rounds, and the calls got slow enough to hit the wall.
            # The results themselves are all kept: truncating to the last three meant
            # that past three rounds the agent answered without what it found first, and
            # spent later rounds re-discovering it.
            # 700 characters, not 1500. Keeping every result is what lets later rounds
            # build on earlier ones, but at six rounds the full-size version grew the
            # prompt to 11k characters and the last two calls took 160s and 132s --
            # turning a 3-hour sweep into a 14-hour one. Head and tail of each result,
            # so a long file listing still shows both ends.
            blob = ((result.stdout or "") + (result.stderr or "")) or "(no output)"
            if len(blob) > 700:
                blob = blob[:450] + "\n... [%d chars cut] ...\n" % (len(blob) - 700) + blob[-250:]
            history.append("$ %s (exit %s)\n%s" % (command, result.exit_code, blob))
            remaining = rounds - used - 1
            # Say when the last command has been spent, in the same call. Inviting
            # "either one more TERMINAL: line, or your final answer" on the final round
            # guaranteed a tool request that had to be refused in a further call: 3.01
            # calls per audit for 0.99 commands, a third of the whole sweep.
            closing = ("You have %d more command%s if you need %s.\n"
                       "Either one more TERMINAL: line, or your final answer now."
                       % (remaining, "" if remaining == 1 else "s",
                          "it" if remaining == 1 else "them")
                       if remaining else
                       "That was your last command -- no more are available.\n"
                       "Answer now, in the two parts asked for above, ending with the "
                       "JSON object.")
            answer = ask(base + "\n\nCommands you have already run, with their output:\n\n"
                         + "\n\n".join(history) + "\n\n" + closing)

        # It asked for a command it cannot have anyway. Returning that leaves the audit
        # with nothing to parse -- scored as though the agent had looked at the run and
        # had no opinion -- so this stays as a fallback, and should now rarely fire.
        if history and TERMINAL_RE.search(answer or ""):
            answer = ask(base + "\n\nCommands you ran, with their output:\n\n"
                         + "\n\n".join(history)
                         + "\n\nNo further commands are available. Answer now, in the two "
                           "parts asked for above, ending with the JSON object.")
        return answer

    run.executor = executor
    return run


def make_agent(model, max_tokens, timeout, budget, reasoning_effort="low"):
    """A counting, cost-tracking wrapper around the litellm agent."""
    import litellm
    litellm.suppress_debug_info = True
    state = {"calls": 0, "cost": 0.0, "errors": 0, "empty": 0}
    lock = threading.Lock()

    # Reasoning effort matters more than anything else here. On a real audit prompt the
    # default spends ~3300 reasoning tokens and 372 seconds; "low" spends ~1500 and 18
    # seconds for an answer of the same quality and shape. At a hundred-odd runs that is
    # the difference between a sweep that finishes and one that does not.
    extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}

    # No thread-pool wrapper around the call. There used to be one, to enforce a
    # timeout litellm was thought not to honour, and it was the cause of the stalls it
    # was meant to prevent: `future.result(timeout=...)` stops waiting but cannot cancel
    # the submitted work, so every timed-out call kept its slot in the bounded pool.
    # Once enough accumulated, new calls only *queued* -- they timed out without ever
    # being sent, which is why the provider's own log looked perfectly healthy while the
    # sweep ground to a halt and stopped spending. litellm's timeout does work; measured
    # at 10-36s against a 60s limit, and unchanged at 8 concurrent.
    extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}

    def _call(prompt, budget_tokens, box):
        try:
            box["value"] = litellm.completion(
                model=model, messages=[{"role": "user", "content": prompt}],
                max_tokens=budget_tokens, timeout=timeout, num_retries=0, **extra)
        except BaseException as exc:                 # reported through the box
            box["error"] = exc

    def once(prompt, budget_tokens):
        last = None
        for attempt in range(2):
            try:
                # A fresh daemon thread per call, never a shared pool. litellm's own
                # timeout is not enforced against this provider -- a call measured at
                # 264s under a 90s limit -- and those stragglers set the pace, because
                # an audit waits on its slowest call. The earlier fix used a bounded
                # ThreadPoolExecutor and deadlocked: abandoned calls kept their slots
                # until new work only queued. A thread that nobody joins simply ends.
                box = {}
                worker = threading.Thread(target=_call, args=(prompt, budget_tokens, box),
                                          daemon=True)
                worker.start()
                worker.join(timeout + 10.0)
                if worker.is_alive():
                    with lock:
                        state["timeouts"] = state.get("timeouts", 0) + 1
                    raise TimeoutError(f"no answer in {timeout + 10:.0f}s")
                if box.get("error") is not None:
                    raise box["error"]
                response = box["value"]
                break
            except Exception as exc:
                last = exc
                with lock:
                    state["api_errors"] = state.get("api_errors", 0) + 1
                if attempt < 1:
                    time.sleep(3.0)
        else:
            with lock:
                state["gave_up"] = state.get("gave_up", 0) + 1
            raise RuntimeError(f"model failed twice: {type(last).__name__}") from last
        with lock:
            state["calls"] += 1
            usage = getattr(response, "usage", None)
            state["cost"] += float(getattr(usage, "cost", 0.0) or 0.0)
        return (response.choices[0].message.content or "").strip()

    def ask(prompt):
        with lock:
            if budget and state["cost"] >= budget:
                # Flagged, not raised. Raising here happens *inside* an audit, so the
                # audit is recorded as an error and the tail of the sweep reads as an
                # agent that looked and had nothing to say -- the same shape as the
                # contaminated run where 151 of 180 audits errored and scored 11%.
                # main() stops scheduling new work instead; what already completed
                # stays clean.
                state["over_budget"] = True
                raise RuntimeError(f"cost budget of ${budget:.2f} reached")
        answer = once(prompt, max_tokens)
        if not answer:
            # A reasoning model can spend the whole completion budget thinking and
            # return nothing at all. Scored naively that reads as "the agent saw the
            # run and said nothing was wrong", which is a benchmark measuring its own
            # token limit. Retry with room, and count it either way.
            with lock:
                state["empty"] += 1
            answer = once(prompt, max_tokens * 2)
        return answer

    ask.state = state
    return ask


def select(args):
    cases = [c for c in limits.CASES]
    if args.rerun_missed:
        # The experiment worth paying for: take what the agent missed without tools and
        # ask whether tools change the answer. Re-running what it already caught tells
        # you nothing new and costs the same.
        with open(args.rerun_missed) as handle:
            previous = json.load(handle)
        wanted = {r["name"] for r in previous["rows"] if r["verdict"] == "MISSED"}
        healthy = {r["name"] for r in previous["rows"] if r["tag"] == "healthy"}
        cases = [c for c in cases if c["name"] in wanted or c["name"] in healthy]
    if args.families:
        wanted = {f.strip() for f in args.families.split(",")}
        cases = [c for c in cases if c["family"] in wanted]
    if args.hardest:
        # One case per family: the subtlest rung, plus every healthy run.
        deepest = {}
        for case in cases:
            if case["tag"] != "fault":
                continue
            best = deepest.get(case["family"])
            if best is None or case["rung"] > best["rung"]:
                deepest[case["family"]] = case
        cases = list(deepest.values()) + [c for c in cases if c["tag"] == "healthy"]
    if args.mid:
        middle = {}
        for case in cases:
            if case["tag"] != "fault":
                continue
            middle.setdefault(case["family"], []).append(case)
        picked = []
        for family, group in middle.items():
            group.sort(key=lambda c: c["rung"])
            picked.append(group[len(group) // 2])
        cases = picked + [c for c in cases if c["tag"] == "healthy"]
    cases.sort(key=lambda c: (c["tag"] != "fault", c["family"], c["rung"]))
    return cases[:args.limit] if args.limit else cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("PULSE_BENCH_MODEL", DEFAULT_MODEL))
    ap.add_argument("--families", default=None, help="comma-separated family names")
    ap.add_argument("--hardest", action="store_true", help="the subtlest rung of each family")
    ap.add_argument("--mid", action="store_true", help="a middle rung of each family")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rerun-missed", default=None,
                    help="a previous results json: re-run only the cases it scored MISSED")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sensitivity", type=float, default=0.3)
    ap.add_argument("--seconds-per-epoch", type=float, default=SECONDS_PER_EPOCH)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--budget", type=float, default=15.0, help="stop after this many dollars")
    ap.add_argument("--max-audits", type=int, default=12, help="cap per run, so a sweep ends")
    ap.add_argument("--reasoning", default="low", help="reasoning effort; '' for the model default")
    ap.add_argument("--no-shell", dest="shell", action="store_false", default=True,
                    help="audit from the numbers alone; the shell is on by default")
    ap.add_argument("--shell", dest="shell", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--shell-rounds", type=int, default=6, help="commands allowed per audit")
    ap.add_argument("--json", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    install_clock()
    cases = select(args)
    agent = make_agent(args.model, args.max_tokens, args.timeout, args.budget,
                       reasoning_effort=args.reasoning or None)
    print(f"{len(cases)} runs, model {args.model}, {args.workers} workers, "
          f"{args.seconds_per_epoch:g}s per epoch")

    rows, failures = [], []
    started = time.time()

    def run_one(case):
        result = replay_with_agent(case, agent, sensitivity=args.sensitivity,
                                   seconds_per_epoch=args.seconds_per_epoch,
                                   max_audits=args.max_audits, shell=args.shell,
                                   shell_rounds=args.shell_rounds)
        return case, result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, case): case for case in cases}
        done = 0
        for future in as_completed(futures):
            case = futures[future]
            done += 1
            if agent.state.get("over_budget"):
                # Cancel what has not started. Those cases are simply absent from the
                # results rather than present as errors, which keeps the percentages
                # honest for the cases that did run.
                for pending, _ in futures.items():
                    pending.cancel()
            try:
                case, result = future.result()
                if result is None:
                    continue
                row = score(case, result)
                rows.append(row)
                if args.verbose:
                    print("  %-44s %-11s audits=%-2d %s" % (
                        row["name"], row["verdict"], row["audits"],
                        ("identified" if row.get("identified") else "")))
            except Exception as exc:
                failures.append({"name": case["name"], "error": f"{type(exc).__name__}: {exc}"})
            if done % 10 == 0 or done == len(cases):
                print("  %d/%d  %d calls  $%.3f  %.0fs elapsed"
                      % (done, len(cases), agent.state["calls"], agent.state["cost"],
                         time.time() - started))

    report(rows, failures, agent, args)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"model": args.model, "rows": rows, "failures": failures,
                       "calls": agent.state["calls"], "cost": agent.state["cost"]},
                      handle, indent=1)
        print(f"\nwrote {args.json}")


def report(rows, failures, agent, args):
    faults = [r for r in rows if r["tag"] == "fault"]
    healthy = [r for r in rows if r["tag"] == "healthy"]
    caught = [r for r in faults if r["verdict"] == "caught"]
    named = [r for r in faults if r.get("identified")]
    alarms = [r for r in healthy if r["verdict"] == "FALSE ALARM"]

    print("\n" + "=" * 78)
    print("AGENT ON ITS OWN SCHEDULE  (%s)" % args.model)
    print("=" * 78)
    print("  %d broken runs, %d healthy" % (len(faults), len(healthy)))
    if faults:
        print("  flagged       %d/%d  (%.0f%%)   said problem or watch"
              % (len(caught), len(faults), 100.0 * len(caught) / len(faults)))
        rubric = [r for r in faults if r.get("identified") is not None]
        if rubric:
            print("  identified    %d/%d  (%.0f%%)   named the mechanism"
                  % (len(named), len(rubric), 100.0 * len(named) / len(rubric)))
    if healthy:
        print("  false alarms  %d/%d  (%.0f%%)   called a healthy run a problem"
              % (len(alarms), len(healthy), 100.0 * len(alarms) / len(healthy)))
    hit_cap = [r for r in rows if r.get("capped")]
    if hit_cap:
        print("  capped        %d runs hit the %d-audit cap (they asked to look more often)"
              % (len(hit_cap), args.max_audits))
    used = [r for r in rows if r.get("commands")]
    if used:
        ran = sum(len(r["commands"]) for r in rows)
        print("  shell         %d commands across %d of %d runs"
              % (ran, len(used), len(rows)))
    audits = [r["audits"] for r in rows]
    if audits:
        print("  audits        %.1f per run (min %d, max %d)"
              % (sum(audits) / len(audits), min(audits), max(audits)))
    intervals = [m for r in rows for m in r["intervals"] if isinstance(m, (int, float))]
    if intervals:
        intervals.sort()
        print("  intervals     median %.0f min, min %.0f, max %.0f"
              % (intervals[len(intervals) // 2], intervals[0], intervals[-1]))
    print("  cost          $%.3f over %d calls" % (agent.state["cost"], agent.state["calls"]))
    errored = sum(1 for r in rows for st in r["statuses"] if st == "error")
    total_audits = sum(len(r["statuses"]) for r in rows)
    if total_audits:
        share = 100.0 * errored / total_audits
        flag = "   <-- RESULTS ARE NOT TRUSTWORTHY" if share > 15 else ""
        print("  audit errors  %d/%d audits never got an answer (%.0f%%)%s"
              % (errored, total_audits, share, flag))
    if agent.state.get("api_errors"):
        print("  api errors    %d calls failed and were retried" % agent.state["api_errors"])
    if agent.state.get("leaked"):
        print("  leaked        %d calls were abandoned mid-flight (each holds provider budget)"
              % agent.state["leaked"])
    if agent.state.get("gave_up"):
        print("  gave up       %d calls failed every retry" % agent.state["gave_up"])
    if agent.state.get("timeouts"):
        print("  timeouts      %d calls hung and were retried" % agent.state["timeouts"])
    if agent.state.get("empty"):
        print("  empty         %d replies came back with no content and were retried"
              % agent.state["empty"])
    if failures:
        print("  errors        %d runs failed: %s" % (len(failures), failures[0]["error"][:60]))

    missed = [r for r in faults if r["verdict"] == "MISSED"]
    if missed:
        print("\n--- flagged nothing " + "-" * 58)
        for r in sorted(missed, key=lambda r: r["name"]):
            print("  %-46s audits=%-2d detectors: %s"
                  % (r["name"], r["audits"], ", ".join(r["detector_findings"][:3]) or "none"))
    wrong = [r for r in caught if r.get("identified") is False]
    if wrong:
        print("\n--- flagged, but did not name the mechanism " + "-" * 34)
        for r in sorted(wrong, key=lambda r: r["name"])[:25]:
            print("  %-42s %s" % (r["name"], "; ".join(r["findings"][:2])[:56]))
    if alarms:
        print("\n--- false alarms " + "-" * 60)
        for r in sorted(alarms, key=lambda r: r["name"]):
            print("  %-42s %s" % (r["name"], "; ".join(r["findings"][:2])[:56]))


if __name__ == "__main__":
    main()
