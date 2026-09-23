"""Score the detector that actually runs by default, on the same scenarios.

Pulse has two detectors. `pulse_detect.DetectionEngine` is the one run_detectbench.py
measures, and it is used by the brain process in stream mode. The default path --
`pulse run train.py`, no PULSE_MODE set -- uses `PulseCLI._check_for_trouble`, a
separate hand-written implementation. Whatever the engine scores is not what a user
gets unless the two are the same, so this replays the identical scenarios through the
legacy detector and prints the two results side by side.

    python3 legacy.py
"""
import os
import sys

# Point PULSE_SRC at a Pulse checkout, or rely on an installed pulse package.
sys.path.insert(0, os.environ.get("PULSE_SRC")
                or os.path.expanduser("~/pulseml/pulse-pkg/src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pulse.pulse_cli as pc  # noqa: E402
from run_detectbench import score as engine_score  # noqa: E402
from scenarios import SCENARIOS  # noqa: E402


def make_cli(sensitivity):
    """A real PulseCLI, constructed the way a run constructs it.

    Hand-building the attributes the detector reads was the first attempt, and it was
    wrong twice over: it missed attributes (the detector raised AttributeError, which
    in a real run means the check silently does nothing) and it would have kept on
    missing whatever a future edit added. The constructor is cheap and offline, so the
    benchmark uses the genuine object.
    """
    cli = pc.PulseCLI(watch_locals={}, pdf_dir=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "_legacy_out"))
    cli.sensitivity = sensitivity
    cli.epoch_scalar_histories = {}
    cli.batch_scalar_histories = {}
    return cli


def replay(builder, sensitivity):
    histories, tensor_stats = builder()
    length = max(len(v) for v in histories.values())
    cli = make_cli(sensitivity)
    if tensor_stats:
        # The legacy cache nests the numbers under "stats"; the engine takes them flat.
        cli._matrix_cache = {name: {"stats": dict(stats)} for name, stats in tensor_stats.items()}
    for step in range(1, length + 1):
        cli.epoch_scalar_histories = {name: [v for v in values[:step] if v is not None]
                                      for name, values in histories.items()}
        cli.scalar_histories = dict(cli.epoch_scalar_histories)
        # The per-variable checks only look at self.tracked_vars, so a name that
        # discovery missed is never checked at all. Assume discovery found everything,
        # which is the most favourable reading for the legacy detector.
        cli.tracked_vars = list(histories)
        try:
            trouble = cli._check_for_trouble()
        except Exception as exc:  # a crash here takes the training run with it
            return "CRASH: %s: %s" % (type(exc).__name__, exc), step
        if trouble:
            return trouble, step
    return None, length


def main():
    sensitivity = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
    engine_rows = {r["scenario"]: r for r in engine_score(sensitivity, 2)}

    caught = missed = alarms = quiet = 0
    print("%-32s %-9s %-14s %s" % ("scenario", "tag", "legacy", "engine"))
    for name, builder, expect, tag in SCENARIOS:
        trouble, at = replay(builder, sensitivity)
        engine_verdict = engine_rows[name]["verdict"]
        if tag == "fault":
            verdict = "caught @%d" % at if trouble else "MISSED"
            caught += bool(trouble)
            missed += not trouble
        elif tag == "healthy":
            verdict = "FALSE ALARM" if trouble else "quiet"
            alarms += bool(trouble)
            quiet += not trouble
        else:
            verdict = "fires" if trouble else "quiet"
        mark = "!! " if verdict in ("MISSED", "FALSE ALARM") else "OK "
        print("%s%-30s %-9s %-14s %s" % (mark, name, tag, verdict, engine_verdict))

    print()
    print("legacy detector (the default path, PulseCLI._check_for_trouble)")
    print("  caught       %d/%d broken runs" % (caught, caught + missed))
    print("  false alarms %d/%d healthy runs" % (alarms, alarms + quiet))


if __name__ == "__main__":
    main()
