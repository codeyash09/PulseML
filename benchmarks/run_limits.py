"""Score Pulse's detection by where it stops, not by whether it works.

Each family in limits.py is a ladder of the same fault from blatant to nearly
invisible. The number that matters is the deepest rung still caught -- the weakest
version of that failure Pulse would notice in a real run.

    python3 run_limits.py
    python3 run_limits.py --sensitivity 0.9 --seeds 5
    python3 run_limits.py --family overfit_gap --verbose
    python3 run_limits.py --json limits.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.environ.get("PULSE_SRC")
                or os.path.expanduser("~/pulseml/pulse-pkg/src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pulse import pulse_detect as detect  # noqa: E402
import limits  # noqa: E402


def replay(builder, sensitivity, confirmations):
    """Feed a run to the engine one epoch at a time, as the monitor does."""
    limits.reset_rngs()          # every case starts from the same draw
    histories, tensor_stats = builder()
    histories = {k: list(v) for k, v in histories.items() if v}
    if not histories:
        return [], 0
    length = max(len(v) for v in histories.values())
    engine = detect.DetectionEngine(sensitivity=sensitivity, confirmations=confirmations)
    raised = []
    for step in range(1, length + 1):
        window = {name: values[:step] for name, values in histories.items()}
        # A builder may hand back one stats dict for the whole run, or a list with one
        # per step for faults that are a *change* in a tensor rather than a state of it.
        if isinstance(tensor_stats, list):
            stats = tensor_stats[min(step - 1, len(tensor_stats) - 1)] if tensor_stats else None
        else:
            stats = tensor_stats if (tensor_stats and step > 3) else None
        try:
            result = engine.update(window, step=step, tensor_stats=stats)
        except Exception as exc:                 # a crash here takes a training run down
            raised.append({"epoch": step, "check": "ENGINE_CRASH", "variable": "",
                           "severity": detect.CRITICAL,
                           "message": f"{type(exc).__name__}: {exc}"})
            break
        for finding in result.get("raised", []):
            raised.append({"epoch": step, "check": finding.check,
                           "variable": finding.variable, "severity": finding.severity,
                           "message": finding.message})
    return raised, length


def run_case(case, sensitivity, confirmations):
    raised, length = replay(case["builder"], sensitivity, confirmations)
    crashed = any(r["check"] == "ENGINE_CRASH" for r in raised)
    actionable = [r for r in raised
                  if r["severity"] in (detect.CRITICAL, detect.WARNING)
                  and r["check"] != "ENGINE_CRASH"]
    first = actionable[0] if actionable else None
    expect = case["expect"]
    if case["tag"] == "fault":
        hit = bool(actionable) and (expect in ("any", None)
                                    or any(r["check"] == expect for r in raised))
        verdict = "caught" if hit else "MISSED"
    else:
        verdict = "quiet" if not actionable else "FALSE ALARM"
    return {"name": case["name"], "family": case["family"], "rung": case["rung"],
            "tag": case["tag"], "verdict": verdict, "crashed": crashed,
            "epochs": length, "latency": first["epoch"] if first else None,
            "check": first["check"] if first else None,
            "checks": sorted({r["check"] for r in actionable}),
            "message": (first["message"] if first else "")[:90]}


def summarise_families(rows):
    """For each fault family: how deep down the ladder detection still holds."""
    by_family = defaultdict(list)
    for row in rows:
        if row["tag"] == "fault":
            by_family[row["family"]].append(row)

    out = []
    for family, cases in sorted(by_family.items()):
        cases.sort(key=lambda r: r["rung"])
        caught = [c["verdict"] == "caught" for c in cases]
        total = len(cases)
        hits = sum(caught)
        # The ladder is ordered easiest-first, so the useful number is how far down it
        # holds before the first miss -- not the total, which a lucky deep rung inflates.
        holds = 0
        for ok in caught:
            if not ok:
                break
            holds += 1
        broke_at = cases[holds]["name"] if holds < total else None
        out.append({"family": family, "total": total, "caught": hits,
                    "holds_to": holds, "breaks_at": broke_at,
                    "blind": family in limits.BLIND_FAMILIES,
                    "checks": sorted({c for case in cases for c in case["checks"]}),
                    "latency": [c["latency"] for c in cases if c["latency"]]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensitivity", type=float, default=0.3)
    ap.add_argument("--confirmations", type=int, default=2)
    ap.add_argument("--family", default=None, help="only this family")
    ap.add_argument("--seeds", type=int, default=1, help="redraw every curve's noise N times")
    ap.add_argument("--json", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    all_rows = []
    for seed in range(args.seeds):
        limits.set_seed_offset(seed)
        for case in limits.CASES:
            if args.family and case["family"] != args.family:
                continue
            all_rows.append(run_case(case, args.sensitivity, args.confirmations))

    faults = [r for r in all_rows if r["tag"] == "fault"]
    healthy = [r for r in all_rows if r["tag"] == "healthy"]
    caught = [r for r in faults if r["verdict"] == "caught"]
    alarms = [r for r in healthy if r["verdict"] == "FALSE ALARM"]
    crashes = [r for r in all_rows if r["crashed"]]

    families = summarise_families(all_rows[:len(all_rows) // args.seeds] if args.seeds > 1
                                  else all_rows)

    if args.verbose:
        for r in all_rows:
            mark = {"caught": "OK ", "quiet": "OK ", "MISSED": "!! ",
                    "FALSE ALARM": "!! "}[r["verdict"]]
            print("%s%-44s %-11s lat=%-4s %-22s %s" % (
                mark, r["name"], r["verdict"], r["latency"] or "-",
                r["check"] or "-", r["message"][:60]))
        print()

    known = [f for f in families if not f["blind"]]
    blind = [f for f in families if f["blind"]]

    print("=" * 78)
    print("DETECTION LIMITS  (sensitivity %.2f, confirmations %d, %d seed%s)"
          % (args.sensitivity, args.confirmations, args.seeds, "s" if args.seeds > 1 else ""))
    print("=" * 78)
    print("  %d fault runs across %d families, %d healthy runs"
          % (len(faults), len(families), len(healthy)))
    print("  caught        %d/%d  (%.0f%%)" % (len(caught), len(faults),
                                               100.0 * len(caught) / max(1, len(faults))))
    print("  false alarms  %d/%d  (%.0f%%)" % (len(alarms), len(healthy),
                                               100.0 * len(alarms) / max(1, len(healthy))))
    if crashes:
        print("  ENGINE CRASHES %d  <-- a crash here would take the training run down"
              % len(crashes))
    lat = sorted(r["latency"] for r in caught if r["latency"])
    if lat:
        print("  latency       median %d epochs, p90 %d, worst %d"
              % (lat[len(lat) // 2], lat[int(len(lat) * 0.9)], lat[-1]))

    print("\n--- families Pulse has a check for " + "-" * 42)
    print("  %-34s %-9s %s" % ("family", "holds to", "first miss"))
    for f in sorted(known, key=lambda f: (f["holds_to"] / max(1, f["total"]), f["family"])):
        bar = "#" * f["holds_to"] + "." * (f["total"] - f["holds_to"])
        print("  %-34s %2d/%-2d %-12s %s" % (f["family"], f["holds_to"], f["total"], bar,
                                             (f["breaks_at"] or "-- holds throughout")[:36]))

    print("\n--- families with no corresponding check " + "-" * 36)
    missed_entirely = [f for f in blind if f["caught"] == 0]
    partial = [f for f in blind if 0 < f["caught"] < f["total"]]
    full = [f for f in blind if f["caught"] == f["total"]]
    print("  %d of %d never detected at all:" % (len(missed_entirely), len(blind)))
    for f in sorted(missed_entirely, key=lambda f: f["family"]):
        print("      %s" % f["family"])
    if partial:
        print("  %d caught only in their blatant form:" % len(partial))
        for f in sorted(partial, key=lambda f: f["family"]):
            print("      %-32s %d/%d  (via %s)" % (f["family"], f["caught"], f["total"],
                                                   ", ".join(f["checks"][:3]) or "-"))
    if full:
        print("  %d caught throughout, by a check meant for something else:" % len(full))
        for f in sorted(full, key=lambda f: f["family"]):
            print("      %-32s via %s" % (f["family"], ", ".join(f["checks"][:3]) or "-"))

    if alarms:
        print("\n--- false alarms " + "-" * 60)
        for r in sorted(alarms, key=lambda r: r["name"]):
            print("  %-34s %-20s %s" % (r["name"], r["check"], r["message"][:50]))

    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"sensitivity": args.sensitivity, "confirmations": args.confirmations,
                       "seeds": args.seeds, "rows": all_rows, "families": families},
                      handle, indent=1)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
