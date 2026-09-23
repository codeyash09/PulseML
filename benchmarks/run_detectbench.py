"""Benchmark Pulse's deterministic detection.

Replays each labelled run from scenarios.py epoch by epoch into a fresh
DetectionEngine, exactly as the monitor feeds it during a real run, and scores:

  recall       -- of the runs that are broken, how many were caught
  false alarms -- of the runs that are healthy, how many raised anything
  latency      -- how many epochs of evidence before the first raise
  severity     -- was a broken run raised as critical/warning, or only info

    python3 run_detectbench.py [--sensitivity 0.3] [--json out.json]
"""
import argparse
import json
import os
import sys

# Point PULSE_SRC at a Pulse checkout, or rely on an installed pulse package.
sys.path.insert(0, os.environ.get("PULSE_SRC")
                or os.path.expanduser("~/pulseml/pulse-pkg/src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pulse import pulse_detect as detect  # noqa: E402
from scenarios import SCENARIOS  # noqa: E402


def replay(builder, sensitivity, confirmations):
    """Feed a run to the engine one epoch at a time; return what it raised."""
    histories, tensor_stats = builder()
    length = max(len(v) for v in histories.values())
    engine = detect.DetectionEngine(sensitivity=sensitivity, confirmations=confirmations)
    raised = []
    for step in range(1, length + 1):
        window = {name: values[:step] for name, values in histories.items()}
        stats = tensor_stats if (tensor_stats and step > 3) else None
        result = engine.update(window, step=step, tensor_stats=stats)
        for finding in result.get("raised", []):
            raised.append({"epoch": step, "check": finding.check, "variable": finding.variable,
                           "severity": finding.severity, "message": finding.message})
    return raised, length


def score(sensitivity, confirmations):
    rows = []
    for name, builder, expect, tag in SCENARIOS:
        raised, length = replay(builder, sensitivity, confirmations)
        actionable = [r for r in raised if r["severity"] in (detect.CRITICAL, detect.WARNING)]
        first = actionable[0] if actionable else None
        if tag == "fault":
            hit = bool(actionable) and (expect in ("any", None) or any(r["check"] == expect for r in raised))
            verdict = "caught" if hit else "MISSED"
        elif tag == "healthy":
            verdict = "quiet" if not actionable else "FALSE ALARM"
        else:
            verdict = "quiet" if not actionable else "fires (by design)"
        rows.append({"scenario": name, "tag": tag, "expected": expect, "verdict": verdict,
                     "epochs": length, "latency": first["epoch"] if first else None,
                     "first_check": first["check"] if first else None,
                     "severity": first["severity"] if first else None,
                     "all_checks": sorted({r["check"] for r in raised}),
                     "message": first["message"] if first else ""})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensitivity", type=float, default=0.3)
    ap.add_argument("--confirmations", type=int, default=2)
    ap.add_argument("--json", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    rows = score(args.sensitivity, args.confirmations)
    faults = [r for r in rows if r["tag"] == "fault"]
    healthy = [r for r in rows if r["tag"] == "healthy"]
    caught = [r for r in faults if r["verdict"] == "caught"]
    alarms = [r for r in healthy if r["verdict"] == "FALSE ALARM"]

    if not args.quiet:
        for r in rows:
            mark = {"caught": "OK ", "quiet": "OK ", "MISSED": "!! ", "FALSE ALARM": "!! "}.get(r["verdict"], "   ")
            print("%s%-32s %-9s %-13s latency=%-5s %-18s %s" % (
                mark, r["scenario"], r["tag"], r["verdict"],
                r["latency"] if r["latency"] else "-", r["first_check"] or "-",
                (r["message"] or "")[:70]))
        print()
    print("sensitivity %.2f, confirmations %d" % (args.sensitivity, args.confirmations))
    print("  caught       %d/%d broken runs" % (len(caught), len(faults)))
    print("  false alarms %d/%d healthy runs" % (len(alarms), len(healthy)))
    lat = [r["latency"] for r in caught if r["latency"]]
    if lat:
        print("  latency      median %d epochs, worst %d" % (sorted(lat)[len(lat) // 2], max(lat)))
    if alarms:
        print("  fired on:    " + ", ".join(f"{r['scenario']} ({r['first_check']})" for r in alarms))
    missed = [r["scenario"] for r in faults if r["verdict"] == "MISSED"]
    if missed:
        print("  missed:      " + ", ".join(missed))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"sensitivity": args.sensitivity, "confirmations": args.confirmations,
                       "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
