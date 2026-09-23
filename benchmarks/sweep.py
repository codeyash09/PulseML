"""Does the detector survive noise it has not seen, and the sensitivity dial?

run_detectbench.py scores one draw of each curve. A threshold can pass that by
luck, so this redraws every curve's noise under many seeds and reports, per
scenario, how often it holds. A check that is right for the right reason is
stable across seeds; one that was fitted to a particular draw is not.

    python3 sweep.py [--seeds 30] [--sensitivities 0.1,0.3,0.5,0.7]
"""
import argparse
import os
import sys

# Point PULSE_SRC at a Pulse checkout, or rely on an installed pulse package.
sys.path.insert(0, os.environ.get("PULSE_SRC")
                or os.path.expanduser("~/pulseml/pulse-pkg/src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import scenarios  # noqa: E402
from run_detectbench import score  # noqa: E402


def run(seeds, sensitivity, confirmations):
    """Returns {scenario: [verdict per seed]} plus the tag of each scenario."""
    per_scenario = {}
    tags = {}
    for seed in range(seeds):
        scenarios.set_seed_offset(seed)
        for row in score(sensitivity, confirmations):
            per_scenario.setdefault(row["scenario"], []).append(row["verdict"])
            tags[row["scenario"]] = row["tag"]
    scenarios.set_seed_offset(0)
    return per_scenario, tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--sensitivities", default="0.1,0.3,0.5,0.7")
    ap.add_argument("--confirmations", type=int, default=2)
    args = ap.parse_args()

    levels = [float(x) for x in args.sensitivities.split(",")]
    print("%d seeds x %d sensitivities\n" % (args.seeds, len(levels)))
    unstable = []
    for sensitivity in levels:
        per_scenario, tags = run(args.seeds, sensitivity, args.confirmations)
        caught = alarms = faults = healthy = 0
        shaky = []
        for name, verdicts in sorted(per_scenario.items()):
            tag = tags[name]
            if tag == "fault":
                ok = sum(1 for v in verdicts if v == "caught")
                caught += ok
                faults += len(verdicts)
                if ok != len(verdicts):
                    shaky.append("%s caught %d/%d" % (name, ok, len(verdicts)))
            elif tag == "healthy":
                bad = sum(1 for v in verdicts if v == "FALSE ALARM")
                alarms += bad
                healthy += len(verdicts)
                if bad:
                    shaky.append("%s false-alarmed %d/%d" % (name, bad, len(verdicts)))
        print("sensitivity %.2f:  caught %d/%d (%.1f%%)   false alarms %d/%d (%.1f%%)" % (
            sensitivity, caught, faults, 100.0 * caught / max(faults, 1),
            alarms, healthy, 100.0 * alarms / max(healthy, 1)))
        for line in shaky:
            print("    " + line)
            unstable.append((sensitivity, line))
    print()
    print("perfectly stable at every sensitivity" if not unstable
          else "%d scenario/sensitivity pairs are not unanimous" % len(unstable))


if __name__ == "__main__":
    main()
