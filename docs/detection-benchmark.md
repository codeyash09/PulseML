# What Pulse's detection actually catches

Pulse's deterministic checks are measured against a suite of **94 distinct failure
families, 490 labelled runs, plus 27 healthy ones**. Every family is a *ladder*: the same
fault generated from blatant down to nearly invisible, with noise, length and the healthy
part of the curve held fixed, so a difference in outcome is a difference in detectability
and not in luck.

A family's score is therefore not "caught or missed" but **the rung it stops at** — the
weakest version of that fault Pulse would still see in a real run. That is the number
that matters, because real faults do not arrive at benchmark strength.

The suite and its harness live outside this repository, in `detectbench/`. It runs
against an installed Pulse or a checkout via `PULSE_SRC`.

```
python3 run_limits.py                  # the ladders
python3 run_limits.py --seeds 5        # redraw every curve's noise five times
python3 run_detectbench.py             # the original 67-run suite
python3 edge_cases.py                  # 63 hostile inputs
python3 wiring.py                      # the real Keras ingestion path
```

## Results

At the default sensitivity (0.3), over five redraws of every curve's noise — 2450 runs:

```
caught        1817/2450  (74%)
false alarms    10/135   (7%)
latency       median 29 epochs, p90 46
```

Split by whether a check exists for that kind of fault at all:

| | caught |
|---|---:|
| families with a matching check | 67% |
| families with no matching check | 25% |
| the hardest rung of every family | 34% |

The 25% is not an accident of the scoring: a broken run usually disturbs *something*
Pulse watches, so a quarter of the faults nobody wrote a check for are caught anyway —
by a check meant for something else. The finding then names the symptom rather than the
cause, which is worth knowing when reading one.

The other suites, on the same build:

| suite | result |
|---|---|
| `run_detectbench.py` — 39 broken, 25 healthy | 39/39 caught, **0/25** false alarms |
| `edge_cases.py` — hostile input | 63/63, nothing crashed the detector |
| `wiring.py` — real Keras log dicts | 8/8 |

## The sensitivity dial

| sensitivity | caught | false alarms |
|---|---:|---:|
| 0.1 | 72% | 7% |
| 0.3 *(default)* | 73% | 7% |
| 0.5 | 76% | 7% |
| 0.7 | 78% | 7% |
| 0.9 | 80% | 15% |

Eight points of recall across the dial with false alarms flat until the very end. Before
the detection work described below, the same sweep moved five points end to end — which
was the clue that the misses were structural rather than threshold-bound, and no amount
of turning the dial would reach them.

## What the benchmark changed

It was built to find the edge, and found three bugs before it found any limits:

* **`loss_spike` could not fire.** It raises on the spike epoch; the engine is only
  re-run when new readings arrive, and by then the spike is no longer the latest value,
  so the finding never returned to be confirmed. A check for a *transient* sat behind a
  gate that requires persistence — 0 of 8 spikes, from 1.25x to 1000x.
* **A collapsing validation *accuracy* raised nothing.** The validation checks tested
  that a value went *up*, which is right for a loss and backwards for a score. 0.94 ->
  0.34 was silent while the equivalent `val_loss` rise raised three checks.
* **A weight tensor could be any size at all.** The tensor check read the NaN and inf
  counts and stopped, so a standard deviation of 1e12 reported nothing.

Then the structural gap: 35 of the 53 families with no matching check were invisible at
every strength, and almost all of them were a *relationship between variables* rather
than the shape of one curve. Detection was almost entirely univariate. What was added:

* **A role per variable** — score, entropy, waste fraction, utilisation, throughput,
  memory, counter, clock, temperature — so one check serves every variable sharing a
  role, whatever a given run calls it.
* **Relations between variables** — shards that agreed and no longer do, one worker out
  of step, an EMA drifting from the weights it shadows, validation identical to
  training, perplexity that is not exp(loss), a reward climbing while episode length
  collapses, a loss term that stopped contributing, a loss settled at exactly
  ln(number of classes).
* **Config values that are simply wrong** — an Adam epsilon of 1.0, a BatchNorm momentum
  of zero — which no curve shape can show, because the run does exactly what it was told.
* **Quantisation and periodicity**, as a grid and as an autocorrelation after high-pass
  filtering, so an fp16-rounded loss and data that is never shuffled are both visible.

Detection also became **5-17x faster than before** that work, despite roughly fifteen new
checks: every check used to call `_finite` on the whole history for itself, so a run
logging 100 variables over 20k steps re-parsed four million values on every update — on
the training thread in cli mode. One conversion per update, a fast path for values that
are already floats, and a bounded analysis window took a 100-variable, 20k-reading update
from 775 ms to 45 ms.

## What is still missing

Five families are never detected, and all five need something the numbers do not carry:

| family | why |
|---|---|
| `eval_in_train_mode` | a constant offset cannot be told from a normal generalisation gap |
| `loss_not_averaged` | the same, in different units |
| `gradient_not_synced` | needs to know the intended world size |
| `metric_off_by_one_epoch` | both curves are individually perfect |
| `resume_regression` | the same shape as a warm restart, and suppressing it is what keeps warm restarts and curriculum steps quiet |

The last one is a deliberate trade: three points of recall given up to hold false alarms
on the original suite at zero.

## Reading the numbers honestly

* The curves are **synthetic** — drawn from what these faults produce, not recorded from
  runs that had them. `wiring.py` covers the real Keras ingestion path; nothing else here
  does.
* This measures the **detectors alone**. Pulse also has an agent that reads the whole run
  on a schedule; a fault missed here may still be caught by it, and that is a separate
  measurement.
* **"Caught" is generous** — any actionable raise on a broken run counts, whatever it
  says. The families caught by an unrelated check pass under that rule.
* Ladders are **not uniformly spaced**, so "holds to 5 of 8" is comparable within a
  family and only roughly comparable between families.
