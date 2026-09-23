# limits — where Pulse's detection stops

`run_detectbench.py` asks whether Pulse catches a broken run and answers **39/39 at every
sensitivity**. That is a real result and a misleading one: it says the 39 cases are
comfortably inside the envelope, and nothing about where the envelope ends.

This suite is built to find the edge. Every failure is generated as a **ladder** — the
same fault from blatant down to nearly invisible, with noise, length, starting loss and
the healthy part of the curve held fixed — so a difference in outcome is a difference in
detectability and not in luck. A family's score is not caught/missed but *the rung it
stops at*: the weakest version of that failure Pulse would still see in a real run.

It also deliberately includes failures Pulse has **no check for**. A benchmark that only
tests what a system was built to catch measures its thresholds, not its coverage.

```
python3 run_limits.py                       # the whole thing
python3 run_limits.py --seeds 5             # redraw every curve's noise 5x
python3 run_limits.py --family overfit_gap --verbose
python3 run_limits.py --sensitivity 0.9 --json out.json
```

## Scale

| | |
|---|---:|
| distinct failure types (families) | **95** |
| fault runs (5 noise redraws) | **2465** |
| families with a matching check | 42 |
| families with no matching check | 53 |
| healthy runs, deliberately adversarial | 115 |

## Headline

At the default sensitivity 0.3, confirmations 2, over 5 redraws:

```
caught        1216/2465  (49%)
false alarms    15/115   (13%)
latency       median 30 epochs, p90 47, worst 60
```

The old suite scores 100% on the same engine. Both numbers are true; they are measuring
different things. 49% is what happens when the faults are not all at benchmark strength.

Split by whether a matching check exists, so the headline is not just an artefact of
including failures Pulse was never built to see:

| | caught |
|---|---:|
| families with a matching check | **950/1415 (67%)** |
| families with no matching check | **266/1050 (25%)** |
| the hardest rung of each family | **32/95 (34%)** |

The 25% is the interesting one: a quarter of the faults nobody wrote a check for are
caught anyway, because a broken run usually disturbs *something* Pulse watches. The 34%
is the honest summary of the whole exercise — for two thirds of these failure types,
the subtlest version in the ladder goes unnoticed.

## After the detector was changed

Everything below this line describes the detector **as the benchmark first found it**,
and is kept because it is what the numbers were measured against. The gaps it found were
then fixed. The same suite on the current detector:

| | before | after |
|---|---:|---:|
| caught (5 noise redraws, 2450 runs) | 49% | **74%** |
| false alarms (135 healthy runs) | 13% | **7%** |
| families never detected at any rung | 35 of 53 | **5 of 52** |
| `run_detectbench.py` false alarms | 0/25 | **0/25** |
| cost, 100 variables x 20k readings | 775 ms/update | **45 ms/update** |

The sensitivity dial works again too: 72% -> 80% across its range, where before it moved
five points end to end because the misses were structural rather than threshold-bound.

What was added, in the order the benchmark asked for it:

* **A role per variable.** Every check keyed off loss-or-score, which is the run's own
  arithmetic and nothing else. A variable now also has a role -- score, entropy, waste
  fraction, utilisation, throughput, memory, counter, clock, temperature -- and one
  check serves every variable that shares a role.
* **Relations between variables**, because most of what goes wrong is a relationship
  coming apart while every individual curve stays unremarkable: shards that agreed and
  no longer do, one worker out of step, an EMA drifting from its weights, validation
  identical to training, perplexity that is not exp(loss), a reward climbing as episode
  length collapses, a loss term that stopped contributing, a loss settled at ln(classes).
* **Direction awareness**, so a falling accuracy is a regression and a rising reward is
  not.
* **Transient reporting**, so a single-epoch spike is raised the moment it happens
  rather than waiting for a confirmation that can never come.

Five families are still never detected, and all five need something the curves do not
contain: `eval_in_train_mode` and `loss_not_averaged` (a constant offset is
indistinguishable from a normal gap or a different unit), `gradient_not_synced` (needs
to know the intended world size), `metric_off_by_one_epoch` (both curves are individually
perfect), and `resume_regression` (the same shape as a warm restart, and suppressing it
is what keeps warm restarts and curriculum steps quiet).

---

## The agent, on the cadence it chooses for itself

`agentbench.py` is the same ladders run past the *other* layer: the agent wakes on a
schedule, is handed the run, says what the checks could not see, and then chooses when to
look again. It does not see every reading -- only the moments it decided to wake up for.

One run per family at its **hardest** rung, plus every healthy run, on
`deepseek/deepseek-v4.1-flash` via OpenRouter. 520 calls, $1.86, about 35 minutes.

```
flagged       66/94  (70%)   said problem or watch
identified    63/94  (67%)   named the mechanism, not just "something is off"
false alarms   3/23  (13%)   and all three are defensible
audits        4.5 per run, median chosen interval 10 min (min 1, max 60)
```

### The two layers are complementary, and neither is enough

At the hardest rung of all 94 families:

| | |
|---|---:|
| both layers caught it | 41% |
| only the deterministic checks | 12% |
| **only the agent** | **29%** |
| neither | 18% |
| **detectors alone** | 53% |
| **agent alone** | 70% |
| **the two together** | **82%** |

The 29% is the answer to whether the agent earns its place: `reward_hacking`,
`moe_expert_collapse`, `straggler_rank`, `memory_growth`, `throughput_decay`,
`resume_regression`, `eval_set_too_small`, `loss_not_averaged` -- most of them faults
where the evidence is a relationship or a plausible-but-wrong value rather than a
threshold. The 12% the other way is just as important: the checks hold the line on the
subtle numeric rungs the agent reads past.

### How it fails matters more than how often

Of the 28 it missed, the causes are not the same thing:

* **8 never really looked.** One audit or none -- a run too short for the first
  scheduled check, or a fault that arrives after the last one. `short_run_nan[2]` and
  `short_run_diverge[4]` got zero audits: the run was over before the first look.
* **20 looked at least twice and still said ok.** That is judgment, not scheduling.
* **5 of those 20 had the deterministic checks already reporting the fault in the
  evidence pack, and the agent said ok anyway** -- `tensor_nonfinite`, `thermal_risk`,
  `group_diverged`, `norm_growth`, `quantised`. Overriding a correct finding is worse
  than missing one, because the finding was right there in the prompt.

### What the benchmark got wrong first

Two of its own bugs, both of which made the agent look worse than it is:

* **A token limit scored as silence.** At `max_tokens=2500` the model spent the whole
  completion budget on reasoning and returned an empty string. `parse_decision` got
  nothing, the run was scored "the agent saw this and said nothing was wrong", and the
  first pilot read 22%. `nan_onset[40]` -- a loss that is NaN from epoch 40 -- was
  "missed" that way. With room it says `ok` at epoch 15 and `problem` at 42.
* **Three healthy fixtures were synthetic in a way the agent could see.**
  `healthy_normal_gap` was literally `train + 0.25` to the last decimal and
  `healthy_zero_loss_target` a noiseless exponential; the agent called them derived and
  implausible, and it was right. False alarms fell from 30% to 13% once they had noise.

Also worth knowing for the product rather than the benchmark: `reasoning_effort` moved a
single audit from **372 seconds to 18** for an answer of the same shape, and
`build_litellm_agent` defaults to `max_tokens=4000` against an audit that used ~3550 at
default effort. A longer run's evidence pushes that over, and an audit that returns
empty leaves the brain recording `status: unknown` and keeping its old interval.

---

## The sensitivity dial does not buy detection

| sensitivity | caught | false alarms |
|---|---:|---:|
| 0.1 | 49% | 17% |
| 0.3 *(default)* | 51% | 17% |
| 0.5 | 53% | 17% |
| 0.7 | 53% | 17% |
| 0.9 | 54% | 26% |

Five points of recall across the entire dial, and false alarms half again as high at the
end of it. On the old suite the dial behaves properly because every case is catchable and
the only question is noise. Here it barely moves, which says the misses are **structural**
— no check exists, or a check exists and is looking the wrong way — and no threshold
reaches them.

## Three concrete bugs this found

**1. `loss_spike` cannot fire at the default settings.** A single-epoch spike, even
1000×, is caught **0/8**. With `confirmations=1` it is 4/8. The check raises on the spike
epoch, the spike is gone by the next reading, and the hysteresis that requires two
consecutive confirmations discards it. It is a check for a *transient* event gated behind
a mechanism that requires *persistence*. `_IMMEDIATE_CHECKS` exists for exactly this and
contains only `nonfinite` and `tensor_nonfinite`.

**2. A collapsing validation *accuracy* raises nothing.** `val_regression` and
`val_drift` test `b > a` and `late > early` — worsening means going **up**. That is right
for `val_loss` and backwards for `val_accuracy`, where worsening means going down.
Measured directly:

```
val_accuracy 0.94 -> 0.34   raised: NOTHING
val_loss     0.30 -> 0.90   raised: divergence, val_drift, val_regression
```

`val_accuracy_regression` scores **0/7**, including the rung where it loses 30 points.

**3. A weight tensor can be arbitrarily large and nothing looks.** `_check_tensor` reads
only the `nan`/`inf` counts. `tensor_norm_blowup` — weight std climbing to 1e12, all
finite — is **0/6**. The scalar `norm_explosion` check catches this when the run happens
to log a `weight_norm`; nothing catches it from the tensor stats Pulse collects itself.

## Where the checks that do exist give out

Ordered by how far down its ladder each holds. `########` = held to the hardest rung.

| family | holds to | first miss |
|---|---|---|
| loss_spike | `........` 0/8 | 1000× spike *(bug 1)* |
| val_accuracy_regression | `........` 0/7 | −0.30 accuracy *(bug 2)* |
| tensor_norm_blowup | `......` 0/6 | std 1e12 *(bug 3)* |
| overfit_flat_val | `.......` 0/7 | val flat, train −0.60 |
| val_equals_train | `......` 0/6 | val identical to train |
| negative_loss | `......` 0/6 | loss → −1e6 |
| throughput_decay | `.......` 0/7 | tokens/s → 0 |
| memory_growth | `.......` 0/7 | memory 64× |
| step_time_growth | `###....` 3/7 | 2× slower |
| short_run_diverge | `###....` 3/7 | 10-epoch run |
| divergence_onset | `####....` 4/8 | starts at epoch 45 |
| overfit_gap | `####....` 4/8 | val +0.12 over best |
| quantized_loss | `####..` 4/6 | 64 distinct values |
| divergence_rate | `#####...` 5/8 | +1.5%/epoch |
| lr_jump | `#####...` 5/8 | 12× jump |
| stagnation_slope | `#####...` 5/8 | −0.003/epoch |
| frozen_metric | `#####..` 5/7 | frozen 6 epochs |
| grad_vanish | `######..` 6/8 | grad → 1e-6 |
| norm_explosion | `######..` 6/8 | norm → 10× |
| plateau_length | `#######.` 7/8 | flat 5 epochs |
| *21 families* | `########` | hold to the hardest rung |

The ones that hold throughout are the catastrophic ones — NaN, inf, overflow, oscillation,
gradient explosion, never-learned. Pulse is reliable at the things that announce
themselves. The slope is what it is bad at: the 8 families it fails at rung 0 are all
faults where the loss curve alone is not the evidence.

## What is missing entirely

**35 of the 53 no-check families were never detected at any rung.**

The pattern is not random. Almost every one is a *relationship between variables* rather
than the shape of one curve, and Pulse's checks are almost entirely univariate:

- **distributed** — `rank_desync` (ranks training separate models), `straggler_rank`,
  `gradient_not_synced`
- **the metric is wrong, not the run** — `eval_in_train_mode`, `bn_stats_frozen`,
  `metric_off_by_one_epoch`, `perplexity_loss_mismatch`, `loss_not_averaged`,
  `eval_set_too_small`
- **objective is being gamed** — `reward_hacking` (reward up, episode length collapsing),
  `kl_collapse`, `catastrophic_forgetting`, `distill_teacher_ignored`
- **architecture-internal** — `moe_expert_collapse`, `moe_load_imbalance`,
  `attention_entropy_collapse`, `softmax_saturation`, `logit_explosion`
- **infrastructure** — `gpu_util_collapse`, `thermal_throttle`, `checkpoint_not_saving`,
  `padding_fraction_growth`, `oov_rate_growth`
- **only visible across runs** — `seed_not_fixed`

A further **10 families are caught only by accident**, by a check meant for something
else: `tokenizer_mismatch` and `shuffled_labels` raise `stagnation`; `duplicate_batches`
raises `never_learned`; `tensor_moved_to_cpu` raises `slowing_down`. The run gets
flagged, which is what matters most — but the finding names the symptom, and an agent
handed "loss stopped improving" for a tokenizer mismatch starts from the wrong place.

## False alarms

3 of 23 healthy shapes fire, consistently across all 5 redraws:

| run | check | verdict |
|---|---|---|
| `healthy_cosine_restarts` | `lr_jump` | **a real false positive.** SGDR restarts the LR by design; 162× in one step is the schedule working |
| `healthy_double_descent` | `overfitting` | defensible — val is genuinely rising at the time it fires, and no detector can know it comes back |
| `healthy_grokking` | `never_learned` | defensible — 40 flat epochs are indistinguishable from stuck until the break |

Only the first is clearly wrong. `lr_jump` should not fire on a decrease-then-increase
that matches a known schedule shape, or should require the loss to react.

## What this benchmark does not measure

- **Synthetic curves, not real runs.** The shapes are drawn from what these faults
  produce, not recorded from them. `wiring.py` covers the real Keras path; this does not.
- **The detector alone.** Pulse also has an agent that reads the whole run and an audit
  step. A fault missed here may still be caught by the model reading the numbers — that
  is a different measurement and this makes no claim about it.
- **"Caught" is generous.** Any actionable raise on a broken run counts, whatever it
  says. The 10 accidental families pass under that rule.
- **Ladders are not uniformly spaced.** "holds to 5/8" is comparable within a family and
  only roughly comparable between families.
