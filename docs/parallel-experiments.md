# Parallel proxy experiments

Pulse can compare candidate training interventions in isolated subprocesses. Start from
the run's observed failure and source code, then create a JSON spec in the project and
run `/experiment experiments/overflow.json` (or have the debugging agent issue
`EXPERIMENT: experiments/overflow.json`).
The agent and CLI start a background job, so use `/experiment status [id]` to inspect it
or `/experiment cancel <id>` to stop it.

## Reproduction gate

The spec must name a measurable failure signature and include evidence from the original
run. Pulse makes temporary copies of the project, scales the integer dimensions in the
configured JSON model config, and runs the unmodified command repeatedly at each proxy
scale. Candidate and control runs are not launched until the same configured signal is
observed consistently enough to meet `minimum_reproduction_confidence`. If smaller models
do not reproduce it, Pulse tries the next, more representative scale. If no scale passes,
the result is `inconclusive`; candidate fixes are not ranked.

Signals currently include a named metric becoming non-finite, a named metric crossing a
numeric threshold, a regular-expression match in stderr, or a non-zero process exit.
Metric signals should be preferred: process exits and text matches are weaker evidence.
The configured signal identifies what Pulse can measure; it does not establish that every
causal aspect of the original failure has been preserved. Review the evidence and proxy
differences before drawing conclusions.

## Spec format

```json
{
  "command": ["python", "train.py", "--config", "{config}"],
  "proxy": {
    "config_file": "configs/model.json",
    "scales": [0.25, 0.5, 0.75]
  },
  "failure_signature": {
    "kind": "metric_threshold",
    "metric": "loss",
    "operator": "gt",
    "threshold": 10000
  },
  "original_metrics": [0.7, 2.1, 18000],
  "original_evidence": "Pulse recorded loss divergence to 18000 at step 420.",
  "repetitions": 3,
  "max_parallel": 2,
  "adaptive_extra_repetitions": 2,
  "adaptive_top": 2,
  "resource_class": "gpu",
  "devices": ["0"],
  "timeout_seconds": 900,
  "objective": {"metric": "loss", "direction": "min"},
  "validation_scales": [0.5, 0.75, 1.0],
  "validate_top": 2,
  "candidates": [
    {
      "name": "lower-learning-rate",
      "hypothesis": "The configured learning rate causes the divergence.",
      "replacements": [
        {
          "file": "configs/training.json",
          "find": "\"learning_rate\": 0.1",
          "replace": "\"learning_rate\": 0.01"
        }
      ]
    }
  ]
}
```

The command is an argument array, never a shell string. Use `{project}` and `{config}` to
refer to the temporary project copy and its generated proxy config; `{seed}` and
`{branch}` are also substituted in command arguments. The runner should use
`PULSE_EXPERIMENT_SEED` for repeatable random initialization/data order and print metric
snapshots as one JSON object per line:

```text
PULSE_METRICS: {"loss": 0.82, "grad_norm": 1.4, "step": 100}
```

The default proxy dimension keys cover common Transformer/config conventions. Customize
them with `proxy.dimension_keys`. Proxy values are derived from the original JSON config
at each scale; attention-head counts are adjusted to divide the resulting hidden size.
Architecture-specific exporters, custom modules and non-JSON configs should be handled
by the user's runner, which can translate the generated config to the framework/model it
uses. Keep data and checkpoint files in the project tree or use `copy_excludes` only for
paths that are safe to omit.

`candidates[].replacements` are exact text replacements applied only to each candidate's
temporary copy. A missing or unexpectedly repeated target fails that candidate run; no
replacement is written to the watched project. The unchanged `control` branch uses the
same proxy, seeds, and runner. `max_parallel` bounds concurrent subprocesses (default 2);
`resource_class: "gpu"` requires explicit `devices`; Pulse assigns same-seed comparisons
to the same device and bounds concurrency to the number of listed devices. Without an
explicit device list, `auto` runs one worker conservatively; use `resource_class: "cpu"`
to opt into CPU parallelism. Pulse sets `CUDA_VISIBLE_DEVICES` for assigned devices;
distributed launch details remain the runner's responsibility.
`adaptive_extra_repetitions` (default 0) allocates additional same-seed-policy trials to
the unchanged control and up to `adaptive_top` candidates that outperformed the control
in the first comparison. Explicit `seeds` must include those additional trials as well.

## State, validation, and safety boundaries

All branches run in independent temporary project copies and are removed on completion.
The source tree and active training process are never loaded or modified by this runner.
Each child process has a timeout and participates in the engine's cancellation event
(`pulse_experiments.run_experiment(..., cancel_event=...)`). Preserve optimizer and
scheduler state, random states, data-loader position, and custom-module state by making
the runner restore the appropriate checkpoint and use `PULSE_EXPERIMENT_SEED`; Pulse's
existing in-process `REPLAY` snapshots are not interchangeable with such cross-process
state.

Candidate ranking requires every candidate repetition to finish successfully without the
reproduced failure. Optional objective metrics break ties using their mean against the
unchanged control. The top candidates can then be run at progressively larger scales,
including scale `1.0` (the configured source architecture), still in isolation. Passing
that stage is not proof against the live process's precise training state or data stream.
Pulse reports it as isolated source-scale validation, not as an applied fix.

The engine never applies a winning patch to the watched project. Review and apply any
proposed change through Pulse's existing approval, verification, restart, and rollback
path. Results are recorded as experiment incidents and appear in the run dashboard when
cloud sync is enabled.
