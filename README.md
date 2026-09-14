# PulseML

**Pulse** is a live ML training debugger for CLI and headless environments. It combines runtime observability, deterministic numerical checks, ML-specific linting, and an AI debugging agent to investigate failures while training is running.

**Track → Diagnose → Verify → Patch**

[Dashboard](https://pulsedashb.netlify.app/) · [GitHub](https://github.com/codeyash09/PulseML) · [PyPI](https://pypi.org/project/pulseml/)

---

## Installation

Pulse is available on PyPI.

### Requirements

- Python 3.9+
- A supported backend: NumPy, PyTorch, TensorFlow, CuPy, or JAX
- An API key for an AI provider if you want to use the AI debugging agent

### Install

```bash
pip install pulseml
```

### Verify

```bash
python -c "import pulse; print('Pulse installed successfully')"
```

### Start Pulse

Import `auto_track` and call it immediately before your training loop:

```python
from pulse import auto_track

if __name__ == "__main__":
    auto_track()

    # Your training loop
    for epoch in range(num_epochs):
        # Training logic here
        pass
```

The `__main__` guard is particularly important in multiprocessing or process-spawning environments.

Once the process is running, Pulse discovers numeric variables available to the training process and exposes them through the CLI.

### AI provider configuration

Pulse supports cloud and local AI providers.

Common provider environment variables include:

```text
ANTHROPIC_API_KEY
OPENAI_API_KEY
GEMINI_API_KEY
DEEPSEEK_API_KEY
MISTRAL_API_KEY
OPENROUTER_API_KEY
```

For example:

```bash
export GEMINI_API_KEY="your-api-key"
```

On Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="your-api-key"
```

Do not place API keys directly in source code or commit them to a repository.

---

## Why Pulse

Machine-learning failures are often diagnosed from the wrong end of the problem:

```text
Train
  ↓
Wait
  ↓
Training fails
  ↓
Read logs
  ↓
Guess
  ↓
Change code
  ↓
Train again
```

Pulse is designed to move the debugging process into the training run itself:

```text
Train
  ↓
Track
  ↓
Detect
  ↓
Measure
  ↓
Verify
  ↓
Diagnose
  ↓
Patch
```

The useful evidence behind an ML failure may be a gradient change, activation distribution, tensor shape, normalization relationship, parameter update, or the exact point at which a numerical value began to diverge.

Pulse gives the debugging agent access to that evidence instead of requiring it to reconstruct the run from logs alone.

---

## Benchmark

Pulse was evaluated against Claude Code on a **28-case ML debugging benchmark** designed to test whether an AI coding agent could identify and repair injected ML faults.

### Results

| Agent | Model | Score |
|---|---|---:|
| **Pulse** | Gemini 3.5 Flash Lite | **15 / 28** |
| **Claude Code** | Opus 5 | **18 / 28** |

Pulse achieved **15/28**, compared with **18/28** for Claude Code.

The benchmark was particularly informative because the task was not simply to generate code that ran. The agent needed to identify the injected mutation, determine whether it was actually responsible for the observed behavior, and apply the necessary correction.

### What the benchmark exposed

Pulse's largest weakness was **diagnostic precision**.

In difficult cases, Pulse could recognize that something was wrong but fail to consistently:

1. identify the exact mutation responsible for the failure;
2. isolate the smallest relevant code region;
3. distinguish the root cause from surrounding symptoms; or
4. apply only the necessary fix instead of proposing a broader change.

This matters because ML debugging is different from generic code generation.

A debugger should not rewrite a training system simply because it can. It should be able to identify the specific value, argument, operation, or line responsible for the failure and make the smallest correct change.

### How the benchmark changed Pulse

The benchmark directly motivated upgrades to the debugging pipeline:

- **Expanded `mllint`** with additional deterministic ML-specific checks.
- **Minimal-fix-first behavior**, prioritizing single-line or small adjacent-line fixes before refactors.
- **Whitespace-tolerant patch matching**, preventing valid fixes from being rejected because of formatting differences.
- **Retry-on-mismatch patching**, allowing the agent to re-quote the exact current code when a proposed patch does not match.
- **Stronger verification**, checking both logical correctness and whether the scope of the patch is unnecessarily broad.
- **More agentic debugging workflows**, reducing the amount of manual investigation the underlying model has to perform.

The benchmark is not presented as proof that Pulse is already better than general-purpose coding agents. It provides a measurable baseline, exposes a concrete failure mode, and gives Pulse a data-driven direction for improvement.

---

## Key Features

### Live ML Training Monitoring

Track losses, metrics, tensors, gradients, activations, weights, and other numerical values while training is running.

Inspect:

- shapes
- dtypes
- devices
- norms
- statistics
- NaN / Inf counts
- scalar histories
- gradient information
- activation information

### CLI / Headless First

Pulse is built for:

- terminals
- SSH sessions
- Google Colab
- containers
- remote servers
- long-running training jobs
- cloud GPU machines

No graphical interface is required.

### CPU-First Tracking

Pulse is intentionally conservative around GPU access.

By default:

```text
TRACKING_MODE = CPU_DEFAULT
GPU_TRACKING = OPT-IN
DEFAULT_GPU_OVERHEAD = 0
```

If a variable already exists on a GPU, Pulse does not automatically copy it back to the CPU on every iteration.

GPU tracking is explicitly requested:

```text
/gputrack <variable>
```

This can introduce device-to-host transfer overhead. That is expected when inspecting GPU-resident data.

The design principle is:

> **If nobody asks Pulse to touch the GPU, Pulse doesn't touch the GPU.**

### Dynamic Variable Tracking

Variables can be added, removed, promoted, or demoted while training is running.

```text
/vars
/tracked
/add <variable>
/track <variable>
/lotrack <variable>
/gputrack <variable>
/gpuuntrack <variable>
/delete <variable>
```

### Deterministic Numerical Verification

Pulse separates AI reasoning from exact numerical calculation.

Instead of asking an AI model to estimate whether a gradient or update is unusually large, Pulse can calculate the relevant quantity directly.

```text
AI hypothesis
      ↓
Numerical calculation
      ↓
Exact result
      ↓
Evidence-backed diagnosis
```

This can be used for:

- gradient/update ratios
- scaling factors
- normalization calculations
- parameter changes
- numerical thresholds
- restricted mathematical expressions

The AI reasons about what the numbers mean. Pulse provides deterministic measurements for the arithmetic.

### ML Linting

`mllint` provides deterministic, code-level checks for common ML failure patterns before an agent has to reason about them from scratch.

Checks include patterns involving:

- incompatible loss/metric combinations
- redundant activation/loss combinations
- `backward()` without the expected optimizer update
- training/evaluation mode issues
- suspicious learning-rate configurations
- other ML-specific static patterns

The goal is not to replace the debugging agent. It is to provide high-confidence evidence and eliminate classes of mistakes that do not require probabilistic reasoning.

### Agentic Debugging

Pulse can combine:

- live training state
- tensor statistics
- scalar histories
- gradients
- activations
- tracebacks
- source code
- deterministic numerical checks
- static lint findings

The goal is to move from:

```text
"What might be wrong?"
```

toward:

```text
"Here is the evidence.
Here is the mutation.
Here is why it caused the failure.
Here is the smallest necessary fix."
```

### Minimal Fixes by Default

Pulse's debugging pipeline is designed to prefer the **smallest correct change**.

The agent should first consider:

1. a single-line change;
2. a small adjacent-line change;
3. a larger edit only when the root cause genuinely requires it.

Unrelated cleanup and refactoring should not be bundled into a debugging patch.

This is especially important for ML debugging because broad changes can hide whether the actual fault was correctly identified.

### Automatic Intervention

With:

```text
/autofix on
```

Pulse can pause training when configured detection logic identifies serious numerical or training problems, allowing the debugging agent to investigate before additional compute is wasted.

---

## CLI

Useful commands include:

```text
/help

/vars
/tracked

/add <variable>
/track <variable>
/lotrack <variable>

/gputrack <variable>
/gpuuntrack <variable>

/delete <variable>

/autofix on|off

/code
/cloud
```

Pulse can pause training for investigation, allowing you to inspect the current state, add variables, ask the AI agent questions, and investigate a failure before continuing.

---

## Quickstart

After installation, import `auto_track` immediately before your training loop:

```python
from pulse import auto_track

if __name__ == "__main__":
    auto_track()

    # Your training loop
    for epoch in range(num_epochs):
        # Training logic here
        pass
```

Pulse discovers numeric variables available to the training process and provides them through the CLI.

You can then control tracking while the run is active:

```text
/vars
/tracked
/add <variable>
/track <variable>
/lotrack <variable>
/gputrack <variable>
/gpuuntrack <variable>
/delete <variable>
```

---

## Supported Backends

| Backend | Support |
|---|---|
| NumPy | Yes |
| PyTorch | Yes |
| TensorFlow | Yes |
| CuPy | Yes |
| JAX | Yes |

Pulse uses a shared backend abstraction so the debugging workflow can remain consistent across frameworks.

---

## Past Debugs

Pulse has been used to investigate numerical failures in custom ML systems.

### Residual normalization failure

A custom LLM became unstable after a 2.5× vocabulary increase. Pulse helped identify a normalization issue in which residual growth was divided by `math.sqrt(num_layers)` rather than `num_layers`.

The important part was not simply observing that training became unstable. The debugging process connected the observed activation behavior to the mathematical relationship causing the instability.

### Attention numerical failure

A custom attention implementation produced NaN loss. Pulse traced the failure to a missing infinity check before a division operation.

These cases represent the intended workflow:

**observe the behavior → inspect the evidence → verify the math → identify the actual failure**

---

## Performance

A debugger should not become the training bottleneck.

Pulse is designed around:

- CPU-first inspection
- opt-in GPU probing
- selective variable tracking
- lightweight tracking modes
- separate probe cadences
- cached statistics
- host-side numerical processing where possible
- minimal intervention in the training loop

The objective is:

```text
MORE VISIBILITY
      +
LESS OVERHEAD
```

Rather than collecting everything continuously, Pulse lets you decide what information is worth monitoring.

---

## Cloud Workspaces

Pulse can optionally synchronize debugging information through shared workspaces.

Depending on configuration, a workspace can provide shared access to:

- debugging sessions
- training incidents
- tracebacks
- agent conversations
- repository metadata
- team membership
- telemetry

Workspaces can be given custom names and managed from the CLI. Members can leave a workspace, while workspace administrators can delete a workspace.

Cloud synchronization is best-effort and is not intended to block the training loop.

For sensitive projects, review your cloud and telemetry configuration carefully.

Telemetry can be disabled with:

```text
PULSE_TELEMETRY=off
```

Using a local AI model keeps model requests local, but does not automatically disable separately enabled Pulse workspace synchronization.

---

## AI Providers

Pulse can use supported cloud or local AI providers for its debugging agent.

Common provider environment variables include:

```text
ANTHROPIC_API_KEY
OPENAI_API_KEY
GEMINI_API_KEY
DEEPSEEK_API_KEY
MISTRAL_API_KEY
OPENROUTER_API_KEY
```

Local and self-hosted models can also be used where supported.

Do not place API keys directly in source code or commit them to a repository.

---

## Why This Matters

The central problem Pulse is trying to solve is not code generation.

Modern coding agents are increasingly capable of writing large amounts of code. ML debugging has a different requirement: **causal precision**.

A useful ML debugger needs to connect three things:

```text
CODE
  ↕
RUNTIME BEHAVIOR
  ↕
MATHEMATICS
```

If an agent only sees code, it can miss what actually happened during training.

If it only sees runtime values, it may not know which source-level mutation produced them.

If it only reasons probabilistically, it can make confident but numerically unsupported claims.

Pulse is designed to put those pieces together.

The benchmark results reinforce why that distinction matters. At the initial 28-case benchmark, Pulse scored 15/28 versus 18/28 for Claude Code with Opus 5. Rather than treating that gap as a dead end, the failures identified a concrete engineering problem: **the system needed to make the agent better at isolating the exact mutation and applying the smallest necessary fix.**

That is the direction of Pulse's development.

---

## Project Direction

Pulse is being built around a debugging stack in which runtime observation, deterministic computation, static ML analysis, and AI reasoning reinforce each other:

```text
TRAINING
   ↓
OBSERVATION
   ↓
STATIC ANALYSIS
   ↓
NUMERICAL EVIDENCE
   ↓
AI REASONING
   ↓
VERIFICATION
   ↓
MINIMAL PATCH
```

The goal is not simply to report:

```text
"Your training is broken."
```

It is to answer:

```text
What broke?

Why did it break?

When did it start?

Can the numbers prove it?

What exactly changed?

What is the smallest correct fix?

Can that fix be verified?
```

---

## Links

- [Dashboard](https://pulsedashb.netlify.app/)
- [GitHub](https://github.com/codeyash09/PulseML)
- [PyPI](https://pypi.org/project/pulseml/)

---

## License

Proprietary. See `LICENSE`.

Use of this software is governed by the terms in that file. Copying, redistribution, and reverse engineering are not permitted.
