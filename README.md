# PulseML

### The live debugger for machine learning

**Observe your tensors. Understand failures. Verify the math. Fix the code.**

PulseML is a live debugging tool for machine-learning training loops. It monitors tensors, gradients, activations, losses, and other metrics while training is running, then presents the evidence through a GUI or a headless CLI. Its AI analyst can diagnose a failure, verify numerical reasoning with deterministic calculations, develop a concrete fix, and apply that fix when explicitly asked.

[Website](https://pulsedb.netlify.app/) | [PyPI](https://pypi.org/project/pulseml/)

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/pulseml?period=total&units=NONE&left_color=BLACK&right_color=GREY&left_text=downloads)](https://pepy.tech/projects/pulseml)

## Why PulseML?

Most ML debugging begins after training has already failed. PulseML makes the internal state of a model visible while it is changing:

```text
TRACK -> VISUALIZE -> ANALYZE -> VERIFY -> FIX
```

Instead of adding custom logging and plotting code throughout a project, place one call before the training loop and let Pulse discover monitorable values.

## Quickstart

Install PulseML from PyPI:

```bash
pip install pulseml
```

Then call `auto_track()` immediately before your training loop. Keep the loop under a `__main__` guard, especially when using multiprocessing or a process-spawning environment.

```python
from pulse import auto_track

if __name__ == "__main__":
    auto_track()

    for epoch in range(num_epochs):
        # Your training code
        pass
```

Pulse discovers numeric variables available to the training process and launches the appropriate debugging interface.

## What It Monitors

- Losses and other scalar metrics as live step charts
- Accuracy, learning rate, gradient norms, and numerical diagnostics
- Matrices, vectors, and higher-dimensional tensors
- Tensor shapes, dtypes, devices, and summary statistics
- NaN and infinity counts
- Gradients and activations
- Heatmaps in GUI mode
- Optional labeled PDF snapshots in CLI mode

Loss-like names such as `loss`, `cost`, `nll`, and `cross_entropy` are recognized automatically and prioritized during setup.

## GUI

The GUI provides an interactive workflow for selecting and inspecting variables:

- **Matrix picker:** choose tensors to monitor while viewing their shapes.
- **Live dashboard:** follow heatmaps, scalar charts, statistics, and updates together.
- **AI analyst:** ask questions about the current training state from alongside the live data.

## CLI and Headless Environments

PulseML also works without a graphical display, making it suitable for Google Colab, SSH sessions, remote GPUs, containers, and headless servers.

The CLI supports:

- Live tensor statistics
- ASCII scalar charts
- Matrix and tensor tracking
- Pause and resume during training
- Adding, removing, or promoting variables while training
- AI analysis from the terminal
- Optional labeled heatmap PDFs
- Automatic intervention when a tracked signal becomes non-finite or shows suspicious behavior

Useful commands include:

```text
/help                 Show all commands
/vars                 List discovered variables
/tracked              List tracked variables
/add <name>           Add a variable
/track <name>         Promote a variable to full tracking
/lotrack <name>       Use lightweight tracking
/gputrack <name>      Opt into GPU-resident probing
/gpuuntrack <name>    Stop GPU-resident probing
/autofix on|off       Toggle automatic intervention
/sensitivity          View or change detection sensitivity
/code                 Toggle training-code context for agent questions
/cloud                Show cloud and workspace status
```

PDF snapshots can be enabled for tracked variables and are written in this form:

```text
Pulse_Output/<variable_name>/step000001.pdf
```

## AI Analyst

Pulse's agentic workflow is designed to move from observation to an actionable change:

```text
+-------------+       +-------------+       +-------------+
|  DESCRIBE   | ----> |   DEVELOP   | ----> |  IMPLEMENT  |
| Understand  |       | Build the   |       | Apply the   |
| the problem |       | actual fix  |       | solution    |
+-------------+       +-------------+       +-------------+
```

### Describe

The agent reasons over the available code, tensor statistics, scalar histories, heatmaps, and training behavior to identify a likely root cause.

### Develop

It proposes a specific solution grounded in the observed values and the relevant code, rather than returning generic advice.

### Implement

When the user explicitly requests a fix, Pulse can apply a structured code change to the relevant file. Fixes are recorded so recurring failures can be recognized and re-applied, and the process can restart to run the updated code.

Pulse can also inspect imported project files when they are available, which helps diagnose bugs that live outside the entry script.

## Deterministic Math Verification

LLMs can reason about math, but they should not be trusted to perform exact arithmetic unaided. Pulse provides a restricted mathematical evaluator for expressions such as:

- Update magnitudes and ratios
- Scaling factors and normalization values
- Gradient relationships
- Numerical thresholds

The evaluator exposes numeric operations and the Python `math` module while disabling builtins and rejecting general code execution. The agent can delegate a calculation and use the exact result in its diagnosis.

```text
             AI AGENT
             /      \
            /        \
      Reasoning    Math check
            \        /
             \      /
             Verified result
```

## Universal Backend Support

PulseML uses a shared backend abstraction so the same monitoring workflow can inspect arrays from different ML ecosystems.

| Backend | Support |
| --- | --- |
| NumPy | Yes |
| PyTorch | Yes |
| TensorFlow | Yes |
| CuPy | Yes |
| JAX | Yes |

Tracked values are converted to host-side NumPy data for inspection and rendering. GPU variables are not synchronized on every training step by default. GPU tracking is opt-in and uses a slower probe cadence because device-to-host reads can affect throughput.

## AI Providers

Cloud providers are accessed through LiteLLM. Configure the provider's environment variable before starting Pulse:

```text
ANTHROPIC_API_KEY
OPENAI_API_KEY
GEMINI_API_KEY
DEEPSEEK_API_KEY
MISTRAL_API_KEY
OPENROUTER_API_KEY
```

Pulse can also use local or self-hosted models, including Ollama and OpenAI-compatible servers such as LM Studio, vLLM, and TGI. Local providers use a model name and local API base instead of a cloud API key.

For unattended runs, provider and runtime settings can be supplied through environment variables or a `pulse_config.json` file beside the training script. See the CLI prompts and `/help` for the available setup options.

## Installation Notes

The base package includes CLI support and the dependencies required for PDF snapshots.

GUI mode requires `tkinter`, which is included with most Python installations. On Debian or Ubuntu:

```bash
sudo apt install python3-tk
```

## Performance Model

Instrumentation should not become the bottleneck. PulseML reduces overhead through:

- Selective tracking of monitored variables
- Lightweight tracking for matrices and tensors by default
- Separate cadences for full, lightweight, and GPU probes
- Cached matrix statistics between probes
- Host-side NumPy conversion only when inspection is needed
- Thumbnail-sized rendering and reusable plotting paths

The objective is simple: **more visibility, less overhead**.

## Examples of Real Debugging Problems

### Vocabulary expansion and unstable training

A custom language model became unstable after a 2.5x vocabulary increase. Pulse's diagnostics exposed a normalization error: residual growth was divided by `sqrt(num_layers)` instead of `num_layers`. The resulting activation growth eventually destabilized training and halted learning.

### Custom attention and NaN loss

In another run, a custom attention implementation produced a NaN loss. Pulse helped trace the failure to a missing infinity check before a division operation.

## Cloud Workspaces and Privacy

Pulse can log debug sessions, agent conversations, tracebacks, telemetry, incidents, team membership, and repository metadata to a shared workspace dashboard. Cloud synchronization is best-effort and does not block training.

For sensitive projects, use a local provider and review cloud settings carefully: a local LLM keeps model requests on the machine, but enabled Pulse Cloud synchronization can still upload session logs and tracebacks. Telemetry can be disabled with:

```text
PULSE_TELEMETRY=off
```

Do not place API keys in source files or commit them to a repository. Pulse stores provider configuration separately from its local account cache and does not store provider API keys in its profile.

## Project Direction

PulseML is being built toward a debugging workflow where observation, AI reasoning, and exact computation reinforce one another:

```text
TRAINING
   |
   v
OBSERVATION
   |
   +------------------+
   |                  |
   v                  v
AI REASONING      EXACT MATH
   |                  |
   +--------+---------+
            |
            v
        SOLUTION
            |
            v
       DEVELOPMENT
            |
            v
       IMPLEMENTATION
```

The goal is not merely to report that a model is broken. PulseML should help determine why, verify the reasoning, develop the solution, and implement the fix.

## License

Proprietary. See [LICENSE](/pulse-pkg/LICENSE).

Use of this software is governed by the terms in that file. Copying, redistribution, and reverse engineering are not permitted.
