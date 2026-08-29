# PulseML

### The live debugger for machine learning.

**Observe your tensors. Understand failures. Verify the math. Fix the code.**

Pulse is a live ML training debugger for GUI and CLI environments, built to work across major ML backends. It monitors tensors and metrics in real time, visualizes what is happening inside your training loop, and gives an AI analyst the ability to **reason, calculate, develop, and implement fixes**.

[Website](https://pulsedb.netlify.app/) · [PyPI](https://pypi.org/project/pulseml/)

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/pulseml?period=total\&units=NONE\&left_color=BLACK\&right_color=GREY\&left_text=downloads)](https://pepy.tech/projects/pulseml)

---

# Quickstart

Import `auto_track` immediately before your training loop.

Make sure your loop is wrapped in `if __name__ == '__main__':`.

```python
from pulse import auto_track

if __name__ == '__main__':
    auto_track()

    for epoch in range(num_epochs):
        # Your training code
        pass
```

Pulse discovers variables available for monitoring and launches the appropriate debugging interface.

---

## Debug the Training Process While It Runs

Most ML debugging starts after something has already gone wrong.

Pulse takes a different approach:

```text
TRACK
  |
  v
VISUALIZE
  |
  v
ANALYZE
  |
  v
VERIFY
  |
  v
FIX
```

Track the tensors responsible for your model's behavior, visualize their evolution, and investigate problems without having to manually instrument every part of your training loop.

---

# Core Features

## Live Tensor Debugging

Monitor tensors and variables directly inside your training loop.

* Live matrix visualization
* Tensor shapes and statistics
* Heatmaps
* Gradient monitoring
* Activation monitoring
* Real-time scalar tracking
* Loss and metric curves

Pulse is designed to make the internal state of a model visible while it is actually training.

---

## Smart Scalars

Pulse automatically recognizes scalar values such as:

* Loss
* Accuracy
* Learning rate
* Gradient norms
* Other numerical training metrics

Instead of rendering scalars as matrices, Pulse automatically displays them as live step-charts.

Loss-like variables can also be automatically detected and pre-selected during setup.

---

# Multi-Step Agentic Debugging

Pulse's AI analyst is designed to do more than explain an error.

It works through debugging tasks using a three-stage process:

```text
+-------------+
|  DESCRIBE   |
|             |
| Understand  |
| the problem |
+------+------+
       |
       v
+-------------+
|   DEVELOP   |
|             |
| Develop the |
| actual fix  |
+------+------+
       |
       v
+-------------+
|  IMPLEMENT  |
|             |
| Apply the   |
| developed   |
| solution    |
+-------------+
```

### 1. Describe

The agent analyzes the available code, tensor statistics, heatmaps, scalar curves, and training behavior to determine what is happening.

### 2. Develop

The agent develops a concrete solution, determining which changes are required rather than immediately modifying the code.

### 3. Implement

When instructed, the agent implements the developed solution directly into the code.

This creates a complete debugging workflow:

```text
Problem
   |
   v
Diagnosis
   |
   v
Solution
   |
   v
Implementation
```

The agent can therefore move beyond:

> "Your model appears unstable."

and toward:

> "This is the mechanism causing the instability, this is the mathematical reason, and this is the change required to fix it."

---

# Deterministic Math Verification

## LLMs should reason about math. They shouldn't be the calculator.

During ML debugging, an agent may need to calculate:

* Update magnitudes
* Ratios
* Scaling factors
* Normalization values
* Gradient relationships
* Numerical thresholds
* Other exact mathematical expressions

Rather than relying on the LLM to perform these calculations itself, Pulse provides a deterministic mathematical evaluation layer.

The agent can delegate an expression to the evaluator and use the exact result in its reasoning.

```text
             AI AGENT
                |
        +-------+-------+
        |               |
        v               v
    Reasoning       Math Check
        |               |
        |         Deterministic
        |           Evaluation
        |               |
        +-------+-------+
                |
                v
         Verified Result
```

The evaluator uses a restricted namespace containing mathematical operations and the Python `math` module while disabling builtins.

This gives the agent a reliable computational primitive for checking numerical claims instead of estimating them.

---

# Universal Backend Support

Pulse is not tied to a single ML framework.

| Backend    | Support |
| ---------- | ------- |
| NumPy      | Yes     |
| PyTorch    | Yes     |
| TensorFlow | Yes     |
| CuPy       | Yes     |
| JAX        | Yes     |

A shared backend abstraction allows Pulse to inspect and monitor tensors across different frameworks without requiring major changes to the user's training code.

---

# GUI

Pulse's GUI provides an interactive workflow for selecting and monitoring variables.

### Matrix Picker

Select tensors to monitor while seeing their shapes before tracking them.

### Live Dashboard

Monitor selected tensors through:

* Heatmap grids
* Scalar charts
* Tensor statistics
* Live updates
* AI analysis

### AI Analyst

Interact with the debugging agent directly alongside the live training data.

---

# CLI

Pulse also works in environments where a graphical interface isn't practical.

Designed for:

* Google Colab
* SSH
* Remote GPUs
* Headless servers

The CLI provides:

* Live tensor statistics
* ASCII scalar charts
* Matrix tracking
* Training pause/resume
* Adding variables while training
* AI analysis directly from the terminal
* Optional labeled PDF snapshots

---

# Performance

Instrumentation should not become the bottleneck.

Pulse is designed to minimize debugging overhead through:

* Matrix caching
* Host-side NumPy conversion
* Reusable Matplotlib figures
* `set_data()` updates instead of rebuilding plots
* Render sizes matched to actual thumbnails
* Selective tracking of monitored variables

The objective is simple:

**More visibility. Less overhead.**

---

# Real Debugging Examples

## Vocabulary Expansion Causing Training Instability

A custom LLM experienced training instability after a 2.5× vocabulary increase.

Pulse's diagnostics exposed a normalization problem where residual growth was divided by:

```text
sqrt(num_layers)
```

instead of:

```text
num_layers
```

This caused activation growth that eventually destabilized training and halted learning.

---

## Custom Attention Producing NaN Loss

Another debugging session involved a custom attention implementation producing NaN loss.

Pulse helped trace the failure to a missing infinity check before a division operation.

---

# Install

```bash
pip install pulseml
```

`tkinter` is required for GUI mode and ships with most Python installations.

On Debian/Ubuntu:

```bash
sudo apt install python3-tk
```

For CLI-mode PDF snapshots, `fpdf2` is installed automatically with the base package.

---

# AI Providers

Pulse supports multiple AI providers through environment variables:

```text
ANTHROPIC_API_KEY
OPENAI_API_KEY
GEMINI_API_KEY
DEEPSEEK_API_KEY
```

If no key is configured, Pulse can prompt for one through the GUI when the AI analyst is first used.

---

# The Goal

Pulse is being built toward a different kind of ML debugging workflow.

```text
              TRAINING
                  |
                  v
             OBSERVATION
                  |
                  v
              ANALYSIS
                  |
          +-------+-------+
          |               |
          v               v
      AI REASONING    EXACT MATH
          |               |
          +-------+-------+
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

The goal isn't simply to tell you that your model is broken.

**Pulse should help you determine why, verify the reasoning, develop the solution, and implement the fix.**

---

## License

Proprietary. See [`LICENSE`](/pulse-pkg/LICENSE).

Use of this software is governed by the terms in that file. Copying,
redistribution, and reverse engineering are not permitted.
