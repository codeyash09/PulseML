# PulseML

**Pulse** — a live ML training debugger, GUI or CLI, any backend.

🔗 [pulsedb.netlify.app](https://pulsedb.netlify.app/)

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/pulseml?period=total&units=NONE&left_color=BLACK&right_color=GREY&left_text=downloads)](https://pepy.tech/projects/pulseml)

Pulse is a live machine learning training debugger designed to monitor
tensors, track metrics, visualize heatmaps and line charts, and interact
with an integrated AI analyst.

## Key Features

- **GUI Mode** — Opens an interactive matrix picker with live shapes,
  followed by a live dashboard with a heatmap grid and an integrated AI
  chat panel.
- **Smart Scalars** — Scalars (loss, accuracy, learning rate)
  automatically render as live step-charts rather than heatmaps.
  Loss-like scalars are auto-detected and pre-selected in the picker.
- **CLI Mode** — Built for Colab, SSH, or headless environments, printing
  tensor stats step-by-step, displaying live ASCII charts for scalars,
  and supporting optional labeled PDF snapshots. Supports pausing
  training so you can tag new matrices or ask the AI to interpret
  results, right from the terminal.
- **Universal Backend Support** — Automatically detects and works with
  NumPy, PyTorch, TensorFlow, CuPy, and JAX via a shared backend
  abstraction layer.
- **High Performance** — Keeps overhead low by converting tensors to
  host-side NumPy arrays, reusing Matplotlib figures (`set_data`) instead
  of rebuilding them every step, and matching render sizes to the actual
  on-screen thumbnail.
- **Pulse AI Agent** — A dedicated agent that not only will suggest, find,
  and explain fixes and errors in your code using heatmap and scalar data,
  but it will also implement the fix if directly prompted while also providing
  an instant undo button in order to make sure that the changes don't
  negatively impact the code.
## Past Debugs

- Debugged a custom LLM after a vocab size increase (2.5x) by catching a
  normalization bug — dividing residual growth by `math.sqrt(num_layers)`
  instead of `num_layers` — that let activations blow up and halted
  training.
- Debugged another developer's custom attention mechanism producing NaN
  loss, tracing it to a missing infinity check before a division.

See [pulsedb.netlify.app](https://pulsedb.netlify.app/) for screenshots
of matrix selection, the live dashboard, and the CLI view.

## Install

```bash
pip install pulseml
```

`tkinter` is required for GUI mode and ships with most Python installs.
On Debian/Ubuntu, if it's missing:

```bash
sudo apt install python3-tk
```

For CLI-mode PDF snapshots, `fpdf2` is installed automatically as part of
the base package.

## Quickstart

Import `auto_track` and call it right before your training loop starts.
Make sure your loop is wrapped in `if __name__ == '__main__':`.

```python
from pulse import auto_track

if __name__ == '__main__':
    auto_track()   # pass your training function for shape discovery, or call directly

    # Your training loop
    for epoch in range(num_epochs):
        # Training logic here
        pass
```

## AI Chat & API Keys

To enable the AI chat panel, set the relevant provider's API key as an
environment variable (e.g. `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`GEMINI_API_KEY`, `DEEPSEEK_API_KEY`) — or leave it unset and Pulse will
prompt you for one inside the GUI the first time you send a message.

## License

Proprietary. See `LICENSE`. Use of this software is governed by the
terms in that file — copying, redistribution, and reverse engineering
are not permitted.
