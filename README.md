# PulseML

> **Pulse — a live ML training debugger, GUI or CLI, any backend.**

https://pulsedb.netlify.app/

Pulse is a live machine learning training debugger designed to monitor tensors, track metrics, visualize heatmaps and line charts, and interact with an integrated AI analyst.

---

## Key Features

* **GUI Mode:** Opens an interactive matrix picker with live shapes, followed by a live dashboard with a heatmap grid and an integrated AI chat panel.
* **Smart Scalars:** Scalars (loss, accuracy, learning rate) automatically render as live step-charts rather than heatmaps. Loss-like scalars are auto-detected and pre-selected in the picker.
* **CLI Mode:** Built for Colab, SSH, or headless environments, printing tensor stats step-by-step, displaying live ASCII charts for scalars, and supporting optional labeled PDF snapshots.
* **Universal Backend Support:** Automatically detects and works with NumPy, PyTorch, TensorFlow, CuPy, and JAX via `pulse_backend.py`.
* **High Performance:** Keeps CPU usage low by converting tensors to host-side NumPy arrays, reusing Matplotlib figures (`set_data`) instead of rebuilding them every step, and matching render thumbnail sizes.
* **Pulse AI Analyst:** Context-aware chat panel briefed on its role that can inspect live matrix statistics, heatmaps, and your training code when "Send Code" is enabled.

---

## Screenshots

### Matrix Selection & Configuration
*Choose specific variables to monitor or track everything automatically before launching your run.*
<img width="1423" height="874" alt="Matrix Selection" src="https://github.com/user-attachments/assets/dc8f4047-3e63-4e2b-9d19-04ba54f27258" />

### Live Dashboard & AI Analyst
*Monitor live metrics while chatting directly with the AI Analyst to debug your model's performance in real time.*
<img width="1296" height="806" alt="Live Dashboard" src="https://github.com/user-attachments/assets/e9c5e13f-7d2c-4f26-8157-c296bd9ac745" />

---

## Installation

1. Download the **Pulse** folder and place it in your project workspace.
2. Install the required dependencies:
   ```bash
   pip install numpy matplotlib pillow litellm --break-system-packages

1. Ensure Tkinter is available (on Debian/Ubuntu:
   ```bash
   sudo apt install python3-tk)
2. Optional for CLI-mode PDF snapshots:
   ```bash
   pip install fpdf --break-system-packages

## Quickstart & Usage


In your training file, import auto_track from pulse and call it right before your training loop starts (Make sure your loop is wrapped in an if __name__ == '__main__':):
```python
from pulse.pulse import auto_track

# Pass your training function for shape discovery (optional), or call directly
auto_track()

# Your training loop
for epoch in range(num_epochs):
    # Training logic here
    pass

```
## AI Chat & API Keys
To enable the AI chat panel, set the relevant provider's API key as an environment variable (e.g., ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY), or leave it unset to be prompted inside the GUI the first time you send a message.
