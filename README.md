# PulseML

> **Pulse debugs ML.**

PulseML is an intelligent, interactive debugging and monitoring tool designed for machine learning workflows. It combines real-time metric tracking with an integrated AI analyst to help you understand training dynamics, diagnose anomalies, and troubleshoot code on the fly.

---

## Key Features

* **Flexible Matrix Tracking:** Easily select which matrices and metrics to track, or enable automatic tracking for all current and future variables (losses, errors, attention weights, layers, and custom counters).
* **Interactive Live Dashboard:** Visualize your training runs in real time with dynamic charts, line graphs, and customizable axis configurations.
* **Pulse AI Analyst:** Powered by Google AI Studio (Gemini), the built-in AI assistant can inspect your graphs and code context to answer complex training questions—such as analyzing loss plateaus, detecting NaN or infinity values, and validating convergence behavior.

---

## Screenshots

### Matrix Selection & Configuration
*Choose specific variables to monitor or track everything automatically before launching your run.*
<img width="1423" height="874" alt="Matrix Selection" src="https://github.com/user-attachments/assets/dc8f4047-3e63-4e2b-9d19-04ba54f27258" />

### Live Dashboard & AI Analyst
*Monitor live metrics while chatting directly with the AI Analyst to debug your model's performance in real time.*
<img width="1296" height="806" alt="Live Dashboard" src="https://github.com/user-attachments/assets/e9c5e13f-7d2c-4f26-8157-c296bd9ac745" />

---

## Installation & Usage

1. **Download the Pulse Folder** and place it in your project workspace.
2. **Import and call `auto_track()`** in your training script right before your training loop starts:

```python
from PULSE.pulse import auto_track

# Initialize auto-tracking before the loop starts
auto_track()

# Example training loop
for epoch in range(num_epochs):
    # Your training logic here
    pass
