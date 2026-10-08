# Headless mode — operator's guide

Headless mode is for runs you want to start and leave. Pulse watches the run in the
background, with no screen, and when something goes wrong it fixes the code and restarts the
run by itself. You read what it did afterwards — in a log, in the app, or on the dashboard.

```bash
pulse run --headless train.py --epochs 3
```

The command returns at once and you can close the terminal.

---

## 1. Before the first run

Headless mode never asks anything once it is in the background, so two things have to be
set up beforehand. Both are one-time steps.

**An agent model** — the AI that diagnoses and fixes. Run `pulse` once and pick a model
(or sign in with OpenRouter, which needs no key). Pulse remembers the model and its key
(`pulse config` shows them). Without an agent, a headless run is still watched and
checked, but nothing gets fixed — the log says so at the top.

You can also name a model for one run: `PULSE_MODEL=openrouter/deepseek/deepseek-v4-flash
pulse run --headless train.py` (its key must be in the environment, e.g. `OPENROUTER_API_KEY`).

**Where the run shows up online** — asked the first time you start a headless run, in the
terminal, before it goes to the background:

1. if this machine is not signed in to Pulse Cloud: *"Sign in to Pulse Cloud, so this run
   shows up on your dashboard?"* — yes opens the usual sign-in or sign-up; no means the run
   is logged on this machine only;
2. the workspace picker;
3. the project picker.

The choice is remembered: the next `pulse run --headless` asks nothing.

---

## 2. Starting a run

| what you want | command |
|---|---|
| start a script | `pulse run --headless train.py` |
| with the script's own arguments | `pulse run --headless train.py --epochs 3 --lr 1e-3` |
| send it to another workspace and project (remembered) | `pulse run --headless --to "Lab/GPT small" train.py` |
| from another folder | `pulse run --headless --cwd ~/experiments train.py` |
| with a different reviewer model | `pulse run --headless --approver openrouter/anthropic/claude-sonnet-5 train.py` |

Options go **before** the script name. Everything after it goes to the script untouched.

`--to WORKSPACE/PROJECT` always names both halves. A workspace without a project is an
unfinished setup, so `--to Lab` alone is refused. If a name matches nothing, or more than one
thing, Pulse says what there is and starts nothing. You can type part of a name
(`--to "lab/gpt"`), as long as only one matches.

**From inside the script.** Call `auto_track(mode="headless")`, or set `PULSE_MODE=headless`
and call `auto_track()`. Here the script runs in your terminal as usual and the background
debugger watches it. `PULSE_TO="Lab/GPT small"` chooses where it goes.

There is one difference from `pulse run --headless`. Pulse did not start this process, so it
cannot restart it while it is running. A fix is applied to the code, and the run is started
again once the script has ended — without the script's original command-line arguments,
which Pulse does not know. If you want restarts with your arguments, use
`pulse run --headless`.

When the start succeeds, you see:

```
[Pulse] train.py is running, and Pulse is debugging it in the background (pid 12345).
[Pulse] You can close this terminal. If it crashes or goes wrong, the agent fixes the code and restarts it.
[Pulse] Online: workspace Lab, project GPT small -- https://pulsedashb.netlify.app/
[Pulse]   (remembered for next time; --to WORKSPACE/PROJECT, or `pulse config`, changes it)
[Pulse] Everything it does: ~/.pulse/headless/20261007-195732-train.py.log
[Pulse] Look at it: `pulse`, then /monitor · list: `pulse headless` · stop debugging: `pulse headless stop`
```

---

## 3. What it does on its own

While the run is going, Pulse reads its values a few times a second and runs its checks on
every new reading. On a schedule it chooses for itself (every 1 to 60 minutes), an AI also
reviews the whole run.

The agent is started **by itself** when:

- **the run crashes** — with the traceback as evidence;
- **a check finds something critical** — a NaN or infinite value, an exploding norm, a loss
  spike, a loss that never learned, accuracy stuck at chance;
- **the periodic review says "problem"** and does not rate it low risk.

Warnings — a plateau, a widening train/validation gap, a run that has gone quiet — are
written to the log and do **not** start the agent.

Each time it starts, the agent reads the code, explains the cause, fixes it, and decides
what to do with the run:

- **restart it** with the fix — after a crash, almost always;
- **stop it** if it is only burning compute;
- **leave it running** if the problem is harmless.

It says which one it chose, and why.

**Nobody is asked anything.** Every shell command the agent wants to run that touches
something important, and every code change, goes to a second AI, the reviewer. Anything the
reviewer does not approve is not done. If there is no reviewer, or it cannot answer, the
answer is no. The reviewer is your agent model unless you chose another (`--approver`, or
`pulse config approver <model>`). `pulse config approver off` means every change is refused
in headless mode, so nothing gets fixed.

Every change the agent applies is recorded with an id, and the log shows it ("/undo to revert
(commit c38cea19)"). To take a change back, open `pulse` in the project folder and type
`/undo c38cea19`, or just `/undo` for the latest change.

There is **no limit** on how many times it fixes and restarts. A run that keeps breaking in
new ways keeps getting fixed, and each fix costs model calls. Watch the log, or stop it
(section 5), if that is not what you want.

**When the run ends** — it finished, crashed and was not restarted, or the agent stopped it —
Pulse does one closing review, writes it to the log, and the background debugger exits.

---

## 4. Watching what it does

| | |
|---|---|
| `pulse headless` | what is being debugged in the background: pid, script, how long, log file |
| `~/.pulse/headless/<time>-<script>.log` | everything Pulse did — findings, the agent's diagnosis, each change (with its diff and the reviewer's verdict), restarts, the closing review |
| `~/.pulse/app-runs/<time>-<script>.log` | what your script itself printed (one file per start; a restart opens a new one) |
| `pulse`, then `/monitor` | open the run in the app, live: curves, findings, the agent. Your run is listed as live; open it and ask about it like any other |
| the [dashboard](https://pulsedashb.netlify.app/) | the run's row in your workspace and project: status, metrics, incidents, the agent's turns |

A typical log, from a run that crashed at step 120:

```
19:33:25  Pulse headless debugger started for train_crash.py (pid 2263192).
19:33:25  Agent: OpenRouter: deepseek/deepseek-v4-flash. It acts on crashes, serious findings and ...
19:33:32  == the agent is looking at a crash of train_crash.py
19:33:46    Read train_crash.py
19:33:53    change to train_crash.py:
19:33:53      -        bad = w @ X
19:33:53      +        bad = X @ w
19:33:58    change: approved by deepseek-v4-flash
19:33:58    change: applied
19:33:59    Restarted the run
19:34:13  agent: The run completed successfully (step 199/200, loss down to 0.371).
19:34:14  train_crash.py is over (finished); a closing audit.
19:35:54  Checked the run: The run appears fine. ...
19:36:14  Pulse headless debugger stopped.
```

---

## 5. Stopping

| | |
|---|---|
| `pulse headless stop` | stop the background debugging (when only one is running) |
| `pulse headless stop train.py` or `pulse headless stop 12345` | stop a particular one |

Stopping the debugger **does not stop your training**: the run carries on, unwatched. To stop
the training itself, open it in `pulse` (`/monitor`) and use `/stop`, or end the process as
you normally would.

---

## 6. Settings that matter here

`pulse config` shows every setting; `pulse config <name> <value>` changes one.

| setting | effect on headless runs |
|---|---|
| `agent` | the model that diagnoses and fixes (set it in `pulse`: `/config agent`) |
| `approver` | the reviewer: `same` (your agent model, the default), another model id, or `off` (every change is then refused) |
| `workspace` / `project` | where runs show up online; set them with `--to` or `/config workspace` |
| `remember_keys` | whether your API key is kept so the background process can use it; with `off`, the key must be in the environment when you start the run |

`review` and `audits` do not apply. In headless mode every change is always reviewed, and
the periodic review is always on.

---

## 7. Troubleshooting

**"No agent is set up" at the top of the log.** Run `pulse` once and pick a model, or start
with `PULSE_MODEL=...` and the model's key in the environment.

**"Not signed in to Pulse Cloud: this machine only."** You answered no to the sign-in, or the
start was not in a terminal (piped, or from a script with no terminal), where Pulse cannot
ask. Run `pulse run --headless` once from a terminal to sign in and choose a workspace and
project; after that, non-terminal starts use the remembered choice.

**`--to` refused.** Give both halves, `WORKSPACE/PROJECT`. If a name matches nothing or several
things, the message lists what exists.

**A change was "not applied".** The reviewer refused it, could not answer, or `approver` is
`off`. The log line next to it says which, and why. The agent is told the reason and may try
another way.

**The run never shows up in `pulse headless`.** The start failed: the script path was wrong,
or the run exited before Pulse saw its first step. The log path printed at the start
explains what happened, and the script's own output is in `~/.pulse/app-runs/`.

**Cost.** Each fix is several model calls, and each periodic review is one. With a small model
such as DeepSeek V4 Flash, a periodic review costs well under a cent. A run that crashes and
is fixed a few times costs a few cents.
