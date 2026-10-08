# Headless mode

Start a run and walk away. Pulse watches it in the background; if it crashes or goes badly
wrong, the agent fixes the code and restarts it. A reviewer AI approves every change.

## Setup (once)

1. Run `pulse` and pick an agent model.
2. The first `pulse run --headless` asks you to sign in and pick a workspace and project.
   It remembers them.

## Use

```bash
pulse run --headless train.py --epochs 3          # start; you can close the terminal
pulse run --headless --to "Lab/GPT small" train.py  # another workspace/project (remembered)
pulse headless                                    # list what is running
pulse headless stop                               # stop debugging (training keeps going)
```

From a script: `auto_track(mode="headless")`.

## See what it did

- Log: `~/.pulse/headless/<run>.log`
- Live: `pulse`, then `/monitor`
- Online: the [dashboard](https://pulsedashb.netlify.app/)
- Undo a fix: `/undo <id>` in `pulse` (the id is in the log)
