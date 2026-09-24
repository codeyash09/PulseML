import sys, time, os
os.environ["PULSE_ASYNC_MODEL_CALLS"] = "0"  # run check-ins synchronously so this test is deterministic
sys.path.insert(0, "/home/claude/PulseML_latest/pulse-pkg/src")
from pulse import pulse_cli as pc

def section(name):
    print(f"\n=== {name} ===")

# =====================================================================================
# Step tracking: tied to a real loop variable, not "increment every tick" or "increment
# only when the loss changes."
# =====================================================================================
section("Step tracking follows a real loop variable's actual advances")

cli = pc.PulseCLI()
cli.continuous = True
cli.script_path = "/tmp/does_not_matter.py"
cli.code_text = ""
cli.tracked_vars = ["loss"]  # so loss_var is not None -- forces the fallback chain to actually
                             # reach the loss-changed tier instead of the "no loss info at all"
                             # tier, which would fire regardless of loop-var detection.
assert cli.step == 0

# Tick 1: first-ever observation for this session -- nothing to diff against for either the
# loop var or the loss value, so the pre-existing "we now have a first real value" fallback
# reasonably counts this as step 1. This part is unchanged, deliberately -- only what happens
# from tick 2 onward (once there's a real baseline to compare against) is what changed.
cli.watch_locals = {"step": 0, "loss": 1.0}
cli.update()
assert cli.step == 1, f"expected the very first observation to establish step 1, got {cli.step}"

# Tick 2: loop var advanced by 1 -- should advance step by exactly 1, even though loss is
# UNCHANGED (this is exactly the case the old loss-only heuristic would silently miss).
cli.watch_locals = {"step": 1, "loss": 1.0}
cli.update()
assert cli.step == 2, f"expected step to follow the loop var to 2, got {cli.step}"

# Tick 3: the observation callback happens to fire only after 3 real iterations this time
# (loop var jumped from 1 to 4) -- the delta should be credited in full, not flattened to +1.
cli.watch_locals = {"step": 4, "loss": 1.0}
cli.update()
assert cli.step == 5, f"expected the real delta (3) to be credited on top of 2, got step={cli.step}"

# Tick 4: loop var UNCHANGED (still 4) but loss changed -- since a confident loop-var
# reading exists (delta 0, i.e. "no step happened"), it should NOT double-count via the old
# loss-change fallback.
cli.watch_locals = {"step": 4, "loss": 0.9}
cli.update()
assert cli.step == 5, f"step should not advance when the real loop var hasn't moved, got {cli.step}"

# Tick 5: loop var went DOWN (a new epoch/run reset it) -- must not decrement or misread this
# as a step; it should just re-baseline.
cli.watch_locals = {"step": 0, "loss": 0.9}
cli.update()
assert cli.step == 5, f"a loop-var reset should not move the step counter at all, got {cli.step}"
cli.watch_locals = {"step": 1, "loss": 0.9}
cli.update()
assert cli.step == 6, f"after re-baselining, the next real advance should count normally, got {cli.step}"

print("PASS -- step count tracks the real loop variable's actual advances: not too fast "
      "(no double counting, no counting a reset), not too slow (multi-step jumps and "
      "loss-unchanged steps both get credited).")

# An explicit passed-in step still wins outright (e.g. a Keras callback's own count).
cli.update(step=999)
assert cli.step == 999
print("PASS -- an explicitly-provided step still takes priority over loop-var detection.")

# =====================================================================================
# An implausible jump (some unrelated large integer, not a step counter) must not be
# mistaken for hundreds of steps happening in one tick.
# =====================================================================================
section("Implausible jumps are rejected, not credited as real steps")
cli2 = pc.PulseCLI()
cli2.continuous = True
cli2.script_path = "/tmp/x.py"
cli2.code_text = ""
cli2.tracked_vars = ["loss"]
cli2.watch_locals = {"i": 0, "loss": 1.0}
cli2.update()
assert cli2.step == 1, f"first observation establishes a baseline step of 1, got {cli2.step}"
cli2.watch_locals = {"i": 50_000, "loss": 1.0}  # e.g. actually a sample counter, not a step counter
cli2.update()
assert cli2.step == 1, f"an implausible single-tick jump must be rejected, got step={cli2.step}"
print("PASS -- an implausible jump was correctly rejected rather than credited as steps.")


# =====================================================================================
# Time-per-step is tracked and surfaced to the agent.
# =====================================================================================
section("Time-per-step is tracked and fed into the agent context")
cli3 = pc.PulseCLI()
cli3.continuous = True
cli3.script_path = "/tmp/y.py"
cli3.code_text = ""
cli3.tracked_vars = ["loss"]
cli3.watch_locals = {"step": 0, "loss": 1.0}
cli3.update()
time.sleep(0.05)
cli3.watch_locals = {"step": 1, "loss": 1.0}
cli3.update()
assert cli3._time_per_step_ema is not None and cli3._time_per_step_ema > 0
context = cli3._build_agent_context(include_code=False)
print(context.splitlines()[0])
assert f"step {cli3.step}" in context.splitlines()[0]
assert "per step" in context.splitlines()[0]
print("PASS -- time-per-step is tracked and shown at the top of the agent's context.")


# =====================================================================================
# Check-ins are scheduled on STEPS, not wall-clock time.
# =====================================================================================
section("Periodic check-ins fire on step count, not on a wall-clock timer")
cli4 = pc.PulseCLI()
cli4.continuous = True
cli4.script_path = "/tmp/z.py"
cli4.code_text = ""
cli4.tracked_vars = ["loss"]
cli4.agent_provider = "test"
cli4.agent_key = "test"
cli4.auto_intervene = False
cli4.checkin_interval_steps = 5
calls = []
cli4._call_model = lambda *a, **k: (calls.append(1) or "STATUS: ok\nGPUTRACK: none\nGPUUNTRACK: none\nNEXTCHECK: 50")

cli4.watch_locals = {"step": 0, "loss": 1.0}
cli4.update()  # first observation always establishes a baseline step (see the note above)
counter = 0
while cli4.step < cli4.checkin_interval_steps - 1:
    counter += 1
    cli4.watch_locals = {"step": counter, "loss": 1.0}
    cli4.update()
    assert len(calls) == 0, (
        f"must not check in before reaching the step interval, but a check-in fired at "
        f"step={cli4.step} (interval={cli4.checkin_interval_steps})"
    )
print(f"no check-in yet at step {cli4.step} (interval={cli4.checkin_interval_steps}) -- correct")

counter += 1
cli4.watch_locals = {"step": counter, "loss": 1.0}
cli4.update()
assert len(calls) == 1, f"expected exactly one check-in once the step interval was reached, got {len(calls)}"
print(f"check-in fired at step {cli4.step} -- correct")
assert cli4.checkin_interval_steps == 50, "the agent's NEXTCHECK: reply should update the step interval"
print("PASS -- the agent's NEXTCHECK: reply (now in steps) was applied.")

# A slow run that has taken almost no real steps must not be charged a check-in just
# because wall-clock time has passed -- there is no wall-clock timer left to fire at all.
section("A slow run isn't charged a check-in before it's taken more than a couple of steps")
cli5 = pc.PulseCLI()
cli5.continuous = True
cli5.script_path = "/tmp/slow.py"
cli5.code_text = ""
cli5.tracked_vars = ["loss"]
cli5.agent_provider = "test"
cli5.agent_key = "test"
calls5 = []
cli5._call_model = lambda *a, **k: (calls5.append(1) or "STATUS: ok\nGPUTRACK: none\nGPUUNTRACK: none\nNEXTCHECK: 500")
cli5.watch_locals = {"step": 0, "loss": 1.0}
cli5.update()
time.sleep(0.2)  # plenty of "wall-clock time" for the old 900s-based scheduler to still not fire,
cli5.watch_locals = {"step": 1, "loss": 1.0}  # but this also proves steps (not time) are what's being measured
cli5.update()
time.sleep(0.2)
cli5.watch_locals = {"step": 2, "loss": 1.0}
cli5.update()
assert len(calls5) == 0, "3 real steps is nowhere near the default 500-step interval; must not check in"
print(f"no check-in after {cli5.step} real steps despite real wall-clock time passing -- correct")

print("\nALL STEP-TRACKING AND CHECK-IN TESTS PASSED")
