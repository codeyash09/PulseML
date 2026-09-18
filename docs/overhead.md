# Before / after, same script, same steps (20k), same machine

Loop: 1024x256 numpy regression with a 1 MB activation local, loop in the same frame as
auto_track(), which is the usage the README documents.

## Work Pulse does on the training thread

| | single process | stream mode |
|---|---:|---:|
| `is_trackable` calls (every local, every traced line) | 67,112 | 0 |
| `to_numpy` (device sync + copy on a GPU tensor) | 264 | 0 |
| `statistics` (full array passes) | 60 | 0 |
| `PulseCLI.update()` on the training thread | 4 calls, 0.013 s | never runs there |
| blocking model calls on the training thread | see below | never |
| seconds of Pulse on the training thread | 0.059 | 0 |
| wall clock | 2.23 s | 1.90 s |

## With the agent enabled, which is the real cost

The same script with a model configured, single process:

| Measure | Value |
|---|---|
| wall clock | 15.9 s |
| `PulseCLI.update()` on the training thread | 13.39 s = 84% of the run |
| of which blocking model calls | 13.30 s (2 calls) |

In stream mode that work happens in the brain process. The training run does not wait
for it, so the same two model calls cost the run nothing.

## What the monitor costs, by its own accounting

From `state.json` of a real run: 0.18 ms per sample, 0.06% of wall clock, and that time
is spent on the sampler thread rather than the training thread.

## Caveats

* Wall clock on this machine is noisy (+-15%), which is why the operation counts are the
  headline and the seconds are supporting.
* This is a NumPy loop. `is_trackable` short-circuits on values with no `.shape`, so only
  204 of those 67,112 calls reached a conversion here. On a CUDA run every tensor local
  that reaches `describe_tensor` is a device sync, so the gap widens.
* The 84% figure includes real model latency and will vary with the provider.
