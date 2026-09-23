# Streaming a Keras run

Pulse reads a training run in one of two ways, and Keras is the case where the
difference matters.

**In-process (`mode="cli"` / `"ui"`)** Pulse patches `Model.fit` and adds a callback of
its own, so `loss`, `val_loss` and every metric you compiled with arrive automatically.
Nothing to do.

**Streaming (`mode="stream"`)** the monitor samples the *local variables* of your
training loop and writes them to a spool that a separate brain process reads. That is
what makes streaming cheap — it never blocks the run — and it is also why Keras needs
one extra line:

> **Keras does not put its metrics in local variables.** `loss` and `val_loss` exist
> only inside the `logs` dict that Keras hands to callbacks. A sampler looking at frame
> locals sees the model, the optimizer, the epoch counter — and no loss at all.

A streamed Keras run with nothing added is therefore a run Pulse watches faithfully and
reports nothing about. It has happened: sixteen runs in a row produced no findings, and
the detectors were not at fault — the numbers never reached them.

## The one line

```python
from pulse import auto_track, pulse_monitor

monitor = auto_track(mode="stream")          # returns the monitor

class PulseStream(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        for name, value in (logs or {}).items():
            monitor.observe(name, value, step=epoch)

model.fit(train, validation_data=val, epochs=60, callbacks=[PulseStream()])
```

`monitor.observe(name, value, step)` is the explicit path into the same stream the
sampler writes to. Everything downstream — the detectors, the console, the agent's
scheduled audit — treats those readings exactly like any others.

Names matter more than you would expect. The detectors key off them: anything containing
`loss` is read as lower-is-better, `val_`/`test_`/`eval_` marks a validation curve,
`accuracy`/`f1`/`auc`/`reward` as higher-is-better. Keras' own names (`loss`,
`val_loss`, `accuracy`, `val_accuracy`) are already right, so passing `logs` straight
through is the correct thing to do. Renaming them to something Pulse cannot classify is
how a curve ends up watched but never checked.

### Per-batch, not per-epoch

An epoch is one reading. On a sixty-epoch run that is sixty points, and several checks
need eight to twelve before they will say anything — so a fault in the first few epochs
is not reported until it is well established. If your epochs are long, stream batches
too:

```python
    def on_train_batch_end(self, batch, logs=None):
        if batch % 50 == 0:                  # often enough to see, rare enough to be free
            monitor.observe("loss", (logs or {}).get("loss"), step=self._global_step)
            self._global_step += 1
```

### Things worth streaming that Keras will not give you

`logs` carries what you compiled the model with. These are usually the difference
between "the loss is not moving" and knowing why, and the detectors have checks for all
of them:

```python
    def on_epoch_end(self, epoch, logs=None):
        for name, value in (logs or {}).items():
            monitor.observe(name, value, step=epoch)
        monitor.observe("lr", float(self.model.optimizer.learning_rate), step=epoch)
        monitor.observe("weight_norm",
                        float(tf.linalg.global_norm(self.model.trainable_weights)),
                        step=epoch)
```

A `lr` curve turns "it stopped learning at epoch 20" into "the schedule reached zero at
epoch 20". A `weight_norm` catches a run whose weights are growing without bound long
before the loss shows it.

## Watching it

```bash
python train.py            # in one terminal
pulse                      # in another: finds the run and attaches
```

Or `pulse run --stream train.py` to start it under Pulse without editing the script --
though on a Keras run you still need the callback above, because `pulse run` only adds
`auto_track()` for you; it cannot reach inside `fit`.

## Checking it is working

The cheapest check is `pulse sessions` — if the run is listed with a step count that
climbs, the readings are arriving. If it shows `no steps`, the callback is not wired up.

If you drive a `Brain` yourself to check, note that most findings need the same belief
on two consecutive evaluations before they are raised, so a single `poll_once()` on a
finished spool reports nothing even when something is plainly wrong. That is the
confirmation gate, not a broken pipeline:

```python
brain = Brain(directory)
brain.poll_once()
for _ in range(3):
    brain.ingest([])        # further evaluations, as the live loop does
print(brain.engine.current())
```

`detectbench/wiring.py` exercises this path directly: it drives the real ingestion
function with the log dicts Keras actually emits and asks the real detector whether it
noticed. It needs no TensorFlow, because what is under test is Pulse's plumbing rather
than Keras'.

## Why this is not automatic

It could be: the stream monitor could patch `Model.fit` the way the CLI path does. It
does not today, because `auto_track(mode="stream")` returns before any of the Keras code
runs —

```python
if _stream_mode_requested(mode):
    return _start_stream_monitor(sys._getframe(1), throttle_interval)
```

— and the Keras bridge (`_install_keras_fit_hook`) lives in `pulse_cli.py`, on the other
side of that return. `pulse_monitor.py` and `pulse_brain.py` contain no reference to
Keras at all. Worth fixing; until it is, the callback above is the supported way.
