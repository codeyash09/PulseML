"""Tests for the monitor/brain wire and the in-process monitor.

Run: python -m unittest discover -s pulse-pkg/tests -v
"""
import json
import os
import sys
import tempfile
import time
import unittest
import weakref

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pulse import pulse_stream as stream          # noqa: E402
from pulse.pulse_monitor import Monitor           # noqa: E402


class HostileTensor:
    """Looks like a tensor, explodes if anyone reads its data.

    The point of the monitor is that it answers questions about a value from its
    metadata. Any code path that converts this to numpy fails the test loudly.
    """

    def __init__(self, shape=(1024, 256), dtype="float32", device="cuda:0"):
        self.shape = shape
        self.dtype = dtype
        self.device = device

    def __array__(self, *a, **k):
        raise AssertionError("the monitor converted a tensor it was only supposed to describe")

    def numpy(self):
        raise AssertionError("the monitor called .numpy() on the training thread")

    def cpu(self):
        raise AssertionError("the monitor moved a tensor off the device")

    def detach(self):
        raise AssertionError("the monitor touched tensor data")

    def item(self):
        raise AssertionError("the monitor read an element out of a full-size tensor")


class ScalarTensor:
    """A 0-d tensor: small enough that reading it is allowed."""

    def __init__(self, value):
        self.shape = ()
        self.dtype = "float32"
        self.device = "cuda:0"
        self._value = value
        self.reads = 0

    def item(self):
        self.reads += 1
        return self._value


class StreamTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pulse-stream-")

    def test_frames_round_trip_in_order(self):
        writer = stream.StreamWriter(self.tmp, flush_seconds=0.01)
        for i in range(50):
            writer.emit(stream.KIND_SCALARS, {"step": i, "values": {"loss": 1.0 / (i + 1)}})
        writer.close()

        reader = stream.StreamReader(self.tmp)
        frames = [f for f in reader.poll() if f["kind"] == stream.KIND_SCALARS]
        self.assertEqual(len(frames), 50)
        self.assertEqual([f["step"] for f in frames], list(range(50)))
        self.assertEqual([f["seq"] for f in frames], sorted(f["seq"] for f in frames))
        self.assertEqual(reader.gaps, 0)

    def test_poll_returns_only_new_frames(self):
        writer = stream.StreamWriter(self.tmp, flush_seconds=0.01)
        reader = stream.StreamReader(self.tmp)
        writer.emit(stream.KIND_EVENT, {"event": "one"})
        time.sleep(0.1)
        first = [f for f in reader.poll() if f["kind"] == stream.KIND_EVENT]
        writer.emit(stream.KIND_EVENT, {"event": "two"})
        time.sleep(0.1)
        second = [f for f in reader.poll() if f["kind"] == stream.KIND_EVENT]
        writer.close()
        self.assertEqual([f["event"] for f in first], ["one"])
        self.assertEqual([f["event"] for f in second], ["two"])

    def test_partial_line_is_not_yielded_until_complete(self):
        path = os.path.join(self.tmp, "events.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"seq": 1, "kind": "event", "event": "whole"}) + "\n")
            handle.write('{"seq": 2, "kind": "event", "event": "half')   # no newline
        reader = stream.StreamReader(self.tmp)
        self.assertEqual([f["event"] for f in reader.poll()], ["whole"])
        with open(path, "a", encoding="utf-8") as handle:
            handle.write('"}\n')
        self.assertEqual([f["event"] for f in reader.poll()], ["half"])

    def test_backpressure_drops_and_reports_instead_of_blocking(self):
        writer = stream.StreamWriter(self.tmp, max_frames=8, flush_seconds=5.0)
        started = time.monotonic()
        for i in range(2000):
            writer.emit(stream.KIND_SCALARS, {"step": i, "values": {"loss": 0.1}})
        elapsed = time.monotonic() - started
        writer.flush_seconds = 0.01
        writer.close()
        self.assertLess(elapsed, 2.0, "emit() blocked under backpressure")

        reader = stream.StreamReader(self.tmp)
        reader.poll()
        self.assertGreater(reader.gaps, 0, "frames were dropped but the brain was not told")

    def test_rotation_is_survivable(self):
        writer = stream.StreamWriter(self.tmp, flush_seconds=0.01, max_bytes=2048)
        reader = stream.StreamReader(self.tmp)
        for i in range(400):
            writer.emit(stream.KIND_SCALARS, {"step": i, "values": {"loss": 1.0, "pad": "x" * 40}})
            if i % 50 == 0:
                time.sleep(0.02)
                reader.poll()
        time.sleep(0.1)
        reader.poll()
        writer.close()
        self.assertGreater(reader.rotations, 0, "the spool never rotated")
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "events.jsonl.prev")))

    def test_control_channel_round_trip(self):
        writer = stream.StreamWriter(self.tmp, flush_seconds=0.01)
        reader = stream.StreamReader(self.tmp)
        reader.send_control(stream.CONTROL_TRACK, names=["grads"])
        reader.send_control(stream.CONTROL_SET_INTERVAL, interval=2.0)
        messages = writer.poll_control()
        self.assertEqual([m["action"] for m in messages],
                         [stream.CONTROL_TRACK, stream.CONTROL_SET_INTERVAL])
        self.assertEqual(messages[0]["names"], ["grads"])
        self.assertEqual(writer.poll_control(), [], "a control message was delivered twice")
        writer.close()


class MonitorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pulse-monitor-")

    def _monitor(self, **kwargs):
        kwargs.setdefault("interval", 0.0)
        kwargs.setdefault("tensor_interval", 0.0)
        return Monitor(directory=self.tmp, session_id="test", **kwargs)

    def _frames(self, kind=None):
        reader = stream.StreamReader(self.tmp)
        frames = reader.poll()
        return [f for f in frames if kind is None or f["kind"] == kind]

    def test_never_touches_tensor_data(self):
        monitor = self._monitor()
        monitor.observe_locals({"activations": HostileTensor(), "loss": 0.5})
        monitor.close()
        tensors = self._frames(stream.KIND_TENSOR)
        self.assertEqual([t["name"] for t in tensors], ["activations"])
        self.assertEqual(tensors[0]["shape"], [1024, 256])
        self.assertEqual(tensors[0]["device"], "cuda:0")

    def test_small_tensors_are_read_big_ones_are_not(self):
        loss = ScalarTensor(0.25)
        monitor = self._monitor()
        monitor.observe_locals({"loss": loss, "weights": HostileTensor()})
        monitor.close()
        scalars = self._frames(stream.KIND_SCALARS)
        self.assertEqual(scalars[0]["values"]["loss"], 0.25)
        self.assertEqual(loss.reads, 1)

    def test_holds_no_reference_to_what_it_saw(self):
        monitor = self._monitor()
        tensor = HostileTensor()
        ref = weakref.ref(tensor)
        monitor.observe_locals({"activations": tensor})
        del tensor
        import gc
        gc.collect()
        self.assertIsNone(ref(), "the monitor pinned a tensor after the training loop dropped it")
        monitor.close()

    def test_sampling_interval_is_respected(self):
        monitor = Monitor(directory=self.tmp, session_id="test", interval=60.0, tensor_interval=60.0)
        for step in range(100):
            monitor.observe_locals({"loss": float(step)}, step=step)
        monitor.close()
        scalars = self._frames(stream.KIND_SCALARS)
        self.assertEqual(len(scalars), 1, "sampling interval ignored: %d frames" % len(scalars))

    def test_repeated_values_are_sent_not_deduplicated(self):
        # A value that has stopped moving is the signal, not noise: "unchanged for six
        # readings" is how a frozen loss is detected, and a stream that only carries
        # changes cannot express it.
        monitor = self._monitor()
        for step in range(5):
            monitor.observe_locals({"loss": 1.0, "lr": 0.01}, step=step)
        monitor.close()
        sent = [f["values"] for f in self._frames(stream.KIND_SCALARS)]
        self.assertEqual(len(sent), 5)
        self.assertTrue(all(v == {"loss": 1.0, "lr": 0.01} for v in sent))

    def test_step_is_taken_from_the_training_loop_counter(self):
        monitor = self._monitor()
        monitor.observe_locals({"step": 4096, "loss": 0.5})
        monitor.close()
        frames = self._frames(stream.KIND_SCALARS)
        self.assertEqual(frames[0]["step"], 4096)

    def test_nonfinite_raises_an_urgent_event(self):
        monitor = self._monitor()
        monitor.observe_locals({"loss": float("nan")}, step=7)
        monitor.close()
        events = [f for f in self._frames(stream.KIND_EVENT) if f["event"] == "nonfinite"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["value"], "nan")
        self.assertEqual(events[0]["step"], 7)
        self.assertTrue(events[0]["urgent"])

    def test_explicit_observe_records_every_call(self):
        monitor = Monitor(directory=self.tmp, session_id="test", interval=60.0)
        for step in range(10):
            monitor.observe("loss", 1.0 / (step + 1), step=step)
        monitor.close()
        scalars = self._frames(stream.KIND_SCALARS)
        self.assertEqual(len(scalars), 10, "explicit observe() must not be throttled")

    def test_probe_requests_are_answered_from_live_locals(self):
        import numpy as np
        monitor = self._monitor()
        stream.StreamReader(self.tmp).send_control(stream.CONTROL_SNAPSHOT, names=["weights"])
        monitor._last_control_poll = 0.0
        monitor.observe_locals({"weights": np.arange(12, dtype=np.float32).reshape(3, 4)})
        monitor.observe_locals({"weights": np.arange(12, dtype=np.float32).reshape(3, 4)})
        monitor.close()
        answered = [f for f in self._frames(stream.KIND_TENSOR) if f.get("stats")]
        self.assertEqual(len(answered), 1)
        self.assertEqual(answered[0]["name"], "weights")
        self.assertAlmostEqual(answered[0]["stats"]["max"], 11.0)

    def test_reports_its_own_cost(self):
        monitor = self._monitor()
        for step in range(20):
            monitor.observe_locals({"loss": float(step)}, step=step)
        cost = monitor.cost()
        monitor.close()
        self.assertEqual(cost["samples"], 20)
        self.assertGreaterEqual(cost["seconds_on_training_thread"], 0.0)
        self.assertIn("per_sample_ms", cost)


if __name__ == "__main__":
    unittest.main(verbosity=2)
