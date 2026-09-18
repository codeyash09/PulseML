"""Tests for the brain: stream ingest, the wake-up audit, and risk-adaptive scheduling."""
import json
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pulse import pulse_stream as stream                                    # noqa: E402
from pulse.pulse_brain import (Brain, MAX_INTERVAL_SECONDS, MIN_INTERVAL_SECONDS,  # noqa: E402
                               POST_FIX_INTERVAL_SECONDS, Schedule, downsample,
                               parse_decision)
from pulse.pulse_monitor import Monitor                                     # noqa: E402


class FakeAgent:
    """Records prompts, replies with whatever the test queued."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else '{"status": "ok", "next_check_minutes": 30}'


def reply(status="ok", risk="low", minutes=30, findings=(), text="Looks fine."):
    return text + "\n" + json.dumps({"status": status, "risk": risk, "findings": list(findings),
                                     "next_check_minutes": minutes, "reason": "because"})


class DecisionParsingTest(unittest.TestCase):
    def test_reads_the_json_after_prose(self):
        decision = parse_decision("Some analysis.\n" + json.dumps(
            {"status": "problem", "risk": "high", "next_check_minutes": 2}))
        self.assertEqual(decision["status"], "problem")
        self.assertEqual(decision["next_check_minutes"], 2)

    def test_ignores_json_that_is_not_the_decision(self):
        text = 'Consider {"lr": 0.001} in your code.\n{"status": "ok", "next_check_minutes": 45}'
        self.assertEqual(parse_decision(text)["next_check_minutes"], 45)

    def test_survives_a_reply_with_no_json(self):
        self.assertEqual(parse_decision("I think it is fine, check back later."), {})

    def test_survives_an_empty_reply(self):
        self.assertEqual(parse_decision(""), {})


class ScheduleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pulse-schedule-")
        self.path = os.path.join(self.tmp, "schedule.json")

    def test_interval_is_clamped_both_ways(self):
        schedule = Schedule(self.path)
        schedule.apply_decision({"next_check_minutes": 0.01})
        self.assertEqual(schedule.interval, MIN_INTERVAL_SECONDS)
        schedule.apply_decision({"next_check_minutes": 10000})
        self.assertEqual(schedule.interval, MAX_INTERVAL_SECONDS)

    def test_nonsense_leaves_the_interval_alone(self):
        schedule = Schedule(self.path)
        schedule.apply_decision({"next_check_minutes": 20})
        before = schedule.interval
        schedule.apply_decision({"next_check_minutes": "soon"})
        schedule.apply_decision({})
        self.assertEqual(schedule.interval, before)

    def test_bring_forward_never_pushes_out(self):
        schedule = Schedule(self.path)
        schedule.set(1800, "quiet run", "low")
        far = schedule.next_at
        schedule.bring_forward(60, "a critical check fired")
        self.assertLess(schedule.next_at, far)
        near = schedule.next_at
        schedule.bring_forward(3000, "nothing much")
        self.assertEqual(schedule.next_at, near, "bring_forward pushed the next look out")

    def test_survives_a_restart(self):
        schedule = Schedule(self.path)
        schedule.set(600, "agent asked for 10 minutes", "medium")
        reloaded = Schedule(self.path)
        self.assertEqual(reloaded.interval, 600)
        self.assertEqual(reloaded.risk, "medium")
        self.assertIn("10 minutes", reloaded.reason)

    def test_due_when_the_time_arrives(self):
        schedule = Schedule(self.path)
        schedule.set(MIN_INTERVAL_SECONDS, "x", "low")
        self.assertFalse(schedule.due())
        self.assertTrue(schedule.due(now=time.time() + MIN_INTERVAL_SECONDS + 1))


class DownsampleTest(unittest.TestCase):
    def test_short_histories_pass_through(self):
        self.assertEqual(len(downsample([1.0, 2.0, 3.0], points=10)), 3)

    def test_a_spike_survives_compression(self):
        values = [1.0] * 1000
        values[500] = 99.0
        buckets = downsample(values, points=50)
        self.assertEqual(len(buckets), 50)
        self.assertTrue(any(b["max"] > 90 for b in buckets),
                        "a mean-only downsample would have hidden the spike")


class BrainIngestTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pulse-brain-")

    def _write(self, frames):
        writer = stream.StreamWriter(self.tmp, flush_seconds=0.01)
        for kind, payload in frames:
            writer.emit(kind, payload)
        writer.close()

    def test_builds_histories_from_the_stream(self):
        self._write([(stream.KIND_SCALARS, {"step": i, "values": {"loss": 1.0 / (i + 1)}})
                     for i in range(20)])
        brain = Brain(self.tmp)
        brain.poll_once()
        self.assertEqual(len(brain.histories["loss"]), 20)
        self.assertEqual(brain.step, 19)

    def test_detects_from_streamed_data(self):
        frames = [(stream.KIND_SCALARS, {"step": i, "values": {"loss": 2.3}}) for i in range(15)]
        self._write(frames)
        brain = Brain(self.tmp)
        brain.poll_once()
        brain.ingest([])        # a second evaluation, for the confirmation gate
        checks = {f.check for f in brain.engine.current()}
        self.assertTrue({"frozen", "never_learned"} & checks, f"nothing detected: {checks}")

    def test_a_critical_finding_brings_the_next_look_forward(self):
        brain = Brain(self.tmp)
        brain.schedule.set(MAX_INTERVAL_SECONDS, "quiet", "low")
        far = brain.schedule.next_at
        self._write([(stream.KIND_SCALARS, {"step": 1, "values": {"loss": float("nan")}})])
        brain.poll_once()
        self.assertLess(brain.schedule.next_at, far)

    def test_an_urgent_monitor_event_brings_the_next_look_forward(self):
        brain = Brain(self.tmp)
        brain.schedule.set(MAX_INTERVAL_SECONDS, "quiet", "low")
        far = brain.schedule.next_at
        self._write([(stream.KIND_EVENT, {"event": "nonfinite", "name": "loss", "urgent": True})])
        brain.poll_once()
        self.assertLess(brain.schedule.next_at, far)

    def test_notices_dropped_frames(self):
        writer = stream.StreamWriter(self.tmp, max_frames=4, flush_seconds=5.0)
        for i in range(500):
            writer.emit(stream.KIND_SCALARS, {"step": i, "values": {"loss": 1.0}})
        writer.flush_seconds = 0.01
        writer.close()
        brain = Brain(self.tmp)
        brain.poll_once()
        self.assertGreater(brain.evidence()["stream_gaps"], 0)

    def test_monitor_to_brain_end_to_end(self):
        monitor = Monitor(directory=self.tmp, session_id="e2e", interval=0.0, tensor_interval=0.0)
        for step in range(30):
            monitor.observe_locals({"loss": 1.0 / (step + 1), "accuracy": 0.5 + step / 100.0}, step=step)
        monitor.close()
        brain = Brain(self.tmp)
        brain.poll_once()
        self.assertEqual(len(brain.histories["loss"]), 30)
        self.assertTrue(brain.finished)
        self.assertEqual(brain.engine.current(), [], "a healthy run produced findings")


class AuditTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pulse-audit-")
        writer = stream.StreamWriter(self.tmp, flush_seconds=0.01)
        for i in range(60):
            writer.emit(stream.KIND_SCALARS, {"step": i, "values": {"loss": 1.0 / (i + 1) ** 0.5}})
        writer.close()

    def test_audit_prompt_contains_the_whole_run_not_just_the_last_value(self):
        agent = FakeAgent(reply())
        brain = Brain(self.tmp, agent=agent)
        brain.poll_once()
        brain.audit(include_code=False)
        prompt = agent.prompts[0]
        self.assertIn("60 readings", prompt)
        self.assertIn("Tracked values", prompt)
        self.assertIn("deterministic checks", prompt)
        self.assertIn("they cannot do", prompt)

    def test_audit_applies_the_agents_interval(self):
        agent = FakeAgent(reply(minutes=7, risk="medium"))
        brain = Brain(self.tmp, agent=agent)
        brain.audit(include_code=False)
        self.assertEqual(brain.schedule.interval, 420)
        self.assertEqual(brain.schedule.risk, "medium")

    def test_a_worried_agent_gets_a_shorter_leash_than_a_calm_one(self):
        worried = Brain(self.tmp, agent=FakeAgent(reply(status="problem", risk="high", minutes=2)))
        calm = Brain(self.tmp, agent=FakeAgent(reply(status="ok", risk="low", minutes=55)))
        worried.audit(include_code=False)
        calm.audit(include_code=False)
        self.assertLess(worried.schedule.interval, calm.schedule.interval)

    def test_audit_failure_does_not_stop_monitoring(self):
        class Boom:
            def __call__(self, prompt):
                raise RuntimeError("provider down")
        brain = Brain(self.tmp, agent=Boom())
        result = brain.audit(include_code=False)
        self.assertEqual(result["status"], "error")
        self.assertGreaterEqual(brain.schedule.interval, MIN_INTERVAL_SECONDS)
        self.assertTrue(brain.schedule.next_at > time.time())

    def test_audit_without_an_agent_is_skipped_not_faked(self):
        brain = Brain(self.tmp)
        self.assertEqual(brain.audit()["status"], "skipped")

    def test_a_fix_shortens_the_interval(self):
        brain = Brain(self.tmp, agent=FakeAgent(reply(minutes=55)))
        brain.audit(include_code=False)
        self.assertEqual(brain.schedule.interval, 55 * 60)
        brain.note_fix("changed the learning rate on line 42")
        self.assertLessEqual(brain.schedule.seconds_remaining(), POST_FIX_INTERVAL_SECONDS + 1)

    def test_fixes_are_shown_to_the_next_audit(self):
        agent = FakeAgent(reply(), reply())
        brain = Brain(self.tmp, agent=agent)
        brain.note_fix("set dropout back to 0.2")
        brain.audit(include_code=False)
        self.assertIn("set dropout back to 0.2", agent.prompts[-1])

    def test_previous_audits_are_shown_to_the_next_one(self):
        agent = FakeAgent(reply(status="watch", risk="medium", minutes=5), reply())
        brain = Brain(self.tmp, agent=agent)
        brain.audit(include_code=False)
        brain.audit(include_code=False)
        self.assertIn("previous audits concluded", agent.prompts[-1])

    def test_findings_reach_the_prompt(self):
        writer = stream.StreamWriter(self.tmp, flush_seconds=0.01)
        for i in range(12):
            writer.emit(stream.KIND_SCALARS, {"step": 100 + i, "values": {"val_loss": 1.0 + i * 0.2}})
        writer.close()
        agent = FakeAgent(reply())
        brain = Brain(self.tmp, agent=agent)
        brain.poll_once()
        brain.ingest([])
        brain.audit(include_code=False)
        self.assertIn("val_loss", agent.prompts[-1])
        self.assertIn("deterministic checks currently report", agent.prompts[-1])

    def test_dropped_frames_are_disclosed_to_the_agent(self):
        brain = Brain(self.tmp, agent=FakeAgent(reply()))
        brain.poll_once()
        brain.reader.gaps = 17
        brain.audit(include_code=False)
        self.assertIn("17 sample(s) were dropped", brain.audits[-1]["text"] and brain.agent.prompts[-1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
