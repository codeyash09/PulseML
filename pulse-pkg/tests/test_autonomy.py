"""Pulse Code's work-until-done loop and its context layer (notes, windowing, compaction)."""
import pytest

from pulse import pulse_code as pc
from pulse import pulse_code_agent as pca


@pytest.fixture(autouse=True)
def _legacy_step_loop(monkeypatch):
    # These exercise the step loop; the native tool-calling agent has its own tests.
    monkeypatch.setattr(pca, "supported", lambda cli: False)


class _Cli(pc._CodeAgentCLI):
    def __init__(self):                      # no setup, no network
        self.agent_history = []
        self._project_root = None
        self._last_applied_fix = None
        self.synced = []

    def _sync_agent_turn(self, request, summary, **kw):
        self.synced.append((request, summary))


def test_loop_continues_until_the_agent_says_done(monkeypatch):
    cli = _Cli()
    steps = iter(["a", "b", "c"])
    seen = []

    def fake_step(c, r, e=None):
        seen.append(r)
        return "applied", f"did {next(steps)}"

    follow_ups = iter(["step two", "step three", None])
    monkeypatch.setattr(pc, "_run_step", fake_step)
    monkeypatch.setattr(pc, "_next_step", lambda c, r: next(follow_ups))
    monkeypatch.setattr(cli, "compact_history", lambda force=False: False)
    assert pc.run_turn(cli, "build it") == "applied"
    assert len(seen) == 3 and seen[0] == "build it" and seen[1].startswith("step two")
    assert cli.synced and "did a" in cli.synced[0][1] and "did c" in cli.synced[0][1]


def test_loop_stops_when_a_step_is_declined(monkeypatch):
    cli = _Cli()
    monkeypatch.setattr(pc, "_run_step", lambda c, r, e=None: ("declined", "no"))
    monkeypatch.setattr(pc, "_next_step", lambda c, r: pytest.fail("must not continue after a decline"))
    monkeypatch.setattr(cli, "compact_history", lambda force=False: False)
    assert pc.run_turn(cli, "x") == "declined"


def test_loop_is_bounded(monkeypatch):
    cli = _Cli()
    calls = []
    monkeypatch.setattr(pc, "_run_step", lambda c, r, e=None: (calls.append(r), ("applied", "s"))[1])
    monkeypatch.setattr(pc, "_next_step", lambda c, r: "again")
    monkeypatch.setattr(cli, "compact_history", lambda force=False: False)
    assert pc.run_turn(cli, "x") == "applied"
    assert len(calls) == pc._MAX_AUTONOMOUS_STEPS


def test_project_notes_are_read_and_capped(tmp_path):
    cli = _Cli()
    cli._project_root = str(tmp_path)
    assert cli.project_notes() == (None, "")
    (tmp_path / "CLAUDE.md").write_text("use tabs", encoding="utf-8")
    assert cli.project_notes() == ("CLAUDE.md", "use tabs")
    (tmp_path / "PULSE.md").write_text("x" * (pc._PROJECT_NOTES_CHARS + 500), encoding="utf-8")
    name, text = cli.project_notes()
    assert name == "PULSE.md" and text.endswith("(truncated)")


def test_compaction_replaces_old_history_with_a_summary(monkeypatch):
    cli = _Cli()
    cli.agent_history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(30)]
    monkeypatch.setattr(pc._CodeAgentCLI, "_call_model", lambda self, *a, **k: "SUMMARY TEXT", raising=False)
    # _call_model is the override under test elsewhere; here only compact_history's use of it matters.
    assert cli.compact_history(force=True) is True
    assert len(cli.agent_history) == pc._KEEP_RECENT
    assert cli._context_summary == "SUMMARY TEXT"
    window = cli._context_history()
    assert "SUMMARY TEXT" in window[0]["content"] and window[-1]["content"] == "m29"


def test_small_history_is_not_compacted():
    cli = _Cli()
    cli.agent_history = [{"role": "user", "content": "hi"}] * 4
    assert cli.compact_history() is False


def test_pinned_request_stays_in_the_window():
    cli = _Cli()
    pinned = {"role": "user", "content": "FILES + request"}
    cli.agent_history = [pinned] + [{"role": "assistant", "content": f"r{i}"} for i in range(pc._WINDOW_MESSAGES + 5)]
    cli._turn_question_msg = pinned
    assert cli._context_history()[0] is pinned
