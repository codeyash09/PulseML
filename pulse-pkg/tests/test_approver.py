"""Auto mode: a second model answers the y/N for flagged shell commands."""
import os

import pytest

import pulse.pulse_cli as pc
from pulse import pulse_approver as ap
from pulse.pulse_cli import PulseCLI


def reply(text):
    class Msg:
        content = text
    class Choice:
        message = Msg()
    class Resp:
        choices = [Choice()]
    return Resp()


@pytest.fixture(autouse=True)
def no_approver_env(monkeypatch):
    for name in (ap.APPROVER_ENV, ap.APPROVER_KEY_ENV, ap.APPROVER_BASE_ENV):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("answer, approved, reason", [
    ("APPROVE: clears the run's own cache", True, "clears the run's own cache"),
    ("DENY: deletes the user's dataset", False, "deletes the user's dataset"),
    ("**DENY** — outside the project", False, "outside the project"),
    ("Let me think.\nAPPROVED - pip install of the missing package", True,
     "pip install of the missing package"),
    ("deny: sudo", False, "sudo"),
])
def test_parse(answer, approved, reason):
    decision = ap.parse(answer)
    assert decision.approved is approved and decision.reason == reason


def test_an_unreadable_answer_is_not_a_decision():
    with pytest.raises(ap.ApproverUnavailable):
        ap.parse("I would probably allow it")


def test_settings_come_from_env_then_config(monkeypatch):
    assert ap.settings_from(None) is None
    assert ap.settings_from("off") is None
    assert ap.settings_from("openrouter/x/y").model == "openrouter/x/y"
    s = ap.settings_from({"model": "openai/gpt-z", "api_key": "k1", "api_base": "http://b"})
    assert (s.model, s.api_key, s.api_base) == ("openai/gpt-z", "k1", "http://b")
    monkeypatch.setenv(ap.APPROVER_ENV, "openrouter/env/model")
    assert ap.settings_from("openrouter/x/y").model == "openrouter/env/model"


def test_a_same_provider_approver_reuses_the_agents_key():
    s = ap.settings_from("openrouter/a/b", agent_model="openrouter/deepseek/v4", agent_key="sk-agent")
    assert s.api_key == "sk-agent"
    other = ap.settings_from("anthropic/claude", agent_model="openrouter/deepseek/v4", agent_key="sk-agent")
    assert other.api_key is None          # litellm then reads that provider's own env var


def test_ask_sends_the_command_why_and_folder_and_retries_an_empty_reply():
    calls = []

    def completion(**kw):
        calls.append(kw)
        return reply("" if len(calls) == 1 else "APPROVE: needed to rebuild the cache")

    s = ap.ApproverSettings(model="m", api_key="k")
    d = ap.ask(s, "rm -rf .cache", "deletes files", "/proj", "loss is NaN", completion=completion)
    assert d.approved and len(calls) == 2
    prompt = calls[0]["messages"][1]["content"]
    assert "rm -rf .cache" in prompt and "deletes files" in prompt and "/proj" in prompt
    assert "loss is NaN" in prompt
    assert calls[0]["model"] == "m" and calls[0]["api_key"] == "k"


def test_ask_raises_unavailable_on_errors():
    def boom(**kw):
        raise TimeoutError("slow")
    with pytest.raises(ap.ApproverUnavailable):
        ap.ask(ap.ApproverSettings(model="m"), "x", "y", "/p", completion=boom)


def test_secrets_are_scrubbed_from_what_the_approver_sees():
    prompt = ap.build_prompt("curl -H 'Authorization: Bearer sk-ant-api03-" + "a" * 40 + "' x",
                             "reaches outside the project", "/p")
    assert "sk-ant-api03-" + "a" * 40 not in prompt


def cli_with_approver(monkeypatch, tmp_path, answer):
    monkeypatch.setenv(ap.APPROVER_ENV, "openrouter/judge/model")
    seen = []

    def completion(**kw):
        seen.append(kw)
        if isinstance(answer, Exception):
            raise answer
        return reply(answer)

    monkeypatch.setattr(pc.litellm, "completion", completion)
    cli = PulseCLI.__new__(PulseCLI)
    cli.config = {}
    cli.agent_provider = None
    cli.agent_key = None
    cli.agent_model_string = None
    asked = []
    monkeypatch.setattr(pc, "_prompt_text", lambda *a, **k: asked.append(1) or "n")
    monkeypatch.setattr(pc, "_flush_stdin", lambda: None)

    def unavailable(*args, **kwargs):
        raise pc._ui.Unavailable()

    monkeypatch.setattr(pc._ui, "choose", unavailable)

    class Executor:
        default_cwd = str(tmp_path)
    cli._get_terminal_executor = lambda: Executor()
    return cli, seen, asked


def test_auto_mode_answers_instead_of_the_person(monkeypatch, tmp_path):
    cli, seen, asked = cli_with_approver(monkeypatch, tmp_path, "APPROVE: removes the run's own checkpoints")
    assert cli._confirm_terminal_command("rm -rf checkpoints", {"deletes_files": True}) is True
    assert asked == [] and seen and seen[0]["model"] == "openrouter/judge/model"


def test_a_denial_reaches_the_agent_with_the_reason(monkeypatch, tmp_path):
    cli, _seen, _asked = cli_with_approver(monkeypatch, tmp_path, "DENY: that is the user's raw data")
    cli._terminal_needs_confirmation = lambda command: {"deletes_files": True}
    out = cli._run_terminal("rm -rf data/raw")
    assert "denied it: that is the user's raw data" in out
    assert "openrouter/judge/model" in out


def test_an_unreachable_approver_falls_back_to_asking(monkeypatch, tmp_path):
    cli, _seen, asked = cli_with_approver(monkeypatch, tmp_path, ConnectionError("down"))
    assert cli._confirm_terminal_command("rm x", {"deletes_files": True}) is False
    assert asked == [1]


def test_without_auto_mode_the_person_is_asked(monkeypatch, tmp_path):
    cli, seen, asked = cli_with_approver(monkeypatch, tmp_path, "APPROVE: x")
    monkeypatch.delenv(ap.APPROVER_ENV)
    cli._confirm_terminal_command("rm x", {"deletes_files": True})
    assert asked == [1] and seen == []


def test_terminal_approval_always_is_scoped_to_the_cli_session(monkeypatch, tmp_path):
    cli, _seen, _asked = cli_with_approver(monkeypatch, tmp_path, "unused")
    monkeypatch.delenv(ap.APPROVER_ENV)
    offered = []

    def choose(options, **kwargs):
        offered.extend(option.label for option in options)
        return 2

    monkeypatch.setattr(pc._ui, "choose", choose)
    assert cli._confirm_terminal_command("rm x", {"deletes_files": True}) is True
    assert cli._always_allow_terminal_commands is True
    assert any("Always allow" in label and "this session" in label for label in offered)

    monkeypatch.setattr(pc._ui, "choose", lambda *args, **kwargs: pytest.fail("asked again"))
    assert cli._confirm_terminal_command("rm y", {"deletes_files": True}) is True


def test_config_file_turns_auto_mode_on(monkeypatch, tmp_path):
    cli, seen, asked = cli_with_approver(monkeypatch, tmp_path, "APPROVE: fine")
    monkeypatch.delenv(ap.APPROVER_ENV)
    cli.config = {cli._config_key("approver"): "openrouter/from/config"}
    assert cli._confirm_terminal_command("rm x", {"deletes_files": True}) is True
    assert seen[0]["model"] == "openrouter/from/config" and asked == []


def test_run_flag_sets_the_approver(monkeypatch):
    from pulse import cli as entry
    entry._parse_run_args(["--approver", "openrouter/a/b", "train.py"])
    assert os.environ[ap.APPROVER_ENV] == "openrouter/a/b"
    entry._parse_run_args(["--approver=openrouter/c/d", "train.py"])
    assert os.environ[ap.APPROVER_ENV] == "openrouter/c/d"


def test_no_approver_chosen_means_the_agents_own_model_answers():
    s = ap.settings_from(None, agent_model="openrouter/deepseek/v4", agent_key="sk-agent",
                         agent_base=None)
    assert s.model == "openrouter/deepseek/v4" and s.api_key == "sk-agent" and s.same_as_agent


def test_approver_off_still_asks_the_person():
    assert ap.settings_from("off", agent_model="openrouter/deepseek/v4", agent_key="k") is None


def test_the_default_approver_does_not_see_the_agents_reasoning(monkeypatch, tmp_path):
    cli, seen, asked = cli_with_approver(monkeypatch, tmp_path, "APPROVE: rebuilds the cache")
    monkeypatch.delenv(ap.APPROVER_ENV)
    cli.agent_provider = "openrouter"
    cli.agent_model_string = "openrouter/deepseek/v4"
    cli.agent_key = "sk-agent"
    cli._agent_thoughts = "SECRET-REASONING the loss is NaN because ..."
    assert cli._confirm_terminal_command("rm -rf .cache", {"deletes_files": True}) is True
    assert asked == [] and seen[0]["model"] == "openrouter/deepseek/v4"
    sent = "\n".join(m["content"] for m in seen[0]["messages"])
    assert "SECRET-REASONING" not in sent and "rm -rf .cache" in sent


# ---- auto mode for code changes: the reviewer model answers "apply this change?" ---------

def test_review_change_fences_its_inputs_and_parses_the_verdict():
    seen = {}

    def completion(**kw):
        seen.update(kw)
        return reply("APPROVE: lowers the learning rate as asked")

    s = ap.ApproverSettings(model="m", api_key="k")
    d = ap.review_change(s, "lower the lr", "Lower lr to 0.05", "-lr = 0.3\n+lr = 0.05", "/proj", completion=completion)
    assert d.approved and d.reason == "lowers the learning rate as asked"
    prompt = seen["messages"][1]["content"]
    assert "| lower the lr" in prompt and "| +lr = 0.05" in prompt and "/proj" in prompt
    assert seen["messages"][0]["content"] == ap.CHANGE_SYSTEM_PROMPT and "undone" in ap.CHANGE_SYSTEM_PROMPT


def test_review_change_fails_closed_on_an_unreadable_answer():
    with pytest.raises(ap.ApproverUnavailable):
        ap.review_change(ap.ApproverSettings(model="m"), "r", "e", "d", "/p", completion=lambda **k: reply("Looks fine to me"))


def reviewing_cli(monkeypatch, tmp_path, verdict):
    cli, seen, asked = cli_with_approver(monkeypatch, tmp_path, verdict)
    cli._get_terminal_executor = lambda: type("E", (), {"default_cwd": str(tmp_path)})()
    cli.non_interactive = False
    return cli, seen, asked


def test_the_reviewer_decides_a_change_instead_of_the_person(monkeypatch, tmp_path):
    cli, seen, asked = reviewing_cli(monkeypatch, tmp_path, "APPROVE: as asked")
    assert cli._review_change_with_approver("add a flag", "adds --resume", "+parser.add_argument('--resume')") is True
    assert asked == [] and seen[0]["messages"][0]["content"] == ap.CHANGE_SYSTEM_PROMPT
    cli, seen, asked = reviewing_cli(monkeypatch, tmp_path, "DENY: deletes the whole training loop")
    assert cli._review_change_with_approver("add a flag", "rewrite", "-everything") is False
    assert "deletes the whole training loop" in cli._last_change_denial


def test_an_unreachable_reviewer_hands_the_change_to_the_person(monkeypatch, tmp_path):
    cli, seen, asked = reviewing_cli(monkeypatch, tmp_path, ConnectionError("down"))
    assert cli._review_change_with_approver("r", "e", "d") is None          # the caller asks the person
    cli.non_interactive = True
    assert cli._review_change_with_approver("r", "e", "d") is False        # nobody to ask: not applied
    assert "no user to ask" in cli._last_change_denial


def test_without_any_agent_there_is_no_reviewer(monkeypatch, tmp_path):
    cli, seen, asked = reviewing_cli(monkeypatch, tmp_path, "APPROVE: x")
    monkeypatch.delenv(ap.APPROVER_ENV)
    assert cli._review_change_with_approver("r", "e", "d") is None and seen == []
