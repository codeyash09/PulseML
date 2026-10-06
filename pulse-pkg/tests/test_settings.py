"""Remembered settings (pulse_settings): the next start does not ask again."""
import json
import os
import stat
import types

import pytest

from pulse import pulse_cli as pc
from pulse import pulse_settings as settings
from pulse import pulse_supabase as cloud


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(cloud, "CACHE_PATH", tmp_path / "credentials.json")
    for name in ("PULSE_PROVIDER", "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY", "PULSE_LOCAL_MODEL"):
        monkeypatch.delenv(name, raising=False)
    saved = dict(pc.PROVIDERS)
    yield tmp_path
    pc.PROVIDERS.clear()
    pc.PROVIDERS.update(saved)


def test_defaults_set_and_unset(home):
    assert settings.get("mouse") == "on" and settings.on("review") and not settings.is_set("mouse")
    settings.set("mouse", "off")
    assert settings.get("mouse") == "off" and json.loads((home / "settings.json").read_text()) == {"mouse": "off"}
    settings.unset("mouse")
    assert settings.get("mouse") == "on"
    assert settings.check("mouse", "maybe") == "mouse is on or off"
    assert "no setting called" in settings.check("colour", "red")


def test_a_key_is_kept_owner_only_and_never_beats_the_environment(home, monkeypatch):
    assert settings.save_key("DEEPSEEK_API_KEY", "sk-saved")
    mode = stat.S_IMODE(os.stat(home / "keys.json").st_mode)
    assert mode == 0o600
    assert "sk-saved" not in (home / "settings.json").read_text() if (home / "settings.json").exists() else True
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-env")
    settings.load_keys_into_environment()
    assert os.environ["DEEPSEEK_API_KEY"] == "sk-from-env"
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    assert settings.load_keys_into_environment() == ["DEEPSEEK_API_KEY"] and os.environ["DEEPSEEK_API_KEY"] == "sk-saved"


def test_no_key_is_written_when_keys_are_not_remembered(home):
    settings.set("remember_keys", "off")
    assert not settings.save_key("DEEPSEEK_API_KEY", "sk-x") and not (home / "keys.json").exists()


def _cli():
    cli = pc.PulseCLI.__new__(pc.PulseCLI)
    cli.non_interactive = False
    cli.agent_provider = cli.agent_key = cli.agent_model_string = cli.agent_api_base = None
    cli.agent_history = []
    cli.config = {}
    return cli


def test_the_picked_agent_is_used_at_the_next_start_without_a_question(home, monkeypatch):
    name = next(n for n, info in pc.PROVIDERS.items() if info.get("env_key") == "DEEPSEEK_API_KEY")
    first = _cli()
    first.agent_provider, first.agent_key = name, "sk-picked"
    settings.remember_agent(first)
    assert settings.get("agent") == {"provider": name} and settings.saved_key("DEEPSEEK_API_KEY") == "sk-picked"

    monkeypatch.setattr(pc, "_ui_pick_agent", lambda *a, **k: pytest.fail("asked again"))
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("asked again"))
    second = _cli()
    assert settings.apply_agent(second)
    assert second.agent_provider == name and second.agent_key == "sk-picked"
    assert "PULSE_PROVIDER" not in os.environ                          # the environment is put back


def test_an_openrouter_model_is_remembered_by_its_slug(home, monkeypatch):
    first = _cli()
    first.agent_provider = pc.register_openrouter_model("openrouter/deepseek/deepseek-v4-flash")
    first.agent_key = "sk-or"
    settings.remember_agent(first)
    record = settings.get("agent")
    assert record["model"] == "openrouter/deepseek/deepseek-v4-flash"
    second = _cli()
    assert settings.apply_agent(second) and "deepseek-v4-flash" in second.agent_provider


def test_a_saved_agent_without_its_key_falls_back_to_asking(home):
    name = next(n for n, info in pc.PROVIDERS.items() if info.get("env_key") == "DEEPSEEK_API_KEY")
    settings.set("agent", {"provider": name})
    assert not settings.apply_agent(_cli())


def test_pulse_config_on_the_command_line(home, capsys):
    assert settings.main([]) == 0 and "mouse" in capsys.readouterr().out
    assert settings.main(["mouse", "off"]) == 0 and settings.get("mouse") == "off"
    assert settings.main(["mouse", "sideways"]) == 2
    assert settings.main(["agent", "x"]) == 2 and "/config agent" in capsys.readouterr().out
    settings.save_key("DEEPSEEK_API_KEY", "sk-1")
    assert settings.main(["remember_keys", "off"]) == 0 and not (home / "keys.json").exists()
    assert settings.main(["reset"]) == 0 and not (home / "settings.json").exists()


def test_the_remembered_workspace_skips_the_picker(home, monkeypatch):
    cli = _cli()
    cli.user_id, cli.email, cli.team_id = "u1", "a@b", None
    cli.team_join_code, cli.team_admin_ids = None, []
    teams = [{"team_id": "t1", "join_code": "AAA"}, {"team_id": "t2", "join_code": "BBB"}]
    monkeypatch.setattr(cloud, "find_teams_for_user", lambda uid: teams)
    monkeypatch.setattr(cloud, "save_cached_credentials", lambda *a, **k: None)
    monkeypatch.setattr(pc, "_ui_workspace_menu", lambda *a, **k: pytest.fail("the picker was shown"))
    settings.set("workspace", {"id": "t2", "name": "BBB"})
    cli._team_flow()
    assert cli.team_id == "t2"
