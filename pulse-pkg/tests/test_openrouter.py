"""OpenRouter sign-in: a browser sign-in / sign-up that hands Pulse an API key, next to the
existing "paste a key" path. Nothing here reaches OpenRouter: the two requests are faked."""
import base64
import hashlib
import json
import os
import stat
import sys
import threading
import types
import urllib.error
import urllib.parse
import urllib.request

import pytest

import pulse.pulse_cli as pc
from pulse import cli as entry
from pulse import pulse as core
from pulse import pulse_brain
from pulse import pulse_openrouter as orr
from pulse import pulse_supabase as cloud
from pulse.pulse_cli import PulseCLI

KEY = "sk-or-v1-" + "a1b2c3d4" * 8


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """No real key, no real ~/.pulse, and no request leaves the machine."""
    saved_env = dict(os.environ)
    saved_providers = dict(pc.PROVIDERS)
    for name in (orr.ENV_KEY, "PULSE_PROVIDER", "PULSE_OPENROUTER_SAVED_KEY", "PULSE_OPENROUTER_MODEL",
                 "PULSE_OPENROUTER_NO_BROWSER", "PULSE_APPROVER", "PULSE_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cloud, "CACHE_PATH", tmp_path / "pulsehome" / "credentials.json")
    monkeypatch.setattr(cloud, "save_cached_profile", lambda **k: None)

    def no_network(*a, **k):
        raise AssertionError("a test reached OpenRouter")

    monkeypatch.setattr(orr, "_request_json", no_network)
    yield
    os.environ.clear()
    os.environ.update(saved_env)
    pc.PROVIDERS.clear()
    pc.PROVIDERS.update(saved_providers)


def fake_openrouter(monkeypatch, good_code="good-code", key=KEY):
    """Stand in for the code exchange; records what was sent."""
    calls = []

    def request(url, payload=None, headers=None, timeout=30.0):
        calls.append({"url": url, "payload": payload, "headers": headers})
        if url == orr.EXCHANGE_URL:
            if payload["code"] == good_code:
                return 200, {"key": key}
            return 400, {"error": {"message": "Invalid code", "code": 400}}
        if url == orr.KEY_INFO_URL:
            return 200, {"data": {"label": "Pulse", "usage": 0.0, "limit": None, "is_free_tier": True}}
        raise AssertionError(url)

    monkeypatch.setattr(orr, "_request_json", request)
    return calls


def params_of(url):
    return {k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlparse(url).query).items()}


# ---- PKCE and the exchange ---------------------------------------------------------------

def test_challenge_is_the_base64url_sha256_of_the_verifier():
    verifier, challenge = orr.new_pkce()
    expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    assert challenge == expected and "=" not in challenge and len(verifier) >= 43
    assert orr.new_pkce()[0] != verifier


def test_authorize_url_with_and_without_a_callback():
    with_cb = params_of(orr.authorize_url("CH", "http://localhost:5123/callback"))
    assert with_cb == {"callback_url": "http://localhost:5123/callback", "code_challenge": "CH",
                       "code_challenge_method": "S256", "key_label": "Pulse"}
    assert orr.authorize_url("CH").startswith("https://openrouter.ai/auth?")
    assert "callback_url" not in params_of(orr.authorize_url("CH"))


def test_exchange_sends_code_and_verifier_and_returns_the_key(monkeypatch):
    calls = fake_openrouter(monkeypatch)
    assert orr.exchange(" good-code ", "VER") == KEY
    assert calls[0]["payload"] == {"code": "good-code", "code_verifier": "VER", "code_challenge_method": "S256"}


def test_a_rejected_code_is_explained(monkeypatch):
    fake_openrouter(monkeypatch)
    with pytest.raises(orr.SignInError, match="Invalid code"):
        orr.exchange("wrong", "VER")
    with pytest.raises(orr.SignInError):
        orr.exchange("", "VER")


def test_request_reports_an_unreachable_server(monkeypatch):
    monkeypatch.undo()                      # the real _request_json, but with urlopen failing

    def refuse(*a, **k):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    with pytest.raises(orr.SignInError, match="could not reach OpenRouter"):
        orr._request_json(orr.EXCHANGE_URL, {"code": "x"})


# ---- the browser flow, against the real localhost callback server -------------------------

def browser_that_authorizes(code, also_first=None):
    """A stand-in browser: 'the person clicks Authorize' = a GET on the callback URL."""
    seen = {}

    def open_browser(url):
        seen["url"] = url
        callback = params_of(url)["callback_url"]

        def visit():
            for c in ([also_first] if also_first else []) + [code]:
                with urllib.request.urlopen(f"{callback}?code={c}", timeout=10) as response:
                    seen["page"] = response.read().decode()

        threading.Thread(target=visit, daemon=True).start()

    return open_browser, seen


def test_browser_sign_in_returns_the_key(monkeypatch):
    calls = fake_openrouter(monkeypatch)
    open_browser, seen = browser_that_authorizes("good-code")
    said = []
    key = orr.sign_in(say=said.append, open_browser=open_browser, browser=True, wait_seconds=20)
    assert key == KEY
    sent = params_of(seen["url"])
    assert sent["callback_url"].startswith("http://localhost:") and sent["code_challenge_method"] == "S256"
    # the verifier that was exchanged is the one whose hash went to the browser
    verifier = calls[0]["payload"]["code_verifier"]
    assert base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode() \
        == sent["code_challenge"]
    assert "Signed in to OpenRouter" in seen["page"]
    assert any(seen["url"] in line for line in said)        # the link is printed too


def test_a_stray_request_to_the_callback_does_not_end_the_sign_in(monkeypatch):
    calls = fake_openrouter(monkeypatch)
    open_browser, _seen = browser_that_authorizes("good-code", also_first="forged")
    assert orr.sign_in(say=lambda s: None, open_browser=open_browser, browser=True, wait_seconds=20) == KEY
    assert [c["payload"]["code"] for c in calls] == ["forged", "good-code"]


def test_callback_server_answers_only_the_callback_path(monkeypatch):
    server = orr._Callback()
    try:
        for path in ("/", "/callback", "/other?code=x"):
            with pytest.raises(urllib.error.HTTPError):
                urllib.request.urlopen(f"http://127.0.0.1:{server.port}{path}", timeout=10)
        assert server.codes.empty()
    finally:
        server.close()


def test_no_answer_from_the_browser_without_a_terminal_is_an_error(monkeypatch):
    fake_openrouter(monkeypatch)
    with pytest.raises(orr.SignInError, match="in time"):
        orr.sign_in(say=lambda s: None, open_browser=lambda url: None, browser=True, wait_seconds=0.3)


def test_no_answer_from_the_browser_falls_back_to_pasting(monkeypatch):
    fake_openrouter(monkeypatch)
    said, asked = [], []

    def ask(prompt):
        asked.append(prompt)
        return "http://localhost:5123/callback?code=good-code"     # the page that would not load

    key = orr.sign_in(ask=ask, say=said.append, open_browser=lambda url: None, browser=True, wait_seconds=0.3)
    assert key == KEY and len(asked) == 1
    paste_links = [line.strip() for line in said if line.strip().startswith("https://") and "callback_url" not in line]
    assert paste_links, "the fallback must offer the link that shows a code"


# ---- the paste flow (SSH, containers) -----------------------------------------------------

def test_paste_flow_needs_no_browser(monkeypatch):
    calls = fake_openrouter(monkeypatch)
    said = []
    key = orr.sign_in(ask=lambda prompt: " good-code ", say=said.append, browser=False,
                      open_browser=lambda url: pytest.fail("no browser on this machine"))
    assert key == KEY
    link = next(line.strip() for line in said if line.strip().startswith("https://"))
    assert "callback_url" not in params_of(link) and params_of(link)["code_challenge"]
    assert calls[0]["payload"]["code"] == "good-code"


def test_paste_flow_cancel_and_wrong_code(monkeypatch):
    fake_openrouter(monkeypatch)
    with pytest.raises(orr.SignInError, match="cancelled"):
        orr.sign_in(ask=lambda prompt: "", say=lambda s: None, browser=False)
    with pytest.raises(orr.SignInError, match="did not accept"):
        orr.sign_in(ask=lambda prompt: "nope", say=lambda s: None, browser=False)

    def interrupted(prompt):
        raise KeyboardInterrupt

    with pytest.raises(orr.SignInError, match="cancelled"):
        orr.sign_in(ask=interrupted, say=lambda s: None, browser=False)


def test_over_ssh_there_is_no_browser(monkeypatch):
    monkeypatch.setenv("SSH_CONNECTION", "10.0.0.1 1 10.0.0.2 22")
    assert orr.can_open_browser() is False


@pytest.mark.parametrize("pasted, code", [
    ("abc123", "abc123"), ("  'abc123' ", "abc123"),
    ("http://localhost:5123/callback?code=abc123", "abc123"),
    ("localhost:5123/callback?code=abc123&x=1", "abc123"), ("", ""),
])
def test_code_from_what_was_pasted(pasted, code):
    assert orr.code_from(pasted) == code


# ---- staying signed in --------------------------------------------------------------------

def test_saved_key_is_private_and_can_be_forgotten():
    assert orr.saved_key() is None and orr.forget_key() is False
    assert orr.save_key(KEY)
    path = cloud.CACHE_PATH.parent / "openrouter.json"
    assert json.loads(path.read_text())["key"] == KEY
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert orr.saved_key() == KEY
    assert orr.forget_key() is True and orr.saved_key() is None and not path.exists()


def test_saved_key_can_be_switched_off_and_a_corrupt_file_is_no_key(monkeypatch):
    orr.save_key(KEY)
    monkeypatch.setenv("PULSE_OPENROUTER_SAVED_KEY", "off")
    assert orr.saved_key() is None
    monkeypatch.delenv("PULSE_OPENROUTER_SAVED_KEY")
    (cloud.CACHE_PATH.parent / "openrouter.json").write_text("not json")
    assert orr.saved_key() is None


def test_a_saved_key_is_scrubbed_from_logs():
    odd = "or-odd-format-key-0123456789"
    orr.save_key(odd)
    assert odd not in cloud.scrub_secrets(f"calling with {odd} now")


def test_key_for_only_answers_for_openrouter_models(monkeypatch):
    orr.save_key(KEY)
    assert orr.key_for("openrouter/deepseek/deepseek-v4-flash") == KEY
    assert orr.key_for("anthropic/claude-sonnet-5") is None and orr.key_for(None) is None
    monkeypatch.setenv(orr.ENV_KEY, "sk-or-v1-from-env")
    assert orr.key_for("openrouter/x/y") == "sk-or-v1-from-env"


def test_key_is_rejected_only_when_openrouter_says_so(monkeypatch):
    monkeypatch.setattr(orr, "_request_json", lambda *a, **k: (401, {"error": {"message": "no"}}))
    assert orr.key_is_rejected(KEY) is True

    def offline(*a, **k):
        raise orr.SignInError("could not reach OpenRouter (offline)")

    monkeypatch.setattr(orr, "_request_json", offline)
    assert orr.key_is_rejected(KEY) is False and orr.key_info(KEY) is None


# ---- `pulse openrouter` -------------------------------------------------------------------

def test_cli_login_saves_the_key_and_never_prints_it(monkeypatch, capsys):
    fake_openrouter(monkeypatch)
    monkeypatch.setattr(orr, "sign_in", lambda **k: KEY)
    assert entry.main(["openrouter"]) == 0
    out = capsys.readouterr().out
    assert orr.saved_key() == KEY and KEY not in out and "Signed in to OpenRouter" in out
    assert "free models" in out


def test_cli_status_and_logout(monkeypatch, capsys):
    fake_openrouter(monkeypatch)
    assert entry.main(["openrouter", "status"]) == 0
    assert "Not signed in" in capsys.readouterr().out
    orr.save_key(KEY)
    assert entry.main(["openrouter", "status"]) == 0
    out = capsys.readouterr().out
    assert "Signed in" in out and KEY not in out and KEY[-4:] in out
    assert entry.main(["openrouter", "logout"]) == 0
    assert orr.saved_key() is None and "settings/keys" in capsys.readouterr().out
    assert entry.main(["openrouter", "frobnicate"]) == 1


def test_cli_login_failure_is_reported(monkeypatch, capsys):
    def fail(**k):
        raise orr.SignInError("cancelled")

    monkeypatch.setattr(orr, "sign_in", fail)
    assert entry.main(["openrouter", "login"]) == 1
    assert "did not finish" in capsys.readouterr().out and orr.saved_key() is None


# ---- agent setup --------------------------------------------------------------------------

@pytest.fixture
def cli(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    c = PulseCLI(watch_locals={}, pdf_dir=str(tmp_path / "pdf"))
    c.non_interactive = False
    c.agent_provider = c.agent_key = None
    c._prime_with_agent_if_needed = lambda: None
    monkeypatch.setattr(pc, "_flush_stdin", lambda: None)
    monkeypatch.setattr(pc._ui, "enabled", lambda: False)
    return c


def scripted(monkeypatch, answers):
    """Answer Pulse's prompts in order; returns the prompts that were shown."""
    shown = []
    answers = list(answers)

    def prompt(plain, **k):
        shown.append(plain)
        assert answers, f"unexpected prompt: {plain}"
        return answers.pop(0)

    monkeypatch.setattr(pc, "_prompt_text", prompt)
    return shown


OR_MODEL = "OpenRouter (DeepSeek V4 Flash)"


def pick(monkeypatch, name=OR_MODEL):
    monkeypatch.setattr("builtins.input", lambda *a: name)


def test_setup_pasting_a_key_still_works_and_saves_nothing(cli, monkeypatch):
    pick(monkeypatch)
    shown = scripted(monkeypatch, ["sk-or-v1-pasted"])
    monkeypatch.setattr(orr, "sign_in", lambda **k: pytest.fail("a pasted key needs no sign-in"))
    assert cli._select_agent_provider_and_key(initial=True)
    assert cli.agent_provider == OR_MODEL and cli.agent_key == "sk-or-v1-pasted"
    assert os.environ[orr.ENV_KEY] == "sk-or-v1-pasted" and orr.saved_key() is None
    assert "sign in" in shown[0] and "sign up" in shown[0]


def test_setup_enter_signs_in_and_keeps_the_key(cli, monkeypatch, capsys):
    fake_openrouter(monkeypatch)
    pick(monkeypatch)
    scripted(monkeypatch, [""])
    monkeypatch.setattr(orr, "sign_in", lambda **k: KEY)
    assert cli._select_agent_provider_and_key(initial=True)
    assert cli.agent_key == KEY and os.environ[orr.ENV_KEY] == KEY and orr.saved_key() == KEY
    out = capsys.readouterr().out
    assert "Signed in to OpenRouter" in out and KEY not in out and "free models" in out


def test_setup_reuses_the_saved_sign_in(cli, monkeypatch):
    orr.save_key(KEY)
    monkeypatch.setattr(orr, "key_is_rejected", lambda key: False)
    pick(monkeypatch)
    shown = scripted(monkeypatch, [""])                     # Enter = yes, use it
    monkeypatch.setattr(orr, "sign_in", lambda **k: pytest.fail("already signed in"))
    assert cli._select_agent_provider_and_key(initial=True)
    assert cli.agent_key == KEY and "signed in to OpenRouter" in shown[0] and KEY not in shown[0]


def test_setup_a_deleted_saved_key_is_forgotten_and_asked_again(cli, monkeypatch):
    orr.save_key(KEY)
    monkeypatch.setattr(orr, "key_is_rejected", lambda key: True)
    pick(monkeypatch)
    scripted(monkeypatch, ["y", "sk-or-v1-new"])
    assert cli._select_agent_provider_and_key(initial=True)
    assert cli.agent_key == "sk-or-v1-new" and orr.saved_key() is None


def test_setup_an_environment_key_is_offered_first(cli, monkeypatch):
    monkeypatch.setenv(orr.ENV_KEY, "sk-or-v1-from-env")
    orr.save_key(KEY)
    pick(monkeypatch)
    shown = scripted(monkeypatch, ["y"])
    assert cli._select_agent_provider_and_key(initial=True)
    assert cli.agent_key == "sk-or-v1-from-env" and "already set" in shown[0]


def test_setup_a_sign_in_that_does_not_finish_leaves_the_agent_off(cli, monkeypatch, capsys):
    pick(monkeypatch)
    scripted(monkeypatch, [""])

    def fail(**k):
        raise orr.SignInError("cancelled")

    monkeypatch.setattr(orr, "sign_in", fail)
    assert cli._select_agent_provider_and_key(initial=True) is False
    assert cli.agent_provider is None and orr.ENV_KEY not in os.environ
    assert "did not finish" in capsys.readouterr().out


def test_other_providers_are_asked_for_a_key_as_before(cli, monkeypatch):
    other = next(n for n, info in pc.PROVIDERS.items()
                 if info.get("env_key") and info["env_key"] != orr.ENV_KEY and not info.get("local"))
    monkeypatch.delenv(pc.PROVIDERS[other]["env_key"], raising=False)
    pick(monkeypatch, other)
    shown = scripted(monkeypatch, ["some-key"])
    monkeypatch.setattr(orr, "sign_in", lambda **k: pytest.fail("not an OpenRouter model"))
    assert cli._select_agent_provider_and_key(initial=True)
    assert cli.agent_key == "some-key" and shown == [pc._KEY_PROMPT]
    assert "OpenRouter" in pc._KEY_PROMPT and "Enter" in pc._KEY_PROMPT


# ---- the sign-in is offered wherever a key is asked for ------------------------------------

def other_provider(monkeypatch):
    name = next(n for n, info in pc.PROVIDERS.items()
                if info.get("env_key") and info["env_key"] != orr.ENV_KEY and not info.get("local"))
    monkeypatch.delenv(pc.PROVIDERS[name]["env_key"], raising=False)
    return name


def openrouter_names():
    return [n for n, info in pc.PROVIDERS.items() if info.get("env_key") == orr.ENV_KEY]


def test_no_key_for_another_provider_offers_openrouter_instead(cli, monkeypatch):
    fake_openrouter(monkeypatch)
    pick(monkeypatch, other_provider(monkeypatch))
    # no key -> yes, sign in with OpenRouter -> first model -> Enter = sign in
    shown = scripted(monkeypatch, ["", "y", "", ""])
    monkeypatch.setattr(orr, "sign_in", lambda **k: KEY)
    assert cli._select_agent_provider_and_key(initial=True)
    assert cli.agent_provider == openrouter_names()[0] and cli.agent_key == KEY
    assert os.environ[orr.ENV_KEY] == KEY and orr.saved_key() == KEY
    assert "Sign in to OpenRouter" in shown[1]


def test_declining_the_offer_leaves_the_agent_off_as_before(cli, monkeypatch, capsys):
    pick(monkeypatch, other_provider(monkeypatch))
    scripted(monkeypatch, ["", ""])                       # no key, then Enter = no
    monkeypatch.setattr(orr, "sign_in", lambda **k: pytest.fail("declined"))
    assert cli._select_agent_provider_and_key(initial=True) is False
    assert cli.agent_provider is None and "No API key entered" in capsys.readouterr().out


def test_the_offer_can_pick_any_openrouter_model_and_take_a_pasted_key(cli, monkeypatch):
    pick(monkeypatch, other_provider(monkeypatch))
    names = openrouter_names()
    any_model = str(1 + next(i for i, n in enumerate(names) if pc.PROVIDERS[n].get("openrouter")))
    scripted(monkeypatch, ["", "y", any_model, "qwen/qwen-9", "sk-or-v1-pasted"])
    assert cli._select_agent_provider_and_key(initial=True)
    assert cli.agent_provider == "OpenRouter: qwen/qwen-9" and cli.agent_key == "sk-or-v1-pasted"
    assert orr.saved_key() is None


def test_an_openrouter_model_typed_as_a_custom_model_gets_the_sign_in(cli, monkeypatch):
    fake_openrouter(monkeypatch)
    custom = next(n for n, info in pc.PROVIDERS.items() if info.get("custom"))
    pick(monkeypatch, custom)
    shown = scripted(monkeypatch, ["openrouter/qwen/qwen-9", ""])
    monkeypatch.setattr(orr, "sign_in", lambda **k: KEY)
    assert cli._select_agent_provider_and_key(initial=True)
    assert cli.agent_provider == "OpenRouter: qwen/qwen-9" and cli.agent_key == KEY
    assert "sign in" in shown[1]


def test_the_agent_list_says_openrouter_needs_no_key(cli, monkeypatch, capsys):
    pick(monkeypatch, "")                                 # Enter = skip the agent
    assert cli._select_agent_provider_and_key(initial=True) is False
    listed = [line for line in capsys.readouterr().out.splitlines() if "OpenRouter" in line]
    assert listed and all("no key needed" in line for line in listed)


def test_pulse_code_asks_the_same_key_question():
    from pulse import pulse_code
    assert pulse_code._CodeAgentCLI._select_agent_provider_and_key is PulseCLI._select_agent_provider_and_key
    assert pulse_code._CodeAgentCLI._openrouter_key is PulseCLI._openrouter_key


def test_console_offers_the_sign_in_for_an_openrouter_model_without_a_key(monkeypatch):
    fake_openrouter(monkeypatch)
    monkeypatch.setattr(orr, "sign_in", lambda **k: KEY)
    said, asked = [], []

    def ask(prompt):
        asked.append(prompt)
        return ""                                         # Enter = yes

    assert orr.offer_sign_in_for("anthropic/claude-sonnet-5", ask=ask, say=said.append) is None
    assert asked == []
    assert orr.offer_sign_in_for("openrouter/qwen/qwen-9", ask=ask, say=said.append) == KEY
    assert orr.saved_key() == KEY and len(asked) == 1 and KEY not in "\n".join(said)
    assert orr.offer_sign_in_for("openrouter/qwen/qwen-9", ask=lambda p: pytest.fail("has a key")) == KEY


def test_console_offer_declined_or_without_a_terminal(monkeypatch):
    monkeypatch.setattr(orr, "sign_in", lambda **k: pytest.fail("not asked for"))
    assert orr.offer_sign_in_for("openrouter/qwen/qwen-9", ask=lambda p: "n", say=lambda s: None) is None
    said = []
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: False))
    assert orr.offer_sign_in_for("openrouter/qwen/qwen-9", say=said.append) is None
    assert "pulse openrouter" in said[0]


def test_unattended_run_uses_the_saved_sign_in_when_openrouter_is_asked_for(cli, monkeypatch):
    orr.save_key(KEY)
    cli.non_interactive = True
    monkeypatch.setenv("PULSE_PROVIDER", "openrouter/deepseek/deepseek-v4-flash")
    assert cli._select_agent_provider_and_key(initial=True)
    assert cli.agent_key == KEY and os.environ[orr.ENV_KEY] == KEY


def test_unattended_run_with_no_agent_asked_for_stays_without_one(cli, monkeypatch):
    """A saved sign-in alone must not switch the agent on in an unattended run."""
    orr.save_key(KEY)
    cli.non_interactive = True
    for info in pc.PROVIDERS.values():
        if info.get("env_key"):
            monkeypatch.delenv(info["env_key"], raising=False)
    assert cli._select_agent_provider_and_key(initial=True) is False
    assert cli.agent_provider is None


# ---- the console's brain and the dashboard ------------------------------------------------

def test_brain_agent_uses_the_saved_sign_in(monkeypatch):
    orr.save_key(KEY)
    sent = {}

    def completion(**kw):
        sent.update(kw)
        message = types.SimpleNamespace(content="ok")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=completion))
    assert pulse_brain.build_litellm_agent("openrouter/deepseek/deepseek-v4-flash")("hi") == "ok"
    assert sent["api_key"] == KEY
    sent.clear()
    pulse_brain.build_litellm_agent("anthropic/claude-sonnet-5")("hi")
    assert "api_key" not in sent


@pytest.fixture
def panel(monkeypatch):
    fake_tk = types.SimpleNamespace(Frame=type("Frame", (object,), {}), TclError=Exception)
    monkeypatch.setattr(core, "tk", fake_tk)
    monkeypatch.setattr(core, "HAS_TK", True)
    monkeypatch.setattr(core, "_CHAT_PANEL_CLS", None)
    cls = core._chat_panel_class()
    p = cls.__new__(cls)
    p.session_keys = {}
    p.appended = []
    p.after = lambda ms, fn=None: fn() if fn else None
    p._append = lambda who, text: p.appended.append(text)
    p._update_status_indicator = lambda: None
    yield p
    core._CHAT_PANEL_CLS = None


def dashboard_openrouter_provider():
    return next(n for n, info in core.PROVIDERS.items() if info.get("env_key") == orr.ENV_KEY)


def test_dashboard_picks_up_the_saved_sign_in(panel):
    name = dashboard_openrouter_provider()
    assert panel._has_active_key(name) is False
    orr.save_key(KEY)
    assert panel._has_active_key(name) is True and os.environ[orr.ENV_KEY] == KEY


def test_dashboard_sign_in_keeps_the_key_and_says_so(panel, monkeypatch):
    name = dashboard_openrouter_provider()

    def sign_in(ask=None, say=print, browser=None, **k):
        assert ask is None and browser is True
        say("  https://openrouter.ai/auth?code_challenge=CH")
        return KEY

    monkeypatch.setattr(orr, "sign_in", sign_in)
    panel._openrouter_sign_in(name).join(10)
    assert panel.session_keys[name] == KEY and os.environ[orr.ENV_KEY] == KEY and orr.saved_key() == KEY
    text = "\n".join(panel.appended)
    assert "https://openrouter.ai/auth" in text and "Signed in to OpenRouter" in text and KEY not in text


def test_dashboard_sign_in_failure_is_shown(panel, monkeypatch):
    def fail(**k):
        raise orr.SignInError("nothing came back from the browser in time")

    monkeypatch.setattr(orr, "sign_in", fail)
    panel._openrouter_sign_in(dashboard_openrouter_provider()).join(10)
    assert panel.session_keys == {} and "did not finish" in "\n".join(panel.appended)
    assert panel._openrouter_signing_in is False


def test_dashboard_no_key_for_another_provider_offers_openrouter(panel, monkeypatch):
    monkeypatch.setattr(core, "messagebox", types.SimpleNamespace(askyesno=lambda *a, **k: True), raising=False)
    chosen = []
    panel.provider_var = types.SimpleNamespace(set=chosen.append)
    monkeypatch.setattr(orr, "sign_in", lambda **k: KEY)
    panel._offer_openrouter_instead().join(10)
    assert core.PROVIDERS[chosen[0]]["env_key"] == orr.ENV_KEY
    assert panel.session_keys[chosen[0]] == KEY and orr.saved_key() == KEY


def test_dashboard_offer_declined_changes_nothing(panel, monkeypatch):
    monkeypatch.setattr(core, "messagebox", types.SimpleNamespace(askyesno=lambda *a, **k: False), raising=False)
    panel.provider_var = types.SimpleNamespace(set=lambda name: pytest.fail("declined"))
    assert panel._offer_openrouter_instead() is None and panel.session_keys == {}
