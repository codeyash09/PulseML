"""Sweep 2: pulse_supabase.py (+ the cloud-history code in pulse_cli.py).

Every network call is mocked (urllib.request.urlopen / pulse_supabase functions);
nothing here talks to Supabase.

test_bug_* assert the CORRECT behaviour and fail on the current code.
test_ok_* are regression coverage for behaviour verified to work.
"""
import io
import json
import os
import subprocess
import sys
import textwrap
import threading
import time
import urllib.error

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(os.path.dirname(HERE)), "src")
if os.path.isdir(SRC):
    sys.path.insert(0, SRC)

from pulse import pulse_supabase as cloud  # noqa: E402

UID_A = "11111111-1111-1111-1111-111111111111"


class _Resp:
    def __init__(self, payload):
        self._raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(cloud, "CACHE_PATH", tmp_path / "credentials.json")
    monkeypatch.setattr(cloud, "PROFILE_PATH", tmp_path / "profile.json")
    monkeypatch.setattr(cloud, "_EXTRA_SECRETS", set())

    def _no_network(*a, **k):
        raise AssertionError("unexpected real network call")

    monkeypatch.setattr(cloud.urllib.request, "urlopen", _no_network)
    cloud.clear_session()
    yield
    cloud.clear_session()


def _cli_obj():
    """A PulseCLI with only the cloud-sync state these tests touch."""
    from pulse import pulse_cli as cli
    obj = cli.PulseCLI.__new__(cli.PulseCLI)
    obj.debug_session_id = "sid"
    obj._agent_logs, obj._error_tracebacks, obj._telemetry, obj._incidents = [], [], [], []
    obj._uptime_seconds, obj._downtime_seconds = 5.0, 0.0
    obj._last_synced_commit_sha = "abc"
    obj._cloud_dirty_fields = set()
    obj._last_cloud_flush = time.monotonic()
    obj.cloud_flush_interval = 600.0
    obj._cloud_history_unloaded, obj._cloud_history_retry_at = False, 0.0
    return obj


class _RecordingSync:
    def __init__(self):
        self.bodies = []

    def submit(self, session_id, fields, wait=False):
        self.bodies.append(fields)
        ev = threading.Event()
        ev.set()
        return ev if wait else None


# ============================================================================ bugs

def test_bug_expired_access_token_is_never_refreshed(monkeypatch):
    """Supabase access tokens (JWTs) expire after an hour by default. Pulse gets one at
    sign-in/refresh and never refreshes it again: every Debug_Sessions PATCH after the
    first hour of a run fails with HTTP 401 'JWT expired' (reported once by
    BackgroundSync, then silently dropped) -- cloud logging stops for the rest of any
    run longer than an hour. Correct: on a 401/JWT-expired response, use the stored
    refresh token to get a new access token and retry the request once."""
    cloud.set_session("expired-access-token", "refresh-1")
    seen = []

    def urlopen(req, timeout=None):
        url = req.full_url
        auth = dict(req.header_items()).get("Authorization")
        seen.append((req.get_method(), url, auth))
        if "/auth/v1/token" in url:
            return _Resp({"access_token": "fresh-access-token", "refresh_token": "refresh-2",
                          "user": {"email": "a@example.com"}})
        if auth == "Bearer expired-access-token":
            raise urllib.error.HTTPError(url, 401, "Unauthorized", {},
                                         io.BytesIO(b'{"code":"PGRST301","message":"JWT expired"}'))
        return _Resp(b"")

    monkeypatch.setattr(cloud.urllib.request, "urlopen", urlopen)
    cloud.patch_debug_session("sid", {"uptime_seconds": 3700})   # must not raise
    assert seen[-1][2] == "Bearer fresh-access-token", seen
    assert cloud.get_refresh_token() == "refresh-2"


def test_bug_environment_info_imports_every_installed_framework(tmp_path):
    """collect_environment_info() (run at session start whenever telemetry is on -- the
    default) does `import torch; torch.cuda.is_available()`, `import tensorflow`,
    `import jax; jax.devices()`, `import cupy`, `import mlx` and __import__s all of them
    again for version strings -- whether or not the training script uses them. That is
    the exact cost the lazy pulse_backend rewrite removed from `import pulse`: seconds of
    imports, plus CUDA/XLA initialisation *before the script's own code runs* (auto_track
    and `pulse run` put setup at the top): torch.cuda init freezes CUDA_VISIBLE_DEVICES
    before the script sets it, and jax.devices() initialises the XLA GPU client, which
    by default preallocates most of the GPU's memory in a PyTorch job. Correct: only
    report frameworks already in sys.modules (nvidia-smi still gives the GPU)."""
    fake = tmp_path / "fake"
    for name, body in {
        "torch": "class _C:\n    @staticmethod\n    def is_available():\n        open(MARK + '.cuda', 'w').close(); return False\ncuda = _C()\n__version__ = '9'\n",
        "tensorflow": "__version__ = '9'\n",
        "jax": "def devices():\n    open(MARK + '.devices', 'w').close(); return []\n__version__ = '9'\n",
    }.items():
        pkg = fake / name
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text(
            f"import os\nMARK = os.path.join({str(tmp_path)!r}, {name!r})\nopen(MARK, 'w').close()\n" + body)
    code = textwrap.dedent("""
        import subprocess, sys
        from pulse import pulse_supabase as cloud
        class R:
            returncode = 1; stdout = ""; stderr = ""
        cloud.subprocess.run = lambda *a, **k: R()
        info = cloud.collect_environment_info()
        print("versions", info["framework_versions"])
    """)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(fake), SRC]), CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                       env=env, cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr
    imported = sorted(p.name for p in tmp_path.iterdir() if p.is_file())
    assert imported == [], f"environment info imported/initialised: {imported}"


def test_bug_cloud_history_retry_runs_on_the_training_thread(monkeypatch):
    """After an auto-fix restart whose history preload failed, _build_cloud_patch_body
    retries the preload every 60 s -- synchronously, inside _maybe_flush_cloud, which
    update() calls on the training thread (uptime flush). Offline, each retry blocks the
    user's training loop for the full request timeout (8 s here, longer on a DNS stall).
    Correct: the retry happens on the background sync thread; the flush returns at once."""
    obj = _cli_obj()
    obj._cloud_sync = _RecordingSync()
    obj._cloud_history_unloaded, obj._cloud_history_retry_at = True, 0.0

    def slow_offline(session_id):
        time.sleep(2.0)
        raise cloud.SupabaseError("GET Debug_Sessions -> network error: timed out")

    monkeypatch.setattr(cloud, "fetch_debug_session", slow_offline)
    obj._cloud_dirty_fields = {"uptime_seconds", "agent_logs"}
    t0 = time.monotonic()
    obj._maybe_flush_cloud(force=True)
    elapsed = time.monotonic() - t0
    assert elapsed < 0.5, f"flush blocked the caller for {elapsed:.1f}s"
    # Let the background retry finish while fetch_debug_session is still mocked, so it
    # can never reach the real network after this test's patches are undone.
    for t in threading.enumerate():
        if t.name == "pulse-cloud-history":
            t.join(10)


def test_bug_final_flush_drops_entries_held_while_history_unloaded(monkeypatch):
    """While a resumed session's history is unloaded, array fields are held back (so they
    don't overwrite the saved arrays) and retried at most every 60 s. The final flushes --
    atexit and right before a fix-triggered restart -- don't force that retry: if the
    process ends (or restarts; the child preloads from the server, which never got them)
    within 60 s of a failed attempt, the crash traceback and agent turns logged since are
    silently lost even though the server is reachable again. Correct: the final flush
    retries the load, then sends saved + new."""
    obj = _cli_obj()
    obj._cloud_sync = _RecordingSync()
    obj._cloud_history_unloaded = True
    obj._cloud_history_retry_at = time.monotonic() + 45.0          # failed 15 s ago
    obj._error_tracebacks = ["Traceback: the crash that ended this run"]
    obj._cloud_dirty_fields = {"error_tracebacks"}
    monkeypatch.setattr(cloud, "fetch_debug_session",
                        lambda sid: {"error_tracebacks": [cloud.encode_entry("old crash")]})
    obj._flush_cloud_now()
    sent = [b for b in obj._cloud_sync.bodies if "error_tracebacks" in b]
    assert sent, f"final flush sent {obj._cloud_sync.bodies}"
    assert cloud.decode_entries(sent[-1]["error_tracebacks"]) == [
        "old crash", "Traceback: the crash that ended this run"]


@pytest.mark.parametrize("secret", [
    "hf_AbCdEfGhIjKlMnOpQrStUvWxYz01234567",                                   # Hugging Face
    "sk-svcacct-AbCdEf1234567890_ghIJklMNopQRstuvWXyz0123456789abcdEFGH",      # OpenAI service account
    "sk-admin-AbCdEf1234567890ghIJklMNopQRstuvWXyz01234",                       # OpenAI admin key
    "github_pat_11ABCDEFG0123456789_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUV",  # GitHub fine-grained
    "gho_16C7e42F292c6912E7710c838347Ae178B4a",                                # GitHub OAuth
    "AKIAIOSFODNN7EXAMPLE",                                                   # AWS access key id
    "xai-AbCdEf1234567890ghIJklMNopQRstuvWXyz01234abcdefghijklmnop",           # xAI
])
def test_bug_scrubber_misses_common_vendor_keys(secret):
    """scrub_secrets knows sk-ant/sk-or/sk-proj/AIza/gsk_/ghp_/xox; many keys ML code
    hard-codes are not covered -- above all Hugging Face tokens (`login("hf_...")`,
    `use_auth_token="hf_..."`), which ride along in the user's code/tracebacks into the
    agent log, .pulse_history and cloud agent_logs. Env-var scrubbing doesn't help for a
    key written in the script."""
    text = f'login(token="{secret}")' if not secret.startswith("AKIA") else f"aws key {secret}"
    assert secret not in cloud.scrub_secrets(text)


@pytest.mark.parametrize("line,secret", [
    ('password = "P@ssw0rd!2024"', "P@ssw0rd!2024"),
    ('token = "abcd1234efgh5678ijkl"', "abcd1234efgh5678ijkl"),
    ('SECRET_KEY = "django-insecure-abc123def456ghi789"', "django-insecure-abc123def456ghi789"),
    ('client_secret: "abc123def456ghi789jkl"', "abc123def456ghi789jkl"),
    ("aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"),
    ("postgres://user:hunter2secret@db.example.com/prod", "hunter2secret"),
])
def test_bug_scrubber_misses_secret_assignments(line, secret):
    """The assignment pattern only knows api_key/api_secret/access_token/password and a
    value of [A-Za-z0-9-_./+=]: a password containing @ ! # etc. (most real ones), a
    plain `token`/`secret`/`*_secret`/`secret_access_key` assignment, and a password in a
    connection URL all pass through unredacted."""
    assert secret not in cloud.scrub_secrets(line)


@pytest.mark.parametrize("text", [
    "saved report to desk-analysisreport2024final.csv",
    "the whisk-0123456789abcdefghijklmn config",
])
def test_bug_scrubber_redacts_ordinary_words_ending_in_sk(text):
    """The OpenAI pattern `sk-(?!ant-)[A-Za-z0-9]{20,}` has no word boundary, so any word
    ending in 'sk' followed by '-' and 20 alphanumerics (desk-, disk-, task-, mask-...)
    is mangled into 'de[REDACTED_SECRET]' in logs and cloud history."""
    assert "REDACTED" not in cloud.scrub_secrets(text), cloud.scrub_secrets(text)


def test_bug_scrubber_redacts_non_secret_env_values_named_token(monkeypatch):
    """Every env var whose NAME contains token/secret/key/password has its VALUE redacted
    everywhere -- including ones that are not secrets at all, e.g.
    TOKENIZER_NAME=bert-base-uncased, so every traceback and agent answer mentioning the
    model name reads '[REDACTED_SECRET]'. Correct: match the name as a secret-ish suffix
    (API_KEY, _TOKEN, _SECRET, PASSWORD), not any substring."""
    monkeypatch.setenv("TOKENIZER_NAME", "bert-base-uncased")
    out = cloud.scrub_secrets("OSError: bert-base-uncased is not a local folder")
    assert "bert-base-uncased" in out, out


def test_bug_preloading_cloud_history_loses_a_concurrent_append(monkeypatch):
    """(Found by reading.) _preload_cloud_history rebuilt each array as saved +
    list(current) AFTER the (up to 8 s) fetch: an entry another thread appended while
    the fetch was in flight went into the old list object and was dropped when the
    attribute was replaced. Correct: merge in place, under a lock."""
    obj = _cli_obj()
    obj._agent_logs = [{"q": "before"}]

    def fetch(sid):
        obj._agent_logs.append({"q": "during"})      # another thread's append mid-fetch
        return {"agent_logs": [cloud.encode_entry({"q": "saved"})]}

    monkeypatch.setattr(cloud, "fetch_debug_session", fetch)
    assert obj._preload_cloud_history() is True
    assert obj._agent_logs == [{"q": "saved"}, {"q": "before"}, {"q": "during"}]
    # A retry after it has loaded (a final flush racing the background retry) is a no-op.
    monkeypatch.setattr(cloud, "fetch_debug_session",
                        lambda sid: {"agent_logs": [cloud.encode_entry({"q": "saved"})]})
    assert obj._preload_cloud_history(only_if_unloaded=True) is True
    assert obj._agent_logs == [{"q": "saved"}, {"q": "before"}, {"q": "during"}]


def test_ok_expired_token_is_refreshed_once_not_in_a_loop(monkeypatch):
    """A refresh that still gets 401 is reported, not retried forever."""
    cloud.set_session("expired-access-token", "refresh-1")
    calls = []

    def urlopen(req, timeout=None):
        calls.append(req.full_url)
        if "/auth/v1/token" in req.full_url:
            return _Resp({"access_token": "still-bad", "refresh_token": "refresh-2"})
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, io.BytesIO(b"JWT expired"))

    monkeypatch.setattr(cloud.urllib.request, "urlopen", urlopen)
    with pytest.raises(cloud.SupabaseError):
        cloud.patch_debug_session("sid", {"uptime_seconds": 1})
    assert len(calls) == 3, calls        # request, refresh, one retry


def test_ok_scrubber_keeps_ordinary_ml_text(monkeypatch):
    for text in ('tokenizer.eos_token = "<|endoftext|>"', "max_tokens=40960000",
                 "is_secret: False", "http://localhost:8080/x@y",
                 "model = AutoModel.from_pretrained('bert-base-uncased')"):
        assert cloud.scrub_secrets(text) == text, text
    monkeypatch.setenv("PWD", "/home/someone/project")
    assert cloud.scrub_secrets("cd /home/someone/project") == "cd /home/someone/project"


# ============================================================================ ok

def test_ok_scrubber_catches_known_formats_and_keeps_hashes():
    for secret in ("sk-ant-api03-AbCdEf1234567890ghIJklMNopQRstuvWXyz01234-abcdAA",
                   "sk-or-v1-" + "a1" * 32, "sk-proj-AbCdEf1234567890ghIJklMNop",
                   "AIzaSyA1234567890abcdefghijklmnopqrstu", "gsk_AbCdEf1234567890ghIJklMNop",
                   "ghp_AbCdEf1234567890ghIJklMNop"):
        assert secret not in cloud.scrub_secrets(f"key={secret} and {secret}")
    assert cloud.scrub_secrets('{"api_key": "abcdef123456"}').count("abcdef123456") == 0
    for normal in ("commit 3f2a9c1e5b7d8f0a1c2e3b4d5f6a7b8c9d0e1f2a",
                   "run 550e8400-e29b-41d4-a716-446655440000 finished", "loss=0.12345678"):
        assert cloud.scrub_secrets(normal) == normal


def test_ok_register_secret():
    cloud.register_secret("  custom-provider-key-1234  ")
    cloud.register_secret("short")
    assert "custom-provider-key-1234" not in cloud.scrub_secrets("x custom-provider-key-1234 y")
    assert cloud.scrub_secrets("short") == "short"


@pytest.mark.parametrize("url,expected", [
    ("https://user:ghp_tok@github.com/o/r.git", "https://github.com/o/r.git"),
    ("https://oauth2:glpat-xyz@gitlab.com:8443/o/r.git", "https://gitlab.com:8443/o/r.git"),
    ("ssh://git:pw@[::1]:2222/o/r.git", "ssh://[::1]:2222/o/r.git"),
    ("https://user:p@ss@host/r.git", "https://host/r.git"),
    ("git@github.com:o/r.git", "git@github.com:o/r.git"),
    ("https://github.com/o/r.git", "https://github.com/o/r.git"),
])
def test_ok_strip_url_credentials(url, expected):
    assert cloud.strip_url_credentials(url) == expected


def test_ok_background_sync_reports_once_per_kind_and_survives(monkeypatch):
    errs = iter([OSError("reset"), cloud.SupabaseError("PATCH -> HTTP 500: x"),
                 ConnectionResetError("again"), cloud.SupabaseError("PATCH -> HTTP 500: y")])

    def fail(session_id, fields, timeout=5):
        raise next(errs)

    monkeypatch.setattr(cloud, "patch_debug_session", fail)
    errors = []
    sync = cloud.BackgroundSync(on_error=errors.append)
    for _ in range(4):
        assert sync.submit("sid", {"x": 1}, wait=True).wait(2)
    assert len(errors) == 2


def test_ok_fetch_recent_sessions_falls_back_only_on_unknown_column(monkeypatch):
    selects = []

    def req(method, path, params=None, **k):
        selects.append(params["select"])
        if "incidents" in params["select"]:
            raise cloud.SupabaseError('GET -> HTTP 400: {"code":"42703","message":"column Debug_Sessions.incidents does not exist"}')
        return [{"id": 1}]

    monkeypatch.setattr(cloud, "_request", req)
    assert cloud.fetch_recent_sessions("t", None) == [{"id": 1}]
    assert selects == [cloud._SESSION_SELECT, cloud._SESSION_SELECT_LEGACY]

    def offline(*a, **k):
        raise cloud.SupabaseError("GET -> network error: unreachable")

    monkeypatch.setattr(cloud, "_request", offline)
    with pytest.raises(cloud.SupabaseError):
        cloud.fetch_recent_sessions("t", None)


def test_ok_credentials_expiry_and_email_lowercase(monkeypatch):
    cloud.save_cached_credentials(UID_A, "a@example.com", session_token="tok")
    assert cloud.load_cached_credentials()["session_token"] == "tok"
    data = json.loads(cloud.CACHE_PATH.read_text())
    data["cached_at"] = time.time() - 31 * 86400
    cloud.CACHE_PATH.write_text(json.dumps(data))
    assert cloud.load_cached_credentials() is None
    assert cloud._normalize_email("  Bob@Example.COM ") == "bob@example.com"
    assert oct(os.stat(cloud.CACHE_PATH).st_mode & 0o777) == "0o600"


def test_ok_verify_session_token_fails_closed_without_local_token(monkeypatch):
    monkeypatch.setattr(cloud, "_request", lambda *a, **k: [{"session_token_hash": "abc"}])
    assert cloud.verify_session_token(UID_A, None) is False
    monkeypatch.setattr(cloud, "_request", lambda *a, **k: [{"session_token_hash": None}])
    assert cloud.verify_session_token(UID_A, None) is None
