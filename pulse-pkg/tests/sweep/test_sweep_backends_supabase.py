"""Sweep: pulse_supabase.py -- auth, credential cache, teams, session sync, scrubbing.

Every network call is mocked (urllib.request.urlopen / pulse_supabase._request);
nothing here talks to Supabase.
"""
import http.client
import io
import json
import os
import stat
import threading
import time
import urllib.error

import pytest

from pulse import pulse_supabase as cloud

UID_A = "11111111-1111-1111-1111-111111111111"
UID_B = "22222222-2222-2222-2222-222222222222"


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
    monkeypatch.setattr(cloud.time, "sleep", lambda s: None)

    def _no_network(*a, **k):
        raise AssertionError("unexpected real network call")

    monkeypatch.setattr(cloud.urllib.request, "urlopen", _no_network)
    cloud.clear_session()
    yield
    cloud.clear_session()


# ============================================================================ bugs

def test_bug_request_leaks_connection_reset_instead_of_supabaseerror(monkeypatch):
    """_request promises 'Raises SupabaseError on any failure (network ...)', but a
    connection dropped after the request was sent (http.client.RemoteDisconnected /
    ConnectionResetError, raised from getresponse() and NOT wrapped in URLError by
    urllib) escapes raw. Every caller only catches SupabaseError, so a flaky proxy
    crashes /team commands, the auth flow, etc. Correct: also catch OSError /
    http.client.HTTPException / ValueError and re-raise as SupabaseError."""
    def boom(*a, **k):
        raise http.client.RemoteDisconnected("Remote end closed connection without response")

    monkeypatch.setattr(cloud.urllib.request, "urlopen", boom)
    with pytest.raises(cloud.SupabaseError):
        cloud._request("GET", "profiles")


def test_bug_background_sync_thread_dies_on_non_supabase_error(monkeypatch):
    """BackgroundSync._run only catches SupabaseError. One ConnectionResetError (see
    above) kills the daemon thread silently: every later flush is queued forever and
    never sent, and on_error is never called, so the user is never told cloud sync
    stopped. Correct: catch Exception in _run (report once, keep looping)."""
    calls = []

    def fake_patch(session_id, fields, timeout=5):
        calls.append(fields)
        if len(calls) == 1:
            raise ConnectionResetError("reset by peer")

    monkeypatch.setattr(cloud, "patch_debug_session", fake_patch)
    errors = []
    sync = cloud.BackgroundSync(on_error=errors.append)
    sync.submit("sid", {"a": 1}, wait=True).wait(2)
    done = sync.submit("sid", {"b": 2}, wait=True)
    assert done.wait(2), "second flush never processed -- sync thread is dead"
    assert calls == [{"a": 1}, {"b": 2}]
    assert errors, "the failure was never reported"


def test_bug_refresh_session_raises_on_read_timeout(monkeypatch):
    """refresh_session docstring: 'Returns False (never raises)'. It is the first call
    of _auth_flow, outside any try. A socket read timeout surfaces from urlopen as a raw
    TimeoutError (not URLError), which refresh_session doesn't catch -- _cloud_setup only
    catches SupabaseError, so a slow network crashes Pulse startup instead of falling
    back to local mode. Correct: catch Exception (or at least OSError/ValueError)."""
    def slow(*a, **k):
        raise TimeoutError("The read operation timed out")

    monkeypatch.setattr(cloud.urllib.request, "urlopen", slow)
    assert cloud.refresh_session("some-refresh-token") is False


def test_bug_refresh_session_raises_on_non_json_body(monkeypatch):
    """Same contract: a captive portal / proxy answering 200 with HTML makes
    json.loads raise JSONDecodeError out of refresh_session."""
    monkeypatch.setattr(cloud.urllib.request, "urlopen", lambda *a, **k: _Resp(b"<html>login</html>"))
    assert cloud.refresh_session("some-refresh-token") is False


def test_bug_log_in_raises_raw_timeout_not_supabaseerror(monkeypatch):
    """log_in only catches HTTPError/URLError; a read timeout escapes as TimeoutError,
    which _cloud_setup/_auth_flow (except cloud.SupabaseError) don't handle -> crash.
    Same pattern in sign_up, change_password and delete_account."""
    def slow(*a, **k):
        raise TimeoutError("timed out")

    monkeypatch.setattr(cloud.urllib.request, "urlopen", slow)
    with pytest.raises(cloud.SupabaseError):
        cloud.log_in("user@example.com", "correct horse battery")


def test_bug_verify_session_token_skipped_when_cached_token_missing(monkeypatch):
    """The forged-credentials check: verify_session_token returns None ('deployment
    doesn't support tokens -> trust the TTL') whenever the LOCAL token is missing -- even
    when the server has a session_token_hash stored for that user. So a hand-written
    credentials.json containing only {user_id, email} (user ids are visible to every
    teammate in Teams.members) skips verification entirely and _auth_flow trusts it.
    Correct: when the token is missing, still query; if a hash is stored, return False."""
    monkeypatch.setattr(cloud, "_request", lambda *a, **k: [{"session_token_hash": "ab" * 32}])
    assert cloud.verify_session_token(UID_A, None) is False


def test_bug_cached_credentials_without_cached_at_never_expire():
    """load_cached_credentials only applies SESSION_TTL_DAYS when 'cached_at' is present.
    A credentials.json without it (older Pulse, or forged) is trusted forever. Same for a
    cached_at in the future (age is negative). Correct: treat missing/invalid/future
    cached_at as expired."""
    cloud.CACHE_PATH.write_text(json.dumps({"user_id": UID_A, "email": "a@example.com"}))
    assert cloud.load_cached_credentials() is None


def test_bug_cached_credentials_with_future_timestamp_never_expire():
    """See above: cached_at far in the future makes age negative -> never expires."""
    cloud.CACHE_PATH.write_text(json.dumps({
        "user_id": UID_A, "email": "a@example.com", "cached_at": time.time() + 10 * 365 * 86400,
    }))
    assert cloud.load_cached_credentials() is None


def test_bug_credentials_file_briefly_world_readable(monkeypatch):
    """save_cached_credentials writes the refresh token (a real bearer credential) with
    Path.write_text -- created with the process umask (typically 0644) -- and only
    chmods to 0600 afterwards, so there's a window in which other local users can read
    it (and it stays 0644 if chmod fails, which is silently ignored). Correct: create it
    with os.open(path, O_WRONLY|O_CREAT|O_TRUNC, 0o600) (or write a temp file created
    0600 and os.replace it)."""
    old = os.umask(0o022)
    seen = {}
    real_chmod = os.chmod

    def spy_chmod(path, mode, *a, **k):
        seen["mode_before_chmod"] = stat.S_IMODE(os.stat(path).st_mode)
        return real_chmod(path, mode, *a, **k)

    monkeypatch.setattr(cloud.os, "chmod", spy_chmod)
    try:
        cloud.save_cached_credentials(UID_A, "a@example.com", refresh_token="rt-secret-value")
    finally:
        os.umask(old)
    assert "rt-secret-value" in cloud.CACHE_PATH.read_text()
    assert seen.get("mode_before_chmod", 0o600) & 0o077 == 0, oct(seen["mode_before_chmod"])


def test_bug_save_credentials_carries_previous_users_session_token():
    """save_cached_credentials 'preserves' session_token/refresh_token from the file on
    disk when not passed -- without checking the file belongs to the same user. After
    A's cache exists, saving B's login (e.g. attach_session_token returned None) writes
    A's session_token and refresh_token into B's credentials. Correct: only merge
    when existing['user_id'] == user_id."""
    cloud.save_cached_credentials(UID_A, "a@example.com", session_token="tok-A", refresh_token="rt-A")
    cloud.clear_session()
    cloud.save_cached_credentials(UID_B, "b@example.com")
    data = json.loads(cloud.CACHE_PATH.read_text())
    assert data["user_id"] == UID_B
    assert data.get("session_token") != "tok-A"
    assert data.get("refresh_token") != "rt-A"


def test_bug_git_remote_credentials_uploaded_to_team_repo(monkeypatch):
    """create_team stores `git remote get-url origin` verbatim in Teams.repo. HTTPS
    remotes very often embed a token (https://x-access-token:ghp_...@github.com/...,
    or user:password@), which then gets uploaded to Supabase and shown to every
    teammate. encode_entry scrubs agent logs, but this path isn't scrubbed. Correct:
    strip userinfo from the URL (urllib.parse -> netloc without user:pass) before use."""
    token = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1L2"

    class _Out:
        returncode = 0
        stdout = f"https://x-access-token:{token}@github.com/org/repo.git\n"

    monkeypatch.setattr(cloud.subprocess, "run", lambda *a, **k: _Out())
    sent = {}

    def fake_request(method, path, params=None, body=None, prefer=None, timeout=8):
        sent["body"] = body
        return [dict(body, team_id="t1")]

    monkeypatch.setattr(cloud, "_request", fake_request)
    cloud.create_team(UID_A)
    assert token not in json.dumps(sent["body"])
    assert "github.com/org/repo" in sent["body"]["repo"]


def test_bug_scrub_misses_openrouter_key():
    """OpenRouter is Pulse's default provider, and its keys look like
    'sk-or-v1-<64 hex>'. The OpenAI pattern sk-(?!ant-)[A-Za-z0-9]{20,} stops at the
    '-' after 'or', and no other pattern matches, so the key is only redacted if it
    happens to also sit in an *_API_KEY env var of this very process (it doesn't for a
    custom env-var name, a config-file key, or another process reading the log).
    Correct: add r'sk-or-v1-[A-Za-z0-9]{20,}' (or make the sk- pattern allow '-')."""
    key = "sk-or-v1-" + "0123456789abcdef" * 4
    out = cloud.scrub_secrets(f"litellm.AuthenticationError: bad key {key}")
    assert key not in out


def test_bug_scrub_misses_json_style_secret_assignment():
    """The generic assignment pattern requires the key name to be directly followed by
    ':' or '=', so JSON/dict reprs -- '"api_key": "...."' or "{'password': '...'}" --
    are not matched (the closing quote sits between name and colon). Config dumps and
    tool-call arguments in agent transcripts are exactly this shape. Correct: allow an
    optional quote: (api[_-]?key|...)['\"]?\\s*[:=]."""
    out = cloud.scrub_secrets('{"api_key": "abcd1234efgh5678ijkl", "password": "hunter2hunter2"}')
    assert "abcd1234efgh5678ijkl" not in out
    assert "hunter2hunter2" not in out


def test_bug_encode_entry_raises_on_non_string_keys():
    """encode_entry docstring: 'never raises'. A dict with a tuple key (e.g. a
    confusion-matrix or per-(layer,head) dict in telemetry) makes json.dumps raise
    TypeError, and the fallback json.dumps(obj, default=str) raises the same TypeError
    (default= doesn't apply to keys). Correct: fallback to repr()/str() or stringify keys."""
    out = cloud.encode_entry({(0, 1): 3, "t": 1.0})
    assert isinstance(out, str)


def test_bug_delete_account_reports_success_when_profile_delete_fails(monkeypatch):
    """delete_account swallows a failed DELETE of the profiles row ('pass'), so the CLI
    prints 'Account deleted' (and clears local creds) while the account and profile
    still exist server-side -- a GDPR erasure request silently not honoured. Correct:
    re-raise SupabaseError so the user is told the deletion failed."""
    monkeypatch.setattr(cloud, "fetch_user", lambda uid: {"id": uid, "email": "a@example.com"})
    monkeypatch.setattr(cloud.urllib.request, "urlopen", lambda *a, **k: _Resp({"access_token": "x"}))
    monkeypatch.setattr(cloud, "find_teams_for_user", lambda uid: [])

    def fake_request(method, path, **k):
        if method == "DELETE":
            raise cloud.SupabaseError("DELETE profiles -> HTTP 403: permission denied")
        return None

    monkeypatch.setattr(cloud, "_request", fake_request)
    with pytest.raises(cloud.SupabaseError):
        cloud.delete_account(UID_A, "password123")


def test_bug_change_password_revokes_sessions_although_nothing_changed(monkeypatch):
    """change_password can never succeed (it unconditionally raises 'requires an
    authenticated session'), but its `finally:` still calls revoke_session_token --
    so /password leaves the password unchanged AND wipes the session-token protection
    for every device (verify then returns None -> TTL-only trust). Correct: only revoke
    after the password was actually changed."""
    monkeypatch.setattr(cloud, "fetch_user", lambda uid: {"id": uid, "email": "a@example.com"})
    monkeypatch.setattr(cloud.urllib.request, "urlopen", lambda *a, **k: _Resp({"access_token": "x"}))
    revoked = []
    monkeypatch.setattr(cloud, "revoke_session_token", lambda uid: revoked.append(uid))
    with pytest.raises(cloud.SupabaseError):
        cloud.change_password(UID_A, "oldpassword", "newpassword1")
    assert revoked == []


def test_bug_sign_up_with_capitalised_email_reports_missing_profile(monkeypatch):
    """Supabase Auth lowercases emails (the signup trigger copies the lowercased
    auth.users.email into profiles). sign_up then looks the profile up with the email
    exactly as typed ('eq.Alice@Example.com', case-sensitive) and raises 'Profile was
    not created after signup' although the account was created. log_in avoids this by
    using the email returned by Auth. Correct: normalise email.strip().lower() at the
    top of sign_up/log_in/find_user_by_email."""
    monkeypatch.setattr(cloud.urllib.request, "urlopen", lambda *a, **k: _Resp(
        {"user": {"email": "alice@example.com"}, "access_token": "at", "refresh_token": "rt"}))
    created = {"alice@example.com": {"id": UID_A, "email": "alice@example.com", "personal_plan": "free"}}

    def fake_request(method, path, params=None, **k):
        email = (params or {}).get("email", "")[3:]
        # the pre-signup existence check runs before the account exists
        if not cloud.has_active_session():
            return []
        return [created[email]] if email in created else []

    monkeypatch.setattr(cloud, "_request", fake_request)
    profile = cloud.sign_up("Alice@Example.com", "password123")
    assert profile["id"] == UID_A


def test_bug_fetch_recent_sessions_retries_network_failures_four_times(monkeypatch):
    """The select-list fallback loop catches every SupabaseError, including plain
    network errors, so when offline fetch_recent_sessions makes 4 sequential requests
    (4 x 8 s timeout = 32 s of a frozen prompt) before failing. Correct: only fall back
    on an unknown-column error (_is_unknown_column_error); re-raise anything else."""
    calls = []

    def offline(method, path, params=None, **k):
        calls.append(params.get("select"))
        raise cloud.SupabaseError("GET Debug_Sessions -> network error: [Errno 101] Network is unreachable")

    monkeypatch.setattr(cloud, "_request", offline)
    with pytest.raises(cloud.SupabaseError):
        cloud.fetch_recent_sessions("team-1", None)
    assert len(calls) == 1


# ============================================================================ ok

def test_ok_encode_decode_roundtrip_and_legacy_passthrough():
    entry = {"t": 1.5, "question": "why nan?", "answer": "lr too high", "nested": [1, {"a": None}]}
    enc = cloud.encode_entry(entry)
    assert enc.startswith("PZ1:")
    assert cloud.decode_entry(enc) == entry
    assert cloud.decode_entry({"legacy": True}) == {"legacy": True}
    assert cloud.decode_entry("plain traceback") == "plain traceback"
    assert cloud.decode_entry("PZ1:not-base64!!") == "PZ1:not-base64!!"
    assert cloud.decode_entries(None) == []


def test_ok_encode_entry_scrubs_secrets():
    key = "sk-ant-api03-" + "x" * 40
    enc = cloud.encode_entry({"answer": f"use {key}"})
    assert key not in json.dumps(cloud.decode_entry(enc))


def test_ok_scrub_env_var_values(monkeypatch):
    monkeypatch.setenv("MY_CUSTOM_API_KEY", "zzzz-custom-secret-9999")
    assert "zzzz-custom-secret-9999" not in cloud.scrub_secrets("key=zzzz-custom-secret-9999 !")
    assert cloud.scrub_secrets(None) is None
    assert cloud.scrub_secrets_deep({"a": ["Bearer abcdefghijklmnop"]}) == {"a": ["[REDACTED_SECRET]"]}


def test_ok_credentials_roundtrip_ttl_and_perms():
    cloud.save_cached_credentials(UID_A, "a@example.com", team_id="t1", session_token="tok")
    data = cloud.load_cached_credentials()
    assert data["user_id"] == UID_A and data["session_token"] == "tok"
    assert stat.S_IMODE(os.stat(cloud.CACHE_PATH).st_mode) == 0o600
    data["cached_at"] = time.time() - (cloud.SESSION_TTL_DAYS + 1) * 86400
    cloud.CACHE_PATH.write_text(json.dumps(data))
    assert cloud.load_cached_credentials() is None
    cloud.CACHE_PATH.write_text("not json")
    assert cloud.load_cached_credentials() is None
    cloud.clear_cached_credentials()
    assert not cloud.CACHE_PATH.exists()


def test_ok_profile_merge_never_stores_none():
    cloud.save_cached_profile(agent_provider="Claude", agent_env_key="ANTHROPIC_API_KEY")
    cloud.save_cached_profile(last_git_commit_sha="abc", agent_env_key=None)
    p = cloud.load_cached_profile()
    assert p == {"agent_provider": "Claude", "agent_env_key": "ANTHROPIC_API_KEY", "last_git_commit_sha": "abc"}


def test_ok_input_validation():
    assert cloud.is_valid_email("a.b+c@example.co.uk")
    assert not cloud.is_valid_email("a@b,or(id.gt.0)")
    assert cloud.is_valid_join_code("A1B2C3D4")
    assert not cloud.is_valid_join_code("A1B2.eq")
    assert cloud.find_user_by_email("bad,email") is None
    assert cloud.find_teams_for_user("not-a-uuid") == []


def test_ok_verify_session_token_semantics(monkeypatch):
    good = cloud._hash_session_token("tok")
    monkeypatch.setattr(cloud, "_request", lambda *a, **k: [{"session_token_hash": good}])
    assert cloud.verify_session_token(UID_A, "tok") is True
    assert cloud.verify_session_token(UID_A, "other") is False
    monkeypatch.setattr(cloud, "_request", lambda *a, **k: [{"session_token_hash": None}])
    assert cloud.verify_session_token(UID_A, "tok") is None


def test_ok_request_url_encoding_and_auth_header(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["auth"] = req.headers.get("Authorization")
        return _Resp([])

    monkeypatch.setattr(cloud.urllib.request, "urlopen", fake_urlopen)
    cloud._request("GET", "profiles", params={"email": "eq.a+b@x.com", "select": "id,email"})
    assert "email=eq.a%2Bb@x.com" in seen["url"] or "email=eq.a%2Bb%40x.com" in seen["url"]
    assert "select=id,email" in seen["url"]
    assert seen["auth"] == f"Bearer {cloud.SUPABASE_KEY}"
    cloud.set_session("user-jwt", "rt")
    cloud._request("GET", "profiles")
    assert seen["auth"] == "Bearer user-jwt"


def test_ok_request_http_error_becomes_supabaseerror(monkeypatch):
    def err(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 400, "bad", {}, io.BytesIO(b'{"code":"PGRST204"}'))

    monkeypatch.setattr(cloud.urllib.request, "urlopen", err)
    with pytest.raises(cloud.SupabaseError) as ei:
        cloud._request("GET", "profiles")
    assert cloud._is_unknown_column_error(ei.value)


def test_ok_background_sync_reports_supabase_error_once(monkeypatch):
    def fail(session_id, fields, timeout=5):
        raise cloud.SupabaseError("HTTP 500")

    monkeypatch.setattr(cloud, "patch_debug_session", fail)
    errors = []
    sync = cloud.BackgroundSync(on_error=errors.append)
    for _ in range(3):
        assert sync.submit("sid", {"x": 1}, wait=True).wait(2)
    assert errors == ["HTTP 500"]


def test_ok_telemetry_opt_out(monkeypatch):
    for v in ("off", "0", "False", " no "):
        monkeypatch.setenv("PULSE_TELEMETRY", v)
        assert not cloud.telemetry_enabled()
    monkeypatch.setenv("PULSE_TELEMETRY", "on")
    assert cloud.telemetry_enabled()


def test_ok_delete_team_requires_admin(monkeypatch):
    monkeypatch.setattr(cloud, "_teams_request", lambda *a, **k: [{"team_id": "t", "admin_ids": [UID_A]}])
    monkeypatch.setattr(cloud, "_request", lambda *a, **k: None)
    with pytest.raises(cloud.SupabaseError):
        cloud.delete_team("t", UID_B)
    cloud.delete_team("t", UID_A)


def test_ok_environment_info_has_no_secrets(monkeypatch):
    monkeypatch.setattr(cloud, "_gpu_info", lambda: {"gpu_name": None, "gpu_count": 0, "cuda_version": None})
    info = cloud.collect_environment_info()
    assert set(info) == {"type", "t", "gpu_name", "gpu_count", "cuda_version", "python_version", "framework_versions"}
