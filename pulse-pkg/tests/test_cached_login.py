"""The cached Pulse Cloud login across starts: refresh tokens are single-use."""
import io
import time
import json
import urllib.error

import pytest

from pulse import pulse_cli
from pulse import pulse_supabase as cloud


@pytest.fixture
def cache(tmp_path, monkeypatch):
    path = tmp_path / "credentials.json"
    monkeypatch.setattr(cloud, "CACHE_PATH", path)
    path.write_text(json.dumps({"user_id": "72a1fbfb-1934-4232-b91c-2ca32fadf0e2", "email": "a@b.c",
                                "refresh_token": "old", "cached_at": time.time()}))
    monkeypatch.setattr(cloud, "verify_session_token", lambda uid, tok: None)
    monkeypatch.setattr(cloud, "fetch_user", lambda uid: {"id": uid, "email": "a@b.c"})
    cloud.clear_session()
    yield path
    cloud.clear_session()


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _cli():
    cli = pulse_cli.PulseCLI.__new__(pulse_cli.PulseCLI)
    cli.config, cli.non_interactive = {}, True
    cli.user_id = cli.email = cli.team_id = cli.project_id = None
    return cli


def test_the_new_refresh_token_is_saved_for_the_next_start(cache, monkeypatch):
    body = json.dumps({"access_token": "a1", "refresh_token": "new", "user": {"email": "a@b.c"}}).encode()
    monkeypatch.setattr(cloud.urllib.request, "urlopen", lambda req, timeout=None: Response(body))
    cli = _cli()
    cli._auth_flow()
    assert cli.user_id and json.loads(cache.read_text())["refresh_token"] == "new"


def test_a_refused_token_means_signing_in_again_not_a_fake_sign_in(cache, monkeypatch):
    def refuse(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 400, "invalid refresh token", {}, io.BytesIO(b"{}"))
    monkeypatch.setattr(cloud.urllib.request, "urlopen", refuse)
    cli = _cli()
    cli._auth_flow()                                  # non-interactive: no env login -> local only
    assert cli.user_id is None and not cache.exists()


def test_offline_keeps_the_login(cache, monkeypatch):
    def offline(req, timeout=None):
        raise urllib.error.URLError("no network")
    monkeypatch.setattr(cloud.urllib.request, "urlopen", offline)
    cli = _cli()
    cli._auth_flow()
    assert cache.exists() and json.loads(cache.read_text())["refresh_token"] == "old"


def test_a_cloud_error_while_choosing_does_not_stop_the_start(monkeypatch):
    from pulse import pulse_headless as headless
    monkeypatch.setattr(cloud, "load_cached_credentials", lambda: {"user_id": "u"})
    monkeypatch.setattr(pulse_cli.PulseCLI, "_auth_flow", lambda self: setattr(self, "user_id", "u"))

    def refuse(uid):
        raise cloud.SupabaseError("GET Teams -> HTTP 401: permission denied for table Teams")
    monkeypatch.setattr(cloud, "find_teams_for_user", refuse)
    assert headless.choose_destination(interactive=False) == {}
