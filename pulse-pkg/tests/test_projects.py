"""workspace -> project -> run: the Projects layer in pulse_supabase.py and the
unattended project choice in PulseCLI. Every network call is mocked."""
import pytest

from pulse import pulse_supabase as cloud
from pulse.pulse_cli import PulseCLI

ADMIN = "11111111-1111-1111-1111-111111111111"
MEMBER = "22222222-2222-2222-2222-222222222222"
TEAM_ID = "33333333-3333-3333-3333-333333333333"
TEAM = {"team_id": TEAM_ID, "admin_ids": [ADMIN], "members": [ADMIN, MEMBER]}

OPEN = {"project_id": "p-open", "team_id": TEAM_ID, "name": "Open", "is_secret": False, "members": [ADMIN]}
NULL_FLAG = {"project_id": "p-null", "team_id": TEAM_ID, "name": "Legacy", "is_secret": None, "members": []}
SECRET = {"project_id": "p-secret", "team_id": TEAM_ID, "name": "Hidden", "is_secret": True, "members": [ADMIN]}


class _Recorder:
    def __init__(self, rows=None):
        self.rows = rows
        self.calls = []

    def __call__(self, method, path, params=None, body=None, prefer=None, timeout=8):
        self.calls.append((method, path, params, body))
        if callable(self.rows):
            return self.rows(method, path, params, body)
        return self.rows if self.rows is not None else []


def test_secret_project_is_hidden_from_a_member_who_has_not_joined():
    assert cloud.project_visible_to(OPEN, TEAM, MEMBER)
    assert cloud.project_visible_to(NULL_FLAG, TEAM, MEMBER)
    assert not cloud.project_visible_to(SECRET, TEAM, MEMBER)


def test_secret_project_is_visible_to_admins_and_to_joined_members():
    assert cloud.project_visible_to(SECRET, TEAM, ADMIN)
    assert cloud.project_visible_to(dict(SECRET, members=[ADMIN, MEMBER]), TEAM, MEMBER)


def test_member_listing_filters_server_side_and_client_side(monkeypatch):
    rec = _Recorder([OPEN, SECRET])  # a server that (wrongly) hands back the secret one too
    monkeypatch.setattr(cloud, "_request", rec)
    got = cloud.find_projects_for_team(TEAM, MEMBER)
    assert [p["project_id"] for p in got] == ["p-open"]
    params = rec.calls[0][2]
    assert params["team_id"] == f"eq.{TEAM_ID}"
    assert "or" in params and MEMBER in params["or"]


def test_admin_listing_asks_for_every_project(monkeypatch):
    rec = _Recorder([OPEN, SECRET])
    monkeypatch.setattr(cloud, "_request", rec)
    got = cloud.find_projects_for_team(TEAM, ADMIN)
    assert [p["project_id"] for p in got] == ["p-open", "p-secret"]
    assert "or" not in rec.calls[0][2]


def test_listing_never_selects_the_secret_join_code(monkeypatch):
    rec = _Recorder([])
    monkeypatch.setattr(cloud, "_request", rec)
    cloud.find_projects_for_team(TEAM, ADMIN)
    assert "secret_join_code" not in rec.calls[0][2]["select"]


def test_create_project_normal_has_no_join_code(monkeypatch):
    rec = _Recorder(lambda m, p, params, body: [dict(body, project_id="p1")])
    monkeypatch.setattr(cloud, "_request", rec)
    project = cloud.create_project(TEAM_ID, ADMIN, "  Run A  ", repo="https://x:tok@github.com/o/r.git")
    body = rec.calls[0][3]
    assert body["team_id"] == TEAM_ID and body["name"] == "Run A"
    assert body["is_secret"] is False and body["members"] == [ADMIN]
    assert "secret_join_code" not in body
    assert "tok" not in body["repo"]
    assert project["project_id"] == "p1"


def test_create_project_secret_gets_a_join_code(monkeypatch):
    rec = _Recorder(lambda m, p, params, body: [dict(body, project_id="p1")])
    monkeypatch.setattr(cloud, "_request", rec)
    project = cloud.create_project(TEAM_ID, ADMIN, "Quiet", is_secret=True)
    assert cloud.is_valid_join_code(project["secret_join_code"])


def test_create_project_requires_a_name():
    with pytest.raises(cloud.SupabaseError):
        cloud.create_project(TEAM_ID, ADMIN, "   ")


def test_join_secret_project_adds_the_member_and_is_scoped_to_the_workspace(monkeypatch):
    def rows(method, path, params, body):
        return [dict(SECRET)] if method == "GET" else []

    rec = _Recorder(rows)
    monkeypatch.setattr(cloud, "_request", rec)
    project = cloud.join_secret_project(TEAM, "abcd1234", MEMBER)
    get_params = rec.calls[0][2]
    assert get_params["team_id"] == f"eq.{TEAM_ID}" and get_params["secret_join_code"] == "eq.ABCD1234"
    assert MEMBER in project["members"]
    assert rec.calls[1][0] == "PATCH" and MEMBER in rec.calls[1][3]["members"]


def test_join_secret_project_with_a_wrong_code_fails(monkeypatch):
    monkeypatch.setattr(cloud, "_request", _Recorder([]))
    with pytest.raises(cloud.SupabaseError):
        cloud.join_secret_project(TEAM, "ZZZZ9999", MEMBER)


def test_leave_project_only_removes_that_member(monkeypatch):
    rec = _Recorder(lambda m, p, params, body: [dict(SECRET, members=[ADMIN, MEMBER])] if m == "GET" else [])
    monkeypatch.setattr(cloud, "_request", rec)
    cloud.leave_project("33333333-3333-3333-3333-333333333330", MEMBER)
    assert rec.calls[-1][3] == {"members": [ADMIN]}


def test_only_a_workspace_admin_can_delete_a_project(monkeypatch):
    pid = "44444444-4444-4444-4444-444444444444"
    rec = _Recorder(lambda m, p, params, body: [dict(OPEN, project_id=pid)] if m == "GET" else [])
    monkeypatch.setattr(cloud, "_request", rec)
    with pytest.raises(cloud.SupabaseError):
        cloud.delete_project(pid, TEAM, MEMBER)
    cloud.delete_project(pid, TEAM, ADMIN)
    assert rec.calls[-1][0] == "DELETE"


def test_saved_vars_and_repo_are_written_to_the_project(monkeypatch):
    rec = _Recorder([])
    monkeypatch.setattr(cloud, "_request", rec)
    cloud.update_project_saved_vars("p1", ["loss", "acc"])
    cloud.update_project_repo("p1", "https://tok@github.com/o/r")
    assert rec.calls[0][1] == "Projects" and rec.calls[0][3] == {"saved_vars": ["loss", "acc"]}
    assert rec.calls[1][1] == "Projects" and "tok" not in rec.calls[1][3]["repo"]


def test_teams_select_no_longer_asks_for_repo():
    assert "repo" not in cloud._TEAM_SELECT.split(",")


def test_debug_sessions_hang_off_the_project(monkeypatch):
    rec = _Recorder(lambda m, p, params, body: [{"id": "s1"}] if m == "POST" else [])
    monkeypatch.setattr(cloud, "_request", rec)
    assert cloud.create_debug_session("proj-1", ADMIN, git_commit_sha="abc") == "s1"
    body = rec.calls[0][3]
    assert body["project_id"] == "proj-1" and "team_id" not in body
    cloud.fetch_recent_sessions("proj-1", ADMIN)
    assert rec.calls[1][2]["project_id"] == "eq.proj-1"


def test_cached_credentials_remember_the_project(tmp_path, monkeypatch):
    monkeypatch.setattr(cloud, "CACHE_PATH", tmp_path / "credentials.json")
    cloud.save_cached_credentials(ADMIN, "a@example.com", team_id="t1", project_id="p1")
    assert cloud.load_cached_credentials()["project_id"] == "p1"


# ---------------------------------------------------------------- unattended CLI choice

def _cli(monkeypatch, config=None, resumed=False):
    cli = PulseCLI.__new__(PulseCLI)
    cli.config = config or {}
    cli.user_id, cli.email = ADMIN, "a@example.com"
    cli.team_id, cli.team_join_code, cli.team_admin_ids = TEAM_ID, "ABCD1234", [ADMIN]
    cli.project_id = cli.project_name = cli.project_repo = None
    cli.project_is_secret = False
    cli.non_interactive = True
    cli._resumed_after_restart = resumed
    cli._repo_cwd = None
    monkeypatch.setattr(cloud, "save_cached_credentials", lambda *a, **k: None)
    monkeypatch.setattr(cloud, "git_remote_url", lambda cwd=None: None)
    return cli


def test_unattended_reuses_the_cached_project(monkeypatch):
    cli = _cli(monkeypatch)
    other = dict(OPEN, project_id="p-other", name="Other")
    cli._project_flow_unattended(cli._team_ref(), [OPEN, other], "p-other")
    assert cli.project_id == "p-other"


def test_unattended_with_several_projects_and_no_pick_continues_without_one(monkeypatch):
    cli = _cli(monkeypatch)
    cli._project_flow_unattended(cli._team_ref(), [OPEN, dict(OPEN, project_id="p2")], None)
    assert cli.project_id is None


def test_unattended_picks_the_only_project(monkeypatch):
    cli = _cli(monkeypatch)
    cli._project_flow_unattended(cli._team_ref(), [OPEN], None)
    assert cli.project_id == "p-open"


def test_unattended_picks_a_configured_project_by_name(monkeypatch):
    cli = _cli(monkeypatch, config={"project": "legacy"})
    cli._project_flow_unattended(cli._team_ref(), [OPEN, NULL_FLAG], None)
    assert cli.project_id == "p-null"


def test_unattended_creates_a_first_project_in_an_empty_workspace(monkeypatch):
    cli = _cli(monkeypatch)
    rec = _Recorder(lambda m, p, params, body: [dict(body, project_id="p-new")])
    monkeypatch.setattr(cloud, "_request", rec)
    cli._project_flow_unattended(cli._team_ref(), [], None)
    assert cli.project_id == "p-new" and rec.calls[0][3]["team_id"] == TEAM_ID


def test_unattended_can_join_a_secret_project_by_code(monkeypatch):
    cli = _cli(monkeypatch, config={"project": {"action": "join", "join_code": "ABCD1234"}})
    cli.team_admin_ids = []
    rec = _Recorder(lambda m, p, params, body: [dict(SECRET)] if m == "GET" else [])
    monkeypatch.setattr(cloud, "_request", rec)
    cli._project_flow_unattended(cli._team_ref(), [OPEN, dict(OPEN, project_id="p2")], None)
    assert cli.project_id == "p-secret" and cli.project_is_secret
