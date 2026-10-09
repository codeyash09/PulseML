from pulse import pulse_supabase as cloud


RUN_ID = "11111111-1111-4111-8111-111111111111"
RUNNER_ID = "22222222-2222-4222-8222-222222222222"


def test_claim_commands_only_returns_rows_claimed_by_this_runner(monkeypatch):
    calls = []
    rows = [
        {"id": "command-1", "command": "check loss"},
        {"id": "command-2", "command": "show metrics"},
    ]

    def request(method, path, **kwargs):
        calls.append((method, path, kwargs))
        if method == "GET":
            return rows
        if kwargs["params"]["id"] == "eq.command-1":
            return [rows[0]]
        return []

    monkeypatch.setattr(cloud, "_request", request)

    claimed = cloud.claim_commands(RUN_ID, RUNNER_ID)

    assert claimed == [rows[0]]
    assert calls[0][2]["params"]["status"] == "eq.pending"
    assert calls[1][2]["params"]["status"] == "eq.pending"
    assert calls[1][2]["body"]["status"] == "processing"
    assert calls[1][2]["body"]["claimed_by"] == RUNNER_ID


def test_claim_commands_rejects_invalid_session_ids(monkeypatch):
    monkeypatch.setattr(cloud, "_request", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))

    assert cloud.claim_commands("not-a-uuid", RUNNER_ID) == []
    assert cloud.claim_commands(RUN_ID, "not-a-uuid") == []


def test_finish_command_persists_only_terminal_states(monkeypatch):
    calls = []
    monkeypatch.setattr(cloud, "_request", lambda *args, **kwargs: calls.append((args, kwargs)))

    cloud.finish_command("command-1", "completed", "Done")

    args, kwargs = calls[0]
    assert args[:2] == ("PATCH", "Commands")
    assert kwargs["params"] == {"id": "eq.command-1", "status": "eq.processing"}
    assert kwargs["body"]["status"] == "completed"
    assert kwargs["body"]["result"] == "Done"

    try:
        cloud.finish_command("command-1", "pending", "")
    except ValueError as error:
        assert "completed or failed" in str(error)
    else:
        raise AssertionError("non-terminal command state was accepted")

# ---------------------------------------------------------------------------------- the runner's side

import threading
import time
import types

from pulse import pulse_app as appmod


def wait_for(condition, seconds=5.0):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


class Queue:
    """The Commands table, as claim_commands/finish_command see it."""

    def __init__(self, monkeypatch, *texts):
        self.pending = [{"id": f"c{i}", "command": text} for i, text in enumerate(texts)]
        self.finished = {}
        monkeypatch.setattr(cloud, "claim_commands", self.claim)
        monkeypatch.setattr(cloud, "finish_command", self.finish)
        monkeypatch.setattr(appmod.CommandQueue, "POLL_EVERY", 0.01)

    def claim(self, run_id, runner_id, limit=5):
        claimed, self.pending = self.pending, []
        return claimed

    def finish(self, command_id, status, result):
        self.finished[command_id] = (status, result)


def test_the_queue_runs_commands_in_turn_and_answers_each(monkeypatch):
    table = Queue(monkeypatch, "how is the loss?", "/exit", "/boom")
    ran = []

    def run(text):
        ran.append(text)
        if text == "/boom":
            raise RuntimeError("it broke")
        return f"answer to {text}"

    queue = appmod.CommandQueue(types.SimpleNamespace(user_id=RUNNER_ID), lambda: RUN_ID, run,
                                refuse=lambda text: "not from afar" if text == "/exit" else None)
    queue.start()
    assert wait_for(lambda: len(table.finished) == 3)
    queue.close()
    assert ran == ["how is the loss?", "/boom"]                    # the refused one never ran
    assert table.finished["c0"] == ("completed", "answer to how is the loss?")
    assert table.finished["c1"] == ("failed", "not from afar")
    assert table.finished["c2"] == ("failed", "it broke")


def test_commands_claimed_but_not_started_are_failed_when_the_queue_closes(monkeypatch):
    table = Queue(monkeypatch, "first", "second")
    release = threading.Event()
    queue = appmod.CommandQueue(types.SimpleNamespace(user_id=RUNNER_ID), lambda: RUN_ID,
                                lambda text: release.wait(5) and "done")
    queue.start()
    assert wait_for(lambda: queue._active is not None)
    queue.close()
    release.set()
    assert wait_for(lambda: len(table.finished) == 2)
    assert table.finished["c1"][0] == "failed" and "stopped watching" in table.finished["c1"][1]
    assert table.finished["c0"] == ("completed", "done")


def test_no_row_yet_means_nothing_is_claimed(monkeypatch):
    table = Queue(monkeypatch, "early")
    queue = appmod.CommandQueue(types.SimpleNamespace(user_id=RUNNER_ID), lambda: None, lambda text: "x")
    queue.start()
    time.sleep(0.1)
    queue.close()
    assert table.pending and not table.finished


class FakeCli:
    agent_provider = None
    agent_key = None
    agent_model_string = None
    agent_api_base = None
    focus = []
    _project_root = None


def remote(app, line, session_id=None):
    """Send `line` as the dashboard would and let the app's loop take it up."""
    box = {}

    def send():
        try:
            box["result"] = app._run_remote_command(line, session_id)
        except Exception as exc:      # noqa: BLE001 -- the test asserts on it
            box["error"] = str(exc)

    thread = threading.Thread(target=send, daemon=True)
    thread.start()
    assert wait_for(lambda: app._remote_pending or box)
    while thread.is_alive():
        app._start_pending_remote_command()                      # what the app's loop does
        time.sleep(0.01)
    return box


def test_a_dashboard_prompt_returns_what_the_app_showed(tmp_path):
    app = appmod.App(FakeCli(), str(tmp_path))
    app.install()
    try:
        box = remote(app, "/help")
    finally:
        app.uninstall()
    assert "/findings" in box["result"] or "/monitor" in box["result"]
    lines = [e.text for e in app.view.entries]
    assert "From the dashboard:" in lines and "/help" in lines    # the person sees it was sent


def test_a_prompt_for_a_run_no_longer_watched_fails_with_why(tmp_path):
    app = appmod.App(FakeCli(), str(tmp_path))
    box = remote(app, "/findings", session_id="gone")
    assert "no longer watching" in box["error"]


def test_a_question_nobody_is_there_to_answer_is_declined(tmp_path, monkeypatch):
    app = appmod.App(FakeCli(), str(tmp_path))

    def handle(line, about_run=False):
        print("Planning a change.")
        try:
            input("Apply this change? [y/N] ")
        except EOFError:
            print("Not applied.")

    monkeypatch.setattr(app, "_handle", handle)
    app.install()
    try:
        box = remote(app, "fix it")
    finally:
        app.uninstall()
    assert app._question is None                                 # nothing left waiting on the screen
    assert "Not applied." in box["result"] and "Apply this change?" in box["result"]


def test_commands_about_this_terminal_are_refused_from_the_dashboard(tmp_path):
    app = appmod.App(FakeCli(), str(tmp_path))
    for line in ("/exit", "/mouse off", "/copy", "/config", "/agent", "/monitor", "/change train.py", "/run"):
        assert app._remote_refusal(line), line
    for line in ("/findings", "/stop", "/run train.py", "why is the loss flat?"):
        assert app._remote_refusal(line) is None, line
