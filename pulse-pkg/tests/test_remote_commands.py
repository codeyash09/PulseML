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

import io
import json
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


def test_commands_about_this_terminal_are_refused_from_the_dashboard(tmp_path):
    app = appmod.App(FakeCli(), str(tmp_path))
    for line in ("/exit", "/mouse off", "/copy", "/config", "/agent", "/monitor", "/change train.py", "/run"):
        assert app._remote_refusal(line), line
    for line in ("/findings", "/stop", "/run train.py", "why is the loss flat?"):
        assert app._remote_refusal(line) is None, line


# ---------------------------------------------------------------------------------- questions on the dashboard

PROJECT_ID = "33333333-3333-4333-8333-333333333333"


class Questions:
    """Commands rows for questions: the dashboard's side is `answer`."""

    def __init__(self, monkeypatch):
        self.rows, self.opened, self.finished = {}, [], {}
        monkeypatch.setattr(cloud, "open_question", self.open)
        monkeypatch.setattr(cloud, "read_command", lambda command_id: dict(self.rows[command_id]))
        monkeypatch.setattr(cloud, "finish_command", self.finish)
        monkeypatch.setattr(appmod.DashboardQuestion, "POLL_EVERY", 0.01)

    def open(self, run_id, project_id, user_id, question):
        command_id = f"q{len(self.opened)}"
        self.opened.append((run_id, question))
        self.rows[command_id] = {"id": command_id, "status": "processing", "result": None}
        return command_id

    def answer(self, text, which="q0"):
        assert wait_for(lambda: which in self.rows)
        self.rows[which].update(status="completed", result=text)

    def finish(self, command_id, status, result):
        if self.rows.get(command_id, {}).get("status") == "processing":
            self.rows[command_id].update(status=status, result=result)
            self.finished[command_id] = (status, result)


class SignedInCli(FakeCli):
    user_id = RUNNER_ID
    project_id = PROJECT_ID
    debug_session_id = RUN_ID


def asking(app, work):
    box = {}

    def run():
        try:
            box["value"] = work()
        except BaseException as exc:      # noqa: BLE001 -- the test asserts on it
            box["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    app._job = thread                                            # questions come from a job
    thread.start()
    return thread, box


def test_a_question_is_answered_from_the_dashboard(tmp_path, monkeypatch):
    table = Questions(monkeypatch)
    app = appmod.App(SignedInCli(), str(tmp_path))
    thread, box = asking(app, lambda: app.ask("Apply this change? [y/N]"))
    table.answer("y")
    thread.join(5)
    assert box["value"] == "y"
    assert table.opened[0] == (RUN_ID, {"label": "Apply this change? [y/N]", "options": None})
    assert "Answered on the dashboard." in [e.text for e in app.view.entries]
    assert not table.finished                                    # the dashboard's answer stands


def test_a_choice_is_answered_from_the_dashboard(tmp_path, monkeypatch):
    table = Questions(monkeypatch)
    app = appmod.App(SignedInCli(), str(tmp_path))
    options = [appmod._ui.Option("Keep going"), appmod._ui.Option("Restart")]
    thread, box = asking(app, lambda: app.choose(options, title="The fix is in. Now?"))
    table.answer("#1 Restart")
    thread.join(5)
    assert box["value"] == 1
    assert table.opened[0][1] == {"label": "The fix is in. Now?", "options": ["Keep going", "Restart"]}


def test_answered_at_the_machine_the_dashboard_is_told(tmp_path, monkeypatch):
    table = Questions(monkeypatch)
    app = appmod.App(SignedInCli(), str(tmp_path))
    thread, box = asking(app, lambda: app.ask("Run `rm -rf build`? [y/N]"))
    assert wait_for(lambda: app._question is not None and table.opened)
    for key in ("n", "enter"):
        app.on_key(key)
    thread.join(5)
    assert box["value"] == "n"
    assert wait_for(lambda: "q0" in table.finished)
    assert table.finished["q0"] == ("completed", "Answered at the machine: n")
    table.answer("y")                                            # too late: already answered here
    assert box["value"] == "n"


def test_a_secret_is_never_put_on_the_dashboard(tmp_path, monkeypatch):
    table = Questions(monkeypatch)
    app = appmod.App(SignedInCli(), str(tmp_path))
    thread, box = asking(app, lambda: app.ask("API key", secret=True))
    assert wait_for(lambda: app._question is not None)
    for key in ("k", "enter"):
        app.on_key(key)
    thread.join(5)
    assert box["value"] == "k" and not table.opened


def test_not_signed_in_nothing_is_put_on_the_dashboard(tmp_path, monkeypatch):
    table = Questions(monkeypatch)
    app = appmod.App(FakeCli(), str(tmp_path))
    thread, box = asking(app, lambda: app.ask("Apply? [y/N]"))
    assert wait_for(lambda: app._question is not None)
    app.on_key("enter")
    thread.join(5)
    assert not table.opened


def test_the_queue_leaves_question_rows_alone(monkeypatch):
    table = Queue(monkeypatch, cloud.QUESTION_PREFIX + '{"label": "Apply?"}')
    ran = []
    queue = appmod.CommandQueue(types.SimpleNamespace(user_id=RUNNER_ID), lambda: RUN_ID, ran.append)
    queue.start()
    time.sleep(0.1)
    queue.close()
    assert not ran and not table.finished


def test_headless_waits_for_the_dashboard_then_says_no(tmp_path, monkeypatch):
    from pulse import pulse_headless as headless
    table = Questions(monkeypatch)
    monkeypatch.setattr(headless, "_ANSWER_SECONDS", 0.3)
    app = headless.HeadlessApp(SignedInCli(), str(tmp_path), io.StringIO())
    try:
        app.ask("Apply this change? [y/N]")
        raise AssertionError("no answer should mean no")
    except EOFError:
        pass
    assert wait_for(lambda: "q0" in table.finished)
    assert table.finished["q0"][0] == "failed" and "Nobody answered" in table.finished["q0"][1]


def test_headless_takes_the_dashboard_answer(tmp_path, monkeypatch):
    from pulse import pulse_headless as headless
    table = Questions(monkeypatch)
    app = headless.HeadlessApp(SignedInCli(), str(tmp_path), io.StringIO())
    threading.Timer(0.1, lambda: table.answer("y")).start()
    assert app.ask("Apply this change? [y/N]") == "y"


def test_apply_this_change_carries_the_diff_to_the_dashboard(tmp_path, monkeypatch):
    table = Questions(monkeypatch)
    app = appmod.App(SignedInCli(), str(tmp_path))
    app._edit_pending = appmod.tui.Entry("edit", "", calls=["EDIT: train.py"], output="-lr = 1\n+lr = 0.1")
    thread, box = asking(app, lambda: app.ask("Apply this change? [y/N]"))
    table.answer("n")
    thread.join(5)
    assert table.opened[0][1]["detail"] == "-lr = 1\n+lr = 0.1"


# ---------------------------------------------------------------------------------- the live feed

class LiveRows:
    def __init__(self, monkeypatch):
        self.opened, self.updates, self.finished = [], [], []
        monkeypatch.setattr(cloud, "open_runner_row", self.open)
        monkeypatch.setattr(cloud, "update_runner_row", lambda cid, result: self.updates.append((cid, json.loads(result))))
        monkeypatch.setattr(cloud, "finish_command", lambda cid, status, result: self.finished.append((cid, status)))

    def open(self, run_id, project_id, user_id, command):
        self.opened.append((run_id, command))
        return f"live{len(self.opened)}"


def test_the_live_feed_streams_what_the_agent_is_doing(tmp_path, monkeypatch):
    rows = LiveRows(monkeypatch)
    app = appmod.App(SignedInCli(), str(tmp_path))
    feed = appmod.LiveFeed(app)
    feed.tick()
    assert rows.opened == [(RUN_ID, cloud.LIVE_PREFIX)]
    assert rows.updates[-1][1]["busy"] is False
    sent = len(rows.updates)
    feed.tick()
    assert len(rows.updates) == sent                             # nothing changed, nothing sent
    with app.lock:
        app.view.busy = True
        app._job_from = len(app.view.entries)
        app.view.entries.append(appmod.tui.Entry("thinking", "The loss went NaN at step 40, so", live=True))
        app.view.entries.append(appmod.tui.Entry("tool", "", calls=["READ train.py"]))
        app.view.entries.append(appmod.tui.Entry("trace", "\033[36mweight\033[0m <- train.py:40"))
    feed.tick()
    state = rows.updates[-1][1]
    assert state["busy"] is True and state["t"]
    assert [(a["kind"], a["text"]) for a in state["activity"]] == [
        ("thinking", "The loss went NaN at step 40, so"), ("tool", "READ train.py"),
        ("trace", "weight <- train.py:40")]
    assert state["activity"][0].get("live") == "1"
    assert state["activity"][2]["ansi"] == "\033[36mweight\033[0m <- train.py:40"
    feed.close()
    assert rows.finished == [("live1", "completed")]


def test_the_live_feed_follows_the_row_in_hand(tmp_path, monkeypatch):
    rows = LiveRows(monkeypatch)
    app = appmod.App(SignedInCli(), str(tmp_path))
    feed = appmod.LiveFeed(app)
    feed.tick()
    app.runlog = types.SimpleNamespace(id="44444444-4444-4444-8444-444444444444")   # a run opened
    feed.tick()
    assert [r for r, _c in rows.opened] == [RUN_ID, "44444444-4444-4444-8444-444444444444"]
    assert rows.finished == [("live1", "completed")]             # the app's own row's feed ended


def test_no_live_feed_when_not_signed_in(tmp_path, monkeypatch):
    rows = LiveRows(monkeypatch)
    app = appmod.App(FakeCli(), str(tmp_path))
    appmod.LiveFeed(app).tick()
    assert not rows.opened


def test_a_question_carries_what_was_said_just_before_it(tmp_path, monkeypatch):
    table = Questions(monkeypatch)
    app = appmod.App(SignedInCli(), str(tmp_path))
    app.note("[Pulse] ⚠ This command can delete files: rm -rf build/")
    thread, box = asking(app, lambda: app.ask("Run it anyway? (y/N)"))
    table.answer("n")
    thread.join(5)
    assert "rm -rf build/" in table.opened[0][1]["context"]
