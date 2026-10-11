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
    assert calls[0][2]["params"]["command"] == "not.like.pulse:*"     # the runner's own rows: never claimed
    assert calls[1][2]["params"]["status"] == "eq.pending"
    assert calls[1][2]["body"]["status"] == "processing"
    assert calls[1][2]["body"]["claimed_by"] == RUNNER_ID


def test_claim_commands_rejects_invalid_session_ids(monkeypatch):
    monkeypatch.setattr(cloud, "_request", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))

    assert cloud.claim_commands("not-a-uuid", RUNNER_ID) == []
    assert cloud.claim_commands(RUN_ID, "not-a-uuid") == []


def test_finish_command_persists_only_terminal_states(monkeypatch):
    calls = []
    monkeypatch.setattr(cloud, "_request", lambda *args, **kwargs: calls.append((args, kwargs)) or [{"id": "x"}])

    assert cloud.finish_command("command-1", "completed", "Done") is True

    args, kwargs = calls[0]
    assert args[:2] == ("PATCH", "Commands")
    assert kwargs["params"] == {"id": "eq.command-1", "status": "eq.processing", "select": "id"}
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

    def finish(self, command_id, status, result, overwrite=False):
        if command_id in self.rows and (overwrite or self.rows[command_id]["status"] == "processing"):
            self.rows[command_id].update(status=status, result=result)
            self.finished[command_id] = (status, result)
            return True
        return False


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
    assert state["activity"] == [
        {"kind": "thinking", "text": "The loss went NaN at step 40, so", "live": True},
        {"kind": "tool", "calls": ["READ train.py"], "text": "", "output": ""},
        {"kind": "trace", "text": "weight <- train.py:40", "ansi": "\033[36mweight\033[0m <- train.py:40"}]
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


def test_a_finished_turn_is_kept_on_the_page(tmp_path, monkeypatch):
    rows = LiveRows(monkeypatch)
    finished = []
    monkeypatch.setattr(cloud, "finish_command", lambda cid, status, result: finished.append((cid, status, json.loads(result))))
    app = appmod.App(SignedInCli(), str(tmp_path))
    app._live = feed = appmod.LiveFeed(app)
    feed.tick()                                                  # the idle feed's row: live1
    app.view.entries.append(appmod.tui.Entry("user", "why is the loss flat?"))

    def work():
        with app.lock:
            thinking = appmod.tui.Entry("thinking", "Check the scheduler.")
            thinking.seconds = 3.2
            app.view.entries.append(thinking)
            app.view.entries.append(appmod.tui.Entry("tool", "Looking at the curve.", calls=["CURVE lr"], output="1e-3 ... 1e-7"))
            app.view.entries.append(appmod.tui.Entry("text", "The scheduler drops the rate"))
            app.view.entries.append(appmod.tui.Entry("text", "at step 1500."))

    app._start(work)
    assert wait_for(lambda: app._job is None)
    feed.tick()                                                  # the turn completes live1; a new row follows
    turn = finished[0][2]
    assert finished[0][:2] == ("live1", "completed")
    assert turn["turn"] is True and turn["who"] == "you" and turn["prompt"] == "why is the loss flat?"
    assert turn["activity"] == [
        {"kind": "thinking", "text": "Check the scheduler.", "seconds": 3.2},
        {"kind": "tool", "calls": ["CURVE lr"], "text": "Looking at the curve.", "output": "1e-3 ... 1e-7"},
        {"kind": "text", "text": "The scheduler drops the rate\nat step 1500."}]
    assert turn["ended"] >= turn["started"] > 0
    assert [r for r, _c in rows.opened] == [RUN_ID, RUN_ID]      # the feed goes on in a new row
    assert rows.updates[-1][1]["busy"] is False


# ---------------------------------------------------------------------------------- the bridge's rough edges

OTHER_RUN = "44444444-4444-4444-8444-444444444444"


def test_question_rows_are_scrubbed_of_secrets(monkeypatch):
    sent = []

    def request(method, path, **kwargs):
        sent.append((method, kwargs))
        return [{"id": "q1"}] if method == "POST" else None

    monkeypatch.setattr(cloud, "_request", request)
    key = "sk-or-v1-" + "a" * 40
    cloud.open_question(RUN_ID, PROJECT_ID, RUNNER_ID, {"label": "Run it?", "context": f"export KEY={key}",
                                                        "detail": f"+api = '{key}'"})
    command = sent[0][1]["body"]["command"]
    assert key not in command
    assert json.loads(command[len(cloud.QUESTION_PREFIX):])["label"] == "Run it?"     # still readable JSON


def test_a_half_opened_runner_row_is_ended(monkeypatch):
    sent = []

    def request(method, path, **kwargs):
        sent.append((method, kwargs.get("body")))
        if method == "POST":
            return [{"id": "row1"}]
        if kwargs["body"]["status"] == "processing":
            raise cloud.SupabaseError("network error")
        return None

    monkeypatch.setattr(cloud, "_request", request)
    try:
        cloud.open_runner_row(RUN_ID, PROJECT_ID, RUNNER_ID, cloud.LIVE_PREFIX)
        raise AssertionError("the failure must reach the caller")
    except cloud.SupabaseError:
        pass
    assert sent[-1][0] == "PATCH" and sent[-1][1]["status"] == "failed"


def test_a_claim_failing_midway_hands_over_the_rows_already_claimed(monkeypatch):
    rows = [{"id": "c1", "command": "a"}, {"id": "c2", "command": "b"}]

    def request(method, path, **kwargs):
        if method == "GET":
            return rows
        if kwargs["params"]["id"] == "eq.c2":
            raise cloud.SupabaseError("timeout")
        return [rows[0]]

    monkeypatch.setattr(cloud, "_request", request)
    assert cloud.claim_commands(RUN_ID, RUNNER_ID) == [rows[0]]

    def first_fails(method, path, **kwargs):
        if method == "GET":
            return rows
        raise cloud.SupabaseError("timeout")

    monkeypatch.setattr(cloud, "_request", first_fails)
    try:
        cloud.claim_commands(RUN_ID, RUNNER_ID)
        raise AssertionError("nothing claimed: the failure is the caller's to report")
    except cloud.SupabaseError:
        pass


def test_a_result_that_could_not_be_sent_is_sent_again(monkeypatch):
    table = Queue(monkeypatch, "how is the loss?")
    attempts = []

    def flaky(command_id, status, result):
        attempts.append(command_id)
        if len(attempts) == 1:
            raise cloud.SupabaseError("network error")
        table.finished[command_id] = (status, result)

    monkeypatch.setattr(cloud, "finish_command", flaky)
    errors = []
    queue = appmod.CommandQueue(types.SimpleNamespace(user_id=RUNNER_ID), lambda: RUN_ID, lambda text: "fine",
                                on_error=errors.append)
    queue.start()
    assert wait_for(lambda: "c0" in table.finished)
    queue.close()
    assert table.finished["c0"] == ("completed", "fine") and len(attempts) == 2 and errors


def test_leaving_gives_the_command_in_hand_time_to_say_how_it_ended(monkeypatch):
    table = Queue(monkeypatch, "slow")
    leaving = threading.Event()

    def run(text):
        leaving.wait(5)
        time.sleep(0.2)
        raise RuntimeError("Pulse closed before it got to this command.")

    queue = appmod.CommandQueue(types.SimpleNamespace(user_id=RUNNER_ID), lambda: RUN_ID, run)
    queue.start()
    assert wait_for(lambda: queue._active is not None)
    leaving.set()
    queue.close(wait=3.0)
    assert table.finished["c0"] == ("failed", "Pulse closed before it got to this command.")


def test_the_run_rows_arrays_are_capped(monkeypatch):
    log = appmod.RunLog(types.SimpleNamespace(user_id="u1", project_id="p1"), {"session_id": "r"}, "/p")
    monkeypatch.setattr(appmod.RunLog, "KEEP", {"telemetry": 3, "agent_logs": 2, "incidents": 2,
                                                "error_tracebacks": 2})
    with log._lock:
        for i in range(6):
            log._append("telemetry", {"i": i})
            log._append("incidents", {"i": i})
    assert [e["i"] for e in log._fields["telemetry"]] == [0, 4, 5]      # the environment entry stays
    assert [e["i"] for e in log._fields["incidents"]] == [4, 5]


class Sessions:
    """Debug_Sessions, as RunLog sees it."""

    def __init__(self, monkeypatch, create_fails=0):
        self.create_fails, self.created, self.patches = create_fails, [], []
        monkeypatch.setattr(cloud, "create_debug_session", self.create)
        monkeypatch.setattr(cloud, "patch_debug_session", self.patch)
        monkeypatch.setattr(cloud, "collect_environment_info", lambda: {})
        monkeypatch.setattr(cloud, "current_git_commit_sha", lambda cwd=None: None)

    def create(self, project_id, user_id, git_commit_sha=None):
        if self.create_fails:
            self.create_fails -= 1
            raise cloud.SupabaseError("network error")
        self.created.append(project_id)
        return "row-1"

    def patch(self, session_id, fields, timeout=5):
        self.patches.append({k: ([cloud.decode_entry(e) for e in v] if isinstance(v, list) else v)
                             for k, v in fields.items()})


def test_the_run_row_is_created_later_if_the_first_try_fails(monkeypatch):
    table = Sessions(monkeypatch, create_fails=1)
    log = appmod.RunLog(types.SimpleNamespace(user_id="u1", project_id="p1"), {"session_id": "r"}, "/p")
    log.start()
    log._worker.join(5)
    assert log.id is None
    log.incident("stopped", "x")
    log._worker.join(5)
    assert log.id == "row-1" and table.patches[-1]["telemetry"]       # the first entry too
    log.close()


def test_closing_the_run_row_sends_what_a_send_under_way_missed(monkeypatch):
    table = Sessions(monkeypatch)
    gate, sending = threading.Event(), threading.Event()
    patch = table.patch

    def slow(session_id, fields, timeout=5):
        sending.set()
        gate.wait(5)
        patch(session_id, fields)

    monkeypatch.setattr(cloud, "patch_debug_session", slow)
    log = appmod.RunLog(types.SimpleNamespace(user_id="u1", project_id="p1"), {"session_id": "r"}, "/p")
    log.start()
    assert wait_for(sending.is_set)
    log.incident("stopped", "the run was stopped")          # while the first send is under way
    threading.Timer(0.2, gate.set).start()
    log.close()
    sent = [i["kind"] for body in table.patches for i in body.get("incidents") or []]
    assert sent == ["stopped"]


def test_an_interrupted_question_is_taken_down(tmp_path, monkeypatch):
    table = Questions(monkeypatch)
    app = appmod.App(SignedInCli(), str(tmp_path))

    class Interrupted(threading.Event):
        def wait(self, timeout=None):
            assert wait_for(lambda: "q0" in table.rows)
            raise KeyboardInterrupt                              # Ctrl+C on the job lands while it waits

    question = appmod._Question("Apply this change? [y/N]")
    question.done = Interrupted()
    thread, box = asking(app, lambda: app._ask(question))
    thread.join(5)
    assert isinstance(box["error"], KeyboardInterrupt)
    assert app._question is None and app.view.question == ""
    assert wait_for(lambda: "q0" in table.finished)
    assert table.finished["q0"][0] == "failed"


def test_leaving_pulse_ends_the_open_question_row(tmp_path, monkeypatch):
    table = Questions(monkeypatch)
    app = appmod.App(SignedInCli(), str(tmp_path))
    thread, box = asking(app, lambda: app.ask("Apply this change? [y/N]"))
    assert wait_for(lambda: app._question is not None and "q0" in table.rows)
    app.done = True                                              # what the loop's end does
    with app.lock:
        app._question.cancelled = True
        app._question.done.set()
    thread.join(5)
    app._wait_for_rows()
    assert table.finished["q0"] == ("failed", "No longer asked: Pulse closed.")


def test_answered_at_both_ends_the_row_says_which_answer_was_used(tmp_path, monkeypatch):
    table = Questions(monkeypatch)
    monkeypatch.setattr(appmod.DashboardQuestion, "POLL_EVERY", 30)    # the dashboard's answer is not read in time
    app = appmod.App(SignedInCli(), str(tmp_path))
    thread, box = asking(app, lambda: app.ask("Apply this change? [y/N]"))
    assert wait_for(lambda: app._question is not None and "q0" in table.rows)
    table.answer("y")                                            # on the dashboard...
    for key in ("n", "enter"):                                   # ...and here, first
        app.on_key(key)
    thread.join(5)
    assert box["value"] == "n"
    assert wait_for(lambda: table.rows["q0"]["result"] == "Answered at the machine: n")


def test_a_dashboard_answer_that_lost_is_not_counted_as_used(tmp_path, monkeypatch):
    table = Questions(monkeypatch)
    app = appmod.App(SignedInCli(), str(tmp_path))
    question = appmod._Question("Apply? [y/N]")
    posted = app._post_question(question)
    assert wait_for(lambda: "q0" in table.rows)
    with app.lock:                                               # answered here first
        question.answer = "n"
        question.done.set()
    table.answer("y")
    assert wait_for(lambda: not posted._thread.is_alive())
    assert not posted.answered_there                             # so close() still writes the row
    posted.close("completed", "Answered at the machine: n").join(5)
    assert table.rows["q0"]["result"] == "Answered at the machine: n"


def test_free_text_typed_at_the_machine_is_not_sent_to_the_dashboard(tmp_path, monkeypatch):
    table = Questions(monkeypatch)
    app = appmod.App(SignedInCli(), str(tmp_path))
    thread, box = asking(app, lambda: app.ask("Your database URL"))
    assert wait_for(lambda: app._question is not None and table.opened)
    for key in "pg://me":
        app.on_key(key)
    app.on_key("enter")
    thread.join(5)
    assert box["value"] == "pg://me"
    assert wait_for(lambda: "q0" in table.finished)
    assert table.finished["q0"] == ("completed", "Answered at the machine.")


def test_more_account_and_workspace_commands_are_refused_from_afar(tmp_path):
    app = appmod.App(FakeCli(), str(tmp_path))
    for line in ("/logout", "/deleteaccount", "/password", "/recover", "/admin add x", "/webhook https://x",
                 "/telemetry off", "/project", "/repo https://x", "/repository x", "/review off"):
        assert app._remote_refusal(line), line
    assert app._remote_refusal("/findings") is None


def test_a_dashboard_turn_keeps_its_prompt(tmp_path, monkeypatch):
    app = appmod.App(FakeCli(), str(tmp_path))
    kept = []
    app._live = types.SimpleNamespace(turn_ended=lambda record, target=None: kept.append(record))
    app.install()
    try:
        remote(app, "/help")
    finally:
        app.uninstall()
    assert kept and kept[0]["prompt"] == "/help" and kept[0]["who"] == "dashboard"


def test_a_turn_is_kept_on_the_page_of_the_row_it_started_on(tmp_path, monkeypatch):
    rows = LiveRows(monkeypatch)
    finished = []
    monkeypatch.setattr(cloud, "finish_command", lambda cid, status, result: finished.append((cid, json.loads(result))))
    app = appmod.App(SignedInCli(), str(tmp_path))
    app._live = feed = appmod.LiveFeed(app)
    feed.tick()                                                  # live1, on the app's own row

    def work():
        app.note("looked at it")
        app.runlog = types.SimpleNamespace(id=OTHER_RUN)        # the run in hand changes as it ends

    app._start(work)
    assert wait_for(lambda: app._job is None)
    feed.tick()
    turn = [cid for cid, body in finished if body.get("turn")]
    assert turn == ["live1"]                                     # the row that streamed it, on RUN_ID
    assert rows.opened[-1][0] == OTHER_RUN                       # the feed then follows the new row


def test_a_turn_whose_send_failed_is_retried_in_the_same_row(tmp_path, monkeypatch):
    rows = LiveRows(monkeypatch)
    sends = []

    def flaky(cid, status, result):
        sends.append(cid)
        if len(sends) == 1:
            raise cloud.SupabaseError("network error")

    monkeypatch.setattr(cloud, "finish_command", flaky)
    app = appmod.App(SignedInCli(), str(tmp_path))
    feed = appmod.LiveFeed(app)
    feed.turn_ended({"turn": True, "activity": [{"kind": "note", "text": "x"}]}, (RUN_ID, PROJECT_ID, RUNNER_ID))
    try:
        feed.tick()
    except cloud.SupabaseError:
        pass
    feed.tick()
    assert sends[:2] == ["live1", "live1"] and len(rows.opened) == 2     # the turn's row, then the feed's own


def test_closing_the_live_feed_waits_for_a_tick_under_way(tmp_path, monkeypatch):
    rows = LiveRows(monkeypatch)
    finished, gate = [], threading.Event()

    def slow(cid, status, result):
        gate.wait(5)
        finished.append((cid, json.loads(result)))

    monkeypatch.setattr(cloud, "finish_command", slow)
    monkeypatch.setattr(appmod.LiveFeed, "EVERY", 0.01)
    app = appmod.App(SignedInCli(), str(tmp_path))
    feed = appmod.LiveFeed(app)
    feed.turn_ended({"turn": True, "activity": [{"kind": "note", "text": "x"}]})
    feed.start()
    assert wait_for(lambda: rows.opened)                         # the tick is sending the turn
    threading.Timer(2.5, gate.set).start()                       # longer than close's join
    feed.close()
    assert sum(1 for _cid, body in finished if body.get("turn")) == 1


def test_a_dashboard_command_and_a_problem_turn_never_start_together(tmp_path, monkeypatch):
    app = appmod.App(FakeCli(), str(tmp_path))
    held = []

    def start(work, *args):
        other = threading.Thread(target=lambda: held.append(not app.lock.acquire(timeout=0)))
        other.start()
        other.join()

    monkeypatch.setattr(app, "_start", start)
    app._remote_pending.append({"line": "/help", "session": None, "done": threading.Event()})
    app._start_pending_remote_command()
    assert held == [True]                                        # started with the lock still held


def test_headless_says_a_question_is_waiting(tmp_path, monkeypatch):
    from pulse import pulse_headless as headless
    table = Questions(monkeypatch)
    app = headless.HeadlessApp(SignedInCli(), str(tmp_path), io.StringIO())
    seen = []
    threading.Timer(0.2, lambda: (seen.append(app._live_state()["asking"]), table.answer("y"))).start()
    assert app.ask("Apply this change? [y/N]") == "y"
    assert seen == [True] and app._question is None


def test_headless_too_late_answer_does_not_stand(tmp_path, monkeypatch):
    from pulse import pulse_headless as headless
    table = Questions(monkeypatch)
    monkeypatch.setattr(headless, "_ANSWER_SECONDS", 0.2)
    monkeypatch.setattr(appmod.DashboardQuestion, "POLL_EVERY", 30)
    app = headless.HeadlessApp(SignedInCli(), str(tmp_path), io.StringIO())
    threading.Timer(0.15, lambda: table.answer("y")).start()
    try:
        app.ask("Apply this change? [y/N]")
        raise AssertionError("the answer was no")
    except EOFError:
        pass
    assert wait_for(lambda: table.rows["q0"]["status"] == "failed")
    assert "Nobody answered" in table.rows["q0"]["result"]


# ---------------------------------------------------------------------------------- start_run

class FakeProcess:
    def __init__(self, pid=4242, code=None):
        self.pid, self.code = pid, code

    def poll(self):
        return self.code


def test_start_run_refuses_while_a_run_is_live(tmp_path, monkeypatch):
    app = appmod.App(FakeCli(), str(tmp_path))
    monkeypatch.setattr(app, "_launch", lambda *a, **k: (_ for _ in ()).throw(AssertionError("started")))
    app.console, app.session, app._status = object(), {"session_id": "s1", "script": "/p/train.py"}, "live"
    assert "already running" in app._agent_start_run("train.py")
    app.console = None
    app._slow_starts.append({"process": FakeProcess(), "argv": ["/p/train.py"], "cwd": "/p", "log": "/l",
                             "started": time.time(), "replace": False})
    assert "still starting" in app._agent_start_run("train.py")


def test_a_run_slow_to_report_is_kept_and_opened_when_it_does(tmp_path, monkeypatch):
    from pulse import pulse_console as con
    app = appmod.App(FakeCli(), str(tmp_path))
    process = FakeProcess()
    app._slow_starts.append({"process": process, "argv": ["/p/train.py"], "cwd": "/p", "log": "/l",
                             "started": time.time(), "replace": False})
    found = []
    monkeypatch.setattr(con, "discover", lambda: found)
    opened = []
    monkeypatch.setattr(app, "_open_run", lambda session, replace=False: opened.append(session["session_id"]))
    app._open_slow_starts()
    assert not opened and app._slow_starts
    found.append({"session_id": "s9", "script": "/p/train.py", "pid": process.pid})
    app._slow_at = 0.0
    app._open_slow_starts()
    assert opened == ["s9"] and app.launched["s9"]["process"] is process and not app._slow_starts
