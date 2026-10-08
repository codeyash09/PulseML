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