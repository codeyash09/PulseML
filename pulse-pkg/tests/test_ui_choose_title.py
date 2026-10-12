"""`pulse_ui.choose` takes the question being asked as `title`.

Two callers ("Run flagged command?" in pulse_cli, "Apply this change?" in pulse_code_agent) pass
it, and the TUI host's own `choose` shows it. The module function used to have no such
parameter, so both prompts died with a TypeError instead of asking the person."""
import pytest

from pulse import pulse_ui as ui


def _options():
    return [ui.Option("Yes", key="y"), ui.Option("No", key="n")]


def test_the_title_reaches_the_host_that_shows_the_question():
    seen = {}

    class Host:
        def choose(self, options, initial=0, hotkeys=None, title="", **_):
            seen.update(title=title, hotkeys=hotkeys)
            return 1

    ui.set_host(Host())
    try:
        assert ui.choose(_options(), title="Apply this change?", hotkeys={"a": "always"}) == 1
    finally:
        ui.set_host(None)
    assert seen == {"title": "Apply this change?", "hotkeys": {"a": "always"}}


def test_without_a_terminal_a_titled_choice_is_unavailable_not_a_crash():
    ui.set_host(None)
    with pytest.raises(ui.Unavailable):      # callers catch this and fall back to a typed y/N
        ui.choose(_options(), title="Run flagged command?")