"""Regression tests for the CLI's color palette and nested Pulse highlighting."""
import pulse.pulse_cli as cli


def test_pulse_name_stays_orange_inside_offwhite_output(monkeypatch):
    monkeypatch.setattr(cli, "_COLOR_ENABLED", True)

    rendered = cli._highlight_pulse("Pulse CLI restored the patch", cli._OFFWHITE)

    assert rendered == (
        f"{cli._ORANGE}Pulse CLI{cli._RESET}{cli._OFFWHITE} restored the patch"
    )


def test_patch_output_uses_a_defined_offwhite_color():
    assert cli._OFFWHITE == "\033[97m"
    assert cli._BLUE == cli._OFFWHITE
