"""Sweep 2 -- pulse_terminal.classify_command misclassifications.

The five-category policy is intentional; these are commands that fall INTO a category
(or plainly out of one) but are classified wrongly by the rewritten tokenizer/judge."""
import pytest

from pulse import pulse_terminal as T


def _flags(cmd):
    return {k for k, v in T.classify_command(cmd).items() if v}


@pytest.mark.parametrize("cmd", [
    "for f in *.ckpt; do rm -f \"$f\"; done",
    "if [ -d build ]; then rm -rf build; fi",
    "while read f; do rm \"$f\"; done < list.txt",
    "{ rm -rf data; }",
    "! rm -rf data",
    "[ -f x ] || { rm -rf out; }",
])
def test_bug_shell_keywords_hide_the_command_after_them(cmd):
    """Each simple command is judged by argv[0], but shell reserved words (`do`, `then`,
    `else`, `{`, `!` ...) are taken as the program, so the `rm` behind them is never looked
    at. A loop deleting files -- the most ordinary way to delete a set of files -- runs with
    no confirmation. Correct: skip reserved words like wrappers and judge what follows."""
    assert "deletes_files" in _flags(cmd), cmd


@pytest.mark.parametrize("cmd", [
    "nice -n 10 rm -rf data",
    "ionice -c 3 rm -rf data",
    "timeout -s KILL 60 rm -rf data",
    "timeout -k 5 60 rm -rf data",
    "find . -name '*.pt' -print0 | xargs -0 -n 1 rm -f",
    "xargs -I {} rm {} < files.txt",
    "env -u CUDA_VISIBLE_DEVICES rm -rf data",
])
def test_bug_wrapper_option_values_are_taken_as_the_program(cmd):
    """_strip_wrappers drops every '-x' token after a wrapper but not the option's VALUE,
    so for `nice -n 10 rm`, `xargs -n 1 rm`, `timeout -s KILL 60 rm`, `env -u VAR rm` the
    value ('10', '1', 'KILL'/'60', 'VAR', '{}') becomes argv[0] and `rm` is never judged.
    Correct: know which wrapper options take an argument (nice -n, ionice -c/-n, timeout
    -s/-k, xargs -n/-I/-L/-P/-d/-s/-a/-E, env -u/-C/-S, sudo -u/-g ...) and skip it too."""
    assert "deletes_files" in _flags(cmd), cmd


@pytest.mark.parametrize("cmd", [
    "bash -lc 'rm -rf data'",
    "sh -ec 'rm -rf data'",
    "bash -xc 'rm -rf data'",
    r"""find . -name '*.tmp' -exec sh -c 'rm "$1"' _ {} \;""",
])
def test_bug_combined_shell_c_flag_is_not_recursed_into(cmd):
    """Only a standalone '-c' makes the classifier look inside `bash -c SCRIPT`. `bash -lc`
    (the usual login-shell form), `sh -ec`, `bash -xc` carry -c inside a combined flag, and
    find -exec only checks for a delete program directly, not `-exec sh -c '...'`, so the
    script is never classified. Correct: treat any short-flag cluster containing 'c' as -c
    (script = next non-option arg) and run the find -exec program through _judge_program."""
    assert "deletes_files" in _flags(cmd), cmd


@pytest.mark.parametrize("cmd", [
    'echo "$(rm -rf data)"',
    'out="$(rm -rf data)"',
    'echo "`rm -rf data`"',
])
def test_bug_command_substitution_inside_double_quotes_is_not_classified(cmd):
    """$(...) and `...` inside double quotes still EXECUTE, but the tokenizer folds the
    whole double-quoted string into one inert word, so the inner rm is never judged
    (unquoted $(rm x) is caught). Correct: classify the text of every $(...) / `...`
    found inside double quotes as its own command."""
    assert "deletes_files" in _flags(cmd), cmd


@pytest.mark.parametrize("cmd", [
    "$(which rm) -rf data",
    "`which rm` -rf data",
    "busybox rm -rf data",
])
def test_bug_indirect_program_name_hides_rm(cmd):
    """When the program is computed (`$(which rm)`, backticks) or run through a multi-call
    binary (`busybox rm`), argv[0] is '$'/'busybox' and nothing is flagged. Correct: a
    command whose program word is a substitution should be flagged conservatively (it
    cannot be judged), and busybox should be a wrapper."""
    assert _flags(cmd), cmd


@pytest.mark.parametrize("cmd", [
    "perl -e 'unlink \"model.pt\"'",
    "node -e \"require('fs').rmSync('data', {recursive: true})\"",
    "ruby -e 'File.delete(\"model.pt\")'",
    "python3 -c \"import os; os.system('rm -rf data')\"",
    "python3 -c \"import subprocess; subprocess.run(['rm', '-rf', 'data'])\"",
])
def test_bug_interpreter_one_liners_that_delete_are_not_flagged(cmd):
    """_PYTHON_DELETE_RE only knows Python's own delete APIs; a python one-liner that
    shells out to rm, or perl/node/ruby delete calls, pass as harmless. Correct: extend
    the text scan (unlink/rmSync/unlinkSync/File.delete/rm -r inside os.system/subprocess
    argument lists) or classify the quoted rm argv."""
    assert "deletes_files" in _flags(cmd), cmd


@pytest.mark.parametrize("cmd", [
    "mv results.csv /dev/null",
    "cp /dev/null train.py",
    "ln -sf /dev/null data.csv",
])
def test_bug_destroying_a_file_via_dev_null_is_not_flagged(cmd):
    """Moving a file onto /dev/null deletes it; copying/linking /dev/null over a file
    empties it -- exactly `> file`, which IS flagged. Correct: flag mv/cp/ln whose
    source or destination is /dev/null (deletes_files / overwrites_via_redirect)."""
    assert _flags(cmd), cmd


@pytest.mark.parametrize("cmd", [
    "git push origin +main",
    "git push origin :old-branch",
    "git commit --amend --no-edit",
    "git switch -f main",
    "git switch --discard-changes main",
])
def test_bug_destructive_git_variants_not_flagged(cmd):
    """Force-push via a '+refspec', remote-branch delete via ':branch', rewriting the last
    commit with --amend, and `git switch -f/--discard-changes` (which throws away
    uncommitted work exactly like `git checkout -f`, which IS flagged) are not caught.
    Correct: treat these as modifies_git_state."""
    assert "modifies_git_state" in _flags(cmd), cmd


@pytest.mark.parametrize("cmd", [
    "pkill -f train.py",
    "pkill python",
    "killall python3",
    "kill -9 -1",
    "shutdown -h now",
    "reboot",
])
def test_bug_killing_processes_and_shutdown_are_never_gated(cmd):
    """None of these is flagged, so they run without asking -- `pkill -f train.py` kills
    the user's own training run, `kill -9 -1` everything the user owns, shutdown/reboot
    the machine. The auto-mode approver's system prompt tells it to DENY commands that
    'kill processes the run didn't start', but such commands never reach it because the
    classifier does not flag them. Correct: flag kill/pkill/killall/shutdown/reboot/
    poweroff/halt (reaches_outside_workspace)."""
    assert _flags(cmd), cmd


@pytest.mark.parametrize("cmd", [
    "crontab -l",
    "systemctl status nvidia-persistenced",
    "grep -rn 'os.remove' .",
    "grep -n 'shutil.rmtree' train.py",
    "git log -S'.unlink(' --oneline",
])
def test_bug_read_only_commands_are_flagged(cmd):
    """Read-only commands that are flagged: `crontab -l`/`systemctl status` because the
    basename is in _BACKGROUND_COMMANDS whatever the subcommand; a grep FOR a Python
    delete call because _PYTHON_DELETE_RE scans the raw command text, arguments
    included. Without a TTY (or with a cautious approver) they are declined, so the agent
    cannot even search the code for where files get deleted. Correct: only flag crontab
    when it installs/edits (-e, -r, a file arg), systemctl for start/enable/restart...,
    and only scan interpreter code arguments (python -c ...) with _PYTHON_DELETE_RE."""
    assert _flags(cmd) == set(), cmd


# --------------------------------------------------------------------------- ok cases

@pytest.mark.parametrize("cmd,flag", [
    ("command rm -rf build", "deletes_files"),
    ("\\rm -rf build", "deletes_files"),
    ('"rm" -rf build', "deletes_files"),
    ("eval 'rm -rf build'", "deletes_files"),
    ("exec rm x", "deletes_files"),
    ("sudo rm x", "deletes_files"),
    ("env FOO=1 rm x", "deletes_files"),
    ("find . -exec /bin/rm {} +", "deletes_files"),
    ("bash -c 'rm -rf x'", "deletes_files"),
    ("ls&&rm x", "deletes_files"),
    ("(rm x)", "deletes_files"),
    ("diff <(rm x) y", "deletes_files"),
    ("> file", "overwrites_via_redirect"),
    (">| file", "overwrites_via_redirect"),
    ("echo x &> f", "overwrites_via_redirect"),
    ("echo x 1>f", "overwrites_via_redirect"),
    ("echo x >& f", "overwrites_via_redirect"),
    ("git push -f origin main", "modifies_git_state"),
    ("git rebase main", "modifies_git_state"),
    ("pip uninstall -y torch", "reaches_outside_workspace"),
    ("wget -O x http://h", "reaches_outside_workspace"),
    ("rsync -a --delete a/ b/", "reaches_outside_workspace"),
    ("python server.py & disown", "launches_persistent_process"),
    ("setsid python x.py", "launches_persistent_process"),
    ("echo 'python x.py' | at now", "launches_persistent_process"),
])
def test_ok_listed_forms_are_flagged(cmd, flag):
    assert flag in _flags(cmd)


@pytest.mark.parametrize("cmd", [
    "ls 2>/dev/null", "python x.py 2>&1 | tail", "ls 1>&2", "echo \"a > b\"",
    "python -c \"print(1 > 0)\"", "git checkout -b feat", "git restore --staged f",
    "git stash list", "pip list", "grep -r 'rm -rf' .", "cat <<< hi",
])
def test_ok_plain_commands_not_flagged(cmd):
    assert _flags(cmd) == set()


@pytest.mark.parametrize("cmd", [
    "echo \"$(date)\"", "$PYTHON train.py", "python3 -c \"print('rm is fine')\"", "kill -0 1234",
    "for f in *.py; do wc -l \"$f\"; done", "git push origin HEAD:main", "echo $((1 << 3))",
    "crontab -u bob -l", "$(which python) train.py", "systemctl --user list-units",
    "python3 -c \"import subprocess; subprocess.run(['ls', '-la'])\"",
])
def test_ok_fixed_forms_do_not_flag_ordinary_commands(cmd):
    assert _flags(cmd) == set(), cmd


def test_ok_heredoc_detection_ignores_bit_shifts():
    assert T.has_heredoc("python3<<EOF") and T.has_heredoc("cat <<-'EOF'")
    assert not T.has_heredoc('python3 -c "print(1 << n)"') and not T.has_heredoc("echo $((1<<n))")
    assert not T.has_heredoc("cat <<< hi")
