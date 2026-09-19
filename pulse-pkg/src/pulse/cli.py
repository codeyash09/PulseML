"""
The `pulse` command.

    pulse                          attach to a run on this machine (the console)
    pulse run [options] script.py [script args...]

`pulse run` is `python script.py` with Pulse switched on. It is not a second Pulse. It
does not ask its own questions, write its own config, or pick its own agent: the script
is started with `auto_track()` already called, so everything that happens next is what
happens when you put `auto_track()` in the script yourself -- Pulse Cloud sign-in or
login, the workspace, the agent and its key, tracking, the crash handler, fixes and
restarts. Nothing in this file decides how Pulse behaves; it only starts the script.

How the script is started, and why:

* `auto_track()` is added to the *parsed* script, in memory, and the result is compiled
  under the script's real filename and run in this process with the same `__main__`,
  `sys.argv`, `sys.path[0]` and working directory that `python script.py` would give it.
  Nothing is written next to your code, so tracebacks show your file and your line
  numbers, `__file__` is your file, and when Pulse fixes something it edits the file you
  wrote, not a copy of it.
* The working directory is the directory you ran `pulse` from, exactly like
  `python path/to/script.py`. `--cwd DIR` runs the script somewhere else.
* Running in this process (not a subprocess) is what lets Pulse's Ctrl+C handling, its
  prompts and its crash hook behave as they do for a script that calls `auto_track()`.

A syntax error is the one thing normal Pulse cannot handle, because Python refuses to
run a file that does not parse and `auto_track()` never gets called. `pulse run` covers
that gap by handing the error to the same machinery a runtime crash goes through:
normal Pulse setup first (so you sign in, pick a workspace and an agent as usual), then
the agent diagnoses and fixes the file, then Pulse restarts the script to apply the fix,
and the restarted run is tracked like any other.
"""
import ast
import builtins
import os
import sys
import tokenize
import traceback
import types

USAGE = """\
Pulse - a live ML training debugger.

  pulse                          attach to the run on this machine (interactive)
  pulse watch [n|id|name]        attach to a particular run
  pulse sessions                 list the runs Pulse knows about
  pulse run [options] <script.py> [script args]
                                 run a script under Pulse, like `python script.py`

options for `pulse run` (before the script; everything after it is the script's):
  --stream                       stream to a separate brain instead of tracking in-process
  --cwd DIR                      run the script in DIR (default: the directory you ran
                                 `pulse` from, as with `python path/to/script.py`)

options for `pulse` / `pulse watch`:
  --model <id>                   the agent to think with (or set PULSE_MODEL)

`pulse run` sets Pulse up exactly as auto_track() does -- sign-in, workspace, agent --
and if the script has a syntax error, Pulse fixes it and restarts the run to apply it.
"""

_RUN_MODE = "cli"          # what the injected auto_track() is given; "stream" for --stream


# ---------------------------------------------------------------------------------------
# Adding auto_track() to a parsed script
# ---------------------------------------------------------------------------------------

class PulseASTInjector(ast.NodeTransformer):
    """Add one `auto_track(mode=...)` call to a parsed module.

    `mode` is what the call is given: "cli" is normal Pulse, "stream" starts the light
    monitor instead. The call is written as `__import__("pulse").auto_track(...)`, so it
    binds no name in the script's namespace and cannot collide with anything the script
    defines, and it carries the line number of the statement it sits in front of, so no
    other line number in the script moves.

    Where it goes: first thing inside the script's own top-level `if __name__ ==
    "__main__":` block if it has one, otherwise right after the last top-level import.
    """

    def __init__(self, mode="cli"):
        self.mode = mode

    def _make_auto_track_call(self, anchor):
        call = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Call(
                        func=ast.Name(id="__import__", ctx=ast.Load()),
                        args=[ast.Constant(value="pulse")],
                        keywords=[],
                    ),
                    attr="auto_track",
                    ctx=ast.Load(),
                ),
                args=[],
                keywords=[ast.keyword(arg="mode", value=ast.Constant(value=self.mode))],
            )
        )
        ast.copy_location(call, anchor)
        ast.fix_missing_locations(call)
        return call

    @staticmethod
    def _find_import_insert_idx(body):
        """After the last top-level import; failing that, after a module docstring."""
        idx = 0
        for i, child in enumerate(body):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                idx = i + 1
        if idx == 0 and body:
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                idx = 1
        return idx

    @staticmethod
    def _is_name_main_guard(test):
        if not (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
        ):
            return False
        left, right = test.left, test.comparators[0]

        def is_dunder_name(node):
            return isinstance(node, ast.Name) and node.id == "__name__"

        def is_main_string(node):
            return isinstance(node, ast.Constant) and node.value == "__main__"

        return (is_dunder_name(left) and is_main_string(right)) or (
            is_dunder_name(right) and is_main_string(left)
        )

    def visit_Module(self, node):
        if not node.body:
            return node                     # an empty script has nothing to track

        # Only the script's own top-level guard. One nested in a function is not the
        # entry point, and a second top-level guard must not start tracking twice.
        guard = next(
            (s for s in node.body
             if isinstance(s, ast.If) and self._is_name_main_guard(s.test)),
            None,
        )
        if guard is not None:
            guard.body.insert(0, self._make_auto_track_call(guard.body[0]))
            return node

        idx = self._find_import_insert_idx(node.body)
        anchor = node.body[idx] if idx < len(node.body) else node.body[-1]
        node.body.insert(idx, self._make_auto_track_call(anchor))
        return node


def _already_uses_pulse(tree):
    """A script that already imports Pulse is running it its own way; leave it alone."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pulse" or alias.name.startswith("pulse."):
                    return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and (node.module == "pulse" or node.module.startswith("pulse.")):
                return True
    return False


# ---------------------------------------------------------------------------------------
# Becoming `python script.py`
# ---------------------------------------------------------------------------------------

def _set_process_view(script_path, script_args):
    """Make this process look like `python script.py args...` to anything that asks.

    Pulse asks: a restart re-runs `[python, script_path] + sys.argv[1:]`. The script
    asks: `sys.argv`, and imports of its sibling modules through `sys.path[0]`.
    """
    # `python link.py` puts the directory of the file the link points to first on the path,
    # not the directory of the link, so the script's sibling modules are found beside the
    # real file. (`__file__` and sys.argv[0] stay as typed, also as under python.)
    script_dir = os.path.dirname(os.path.realpath(script_path))
    sys.argv = [script_path] + list(script_args)
    if not sys.path or sys.path[0] != script_dir:
        sys.path.insert(0, script_dir)


def _exit_code(exc):
    """The exit status `python` would use for a SystemExit."""
    code = exc.code
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    print(code, file=sys.stderr)
    return 1


def _execute(code, script_path):
    """Run compiled script code as `__main__`. Returns the exit status.

    An exception that escapes is handed to `sys.excepthook`, because that is what the
    interpreter would do for an uncaught one -- and it is where Pulse's crash handler
    lives. The frames of this function are cut off first so the hook sees a traceback
    that starts in the script, as it would under `python script.py`.
    """
    main_module = types.ModuleType("__main__")
    main_module.__file__ = script_path
    main_module.__builtins__ = builtins
    sys.modules["__main__"] = main_module

    try:
        exec(code, main_module.__dict__)
    except SystemExit as exc:
        return _exit_code(exc)
    except BaseException:
        exc_type, exc_value, tb = sys.exc_info()
        while tb is not None and tb.tb_frame.f_code is _execute.__code__:
            tb = tb.tb_next
        exc_value = exc_value.with_traceback(tb)
        try:
            sys.excepthook(exc_type, exc_value, tb)
        except SystemExit as hook_exit:
            # The interpreter honours a SystemExit raised from inside the hook, and Pulse
            # uses one: a fix that restarted the script and the restart succeeded.
            return _exit_code(hook_exit)
        except BaseException:
            print("Error in sys.excepthook:", file=sys.stderr)
            traceback.print_exc()
            print("\nOriginal exception was:", file=sys.stderr)
            traceback.print_exception(exc_type, exc_value, tb)
        return 130 if issubclass(exc_type, KeyboardInterrupt) else 1
    finally:
        # The script is over, and so is anything Pulse's tracer had to watch. Left
        # installed it is still called for every frame during interpreter teardown, when
        # a late __del__ (multiprocessing connections, say) runs after `os.path` has been
        # cleared and the tracer's first line raises "'NoneType' has no attribute
        # 'normcase'". After the crash hook above, so a crash is still handled with it.
        sys.settrace(None)
    return 0


# ---------------------------------------------------------------------------------------
# Restarting under `pulse run`
# ---------------------------------------------------------------------------------------

# After Pulse fixes something it restarts the script to apply the fix. Normally that is
# `python script.py`, and it is tracked again because the script itself calls
# auto_track(). Under `pulse run` the file on disk has no auto_track() in it -- it is
# added in memory -- so a plain re-run would restart the script with nobody watching.
# Pulse asks this hook how to restart, and the answer is `pulse run` again.
_CHILD_BOOT = (
    "import sys; sys.path.insert(0, {root!r}); "
    "from pulse.cli import main; sys.exit(main({head!r} + sys.argv[1:]))"
)


def _restart_argv(python_exe, script_path, script_args):
    # -c with an explicit path rather than `-m pulse`: the restart has to find this same
    # package whatever directory the run is in, including under --cwd, and whether Pulse
    # is installed or being run from a checkout.
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    head = ["run"] + (["--stream"] if _RUN_MODE == "stream" else [])
    return [python_exe, "-c", _CHILD_BOOT.format(root=root, head=head),
            script_path] + list(script_args)


# ---------------------------------------------------------------------------------------
# Syntax errors
# ---------------------------------------------------------------------------------------

def _make_syntax_repair_cli():
    """PulseCLI, minus two behaviours that are wrong for a script that does not parse.

    Everything else is inherited untouched, including setup, the agent pipeline, fix
    logging (/log, /revert) and `_restart_process`.

    * The empirical check runs a probe of the training loop and reverts the fix if the
      loss does not fall. A file that has not run yet has no loss to measure, and the
      syntax error is already known to be gone: the automatic lint gate compiles the file
      before it is written.
    * The git autostash stashes the working tree before a fix is written -- including,
      here, the very file being repaired, which is uncommitted precisely because the
      person is editing it.
    """
    from .pulse_cli import PulseCLI

    class _SyntaxRepairCLI(PulseCLI):
        def _verify_fix_empirically(self, fix, diagnosis):
            return fix, None, "syntax fix: the file compiles, and there is no loss to measure yet"

        def _git_autostash(self):
            return None

    return _SyntaxRepairCLI()


def _repair_and_restart(script_path, source, exc):
    """Fix a syntax error the way Pulse fixes a crash, and restart to apply it.

    Returns the exit status. On success it does not get that far: the restarted run
    finishes and Pulse exits with its status from inside the crash handler.
    """
    from .pulse import _install_cli_excepthook

    name = os.path.basename(script_path)
    print(f"[Pulse] {name} has a syntax error (line {exc.lineno}); "
          "Pulse can't track a script Python won't run.")

    cli = _make_syntax_repair_cli()
    cli.set_code_text(source, script_path=script_path)
    cli.print_banner()
    try:
        # The same setup auto_track() does: sign in or log in, workspace, agent.
        cli.interactive_setup()
    except (EOFError, KeyboardInterrupt):
        print("\n[Pulse] Setup cancelled -- the syntax error was not fixed.")
        traceback.print_exception(type(exc), exc, None)
        return 1

    # From here it is the normal crash path. The hook prints the error, records the
    # crash, asks the agent (retrying a rate limit in the foreground), and when a fix
    # lands Pulse restarts the script itself -- through _restart_argv, so the restarted
    # run is tracked. tb=None: there is no script frame to show, only the error, and the
    # frames of this function are not the user's code.
    _install_cli_excepthook(cli)
    try:
        sys.excepthook(type(exc), exc, None)
    except SystemExit as restart_exit:
        return _exit_code(restart_exit)

    # The hook returned: no agent, a declined prompt, no fix, or a restart that gave up.
    print(f"[Pulse] {name} still has a syntax error and was not started.")
    return 1


# ---------------------------------------------------------------------------------------
# pulse run
# ---------------------------------------------------------------------------------------

class _UsageError(Exception):
    pass


def _parse_run_args(args):
    """(stream, cwd, script, script_args). Options end at the first non-option: what
    follows is the script and its own arguments, which are never inspected."""
    stream, cwd, i = False, None, 0
    while i < len(args):
        arg = args[i]
        if arg == "--":
            i += 1
            break
        if not arg.startswith("-"):
            break
        if arg == "--stream":
            stream = True
        elif arg == "--cwd":
            if i + 1 >= len(args):
                raise _UsageError("--cwd needs a directory")
            cwd = args[i + 1]
            i += 1
        elif arg.startswith("--cwd="):
            cwd = arg.split("=", 1)[1]
        elif arg in ("-h", "--help"):
            raise _UsageError("")
        else:
            raise _UsageError(f"unknown option {arg!r} (options for the script go after its name)")
        i += 1
    if i >= len(args):
        raise _UsageError("no script given")
    return stream, cwd, args[i], args[i + 1:]


def run_script(script, script_args, stream=False, cwd=None):
    """Run `script` under Pulse. Returns the exit status."""
    global _RUN_MODE
    from . import pulse_cli

    _RUN_MODE = "stream" if stream else "cli"

    # Resolved against where the command was typed, before --cwd can move us.
    script_path = os.path.abspath(script)
    if not os.path.isfile(script_path):
        print(f"Error: training script '{script}' not found.")
        return 1

    if cwd is not None:
        cwd = os.path.abspath(cwd)
        if not os.path.isdir(cwd):
            print(f"Error: --cwd '{cwd}' is not a directory.")
            return 1
        os.chdir(cwd)

    try:
        with tokenize.open(script_path) as handle:      # honours a coding cookie / BOM
            source = handle.read()
    except (SyntaxError, UnicodeDecodeError, LookupError) as exc:
        print(f"Error: cannot decode '{script_path}': {exc}")
        return 1

    pulse_cli._RESTART_ARGV_HOOK = _restart_argv
    _set_process_view(script_path, script_args)

    # A run that Pulse itself restarted reports back to the process that restarted it,
    # which owns the retry loop and feeds the output to the agent. It must not start a
    # repair of its own.
    restarted_by_pulse = os.environ.get(pulse_cli._RESTART_CHILD_ENV) == "1"

    try:
        tree = ast.parse(source, filename=script_path)
        # Some syntax errors (`return` outside a function...) are only found by compile().
        compile(tree, script_path, "exec", dont_inherit=True)
    except SyntaxError as exc:
        # The frames that led here are this file's and the parser's, not the user's. Python
        # shows a syntax error as just the offending line, so show that and only that.
        exc.__traceback__ = None
        if restarted_by_pulse:
            traceback.print_exception(type(exc), exc, None)
            return 1
        return _repair_and_restart(script_path, source, exc)
    except ValueError as exc:                           # e.g. null bytes, on older Pythons
        print(f"Error: cannot parse '{script_path}': {exc}")
        return 1

    if _already_uses_pulse(tree):
        if not restarted_by_pulse:
            print(f"[Pulse] {os.path.basename(script_path)} already uses Pulse; running it as it is.")
    else:
        PulseASTInjector(mode=_RUN_MODE).visit(tree)
        if not restarted_by_pulse:
            print(f"[Pulse] Running {os.path.basename(script_path)} under Pulse.")

    # dont_inherit: nothing from this file's own __future__ imports may leak into the script.
    code = compile(tree, script_path, "exec", dont_inherit=True)
    return _execute(code, script_path)


def main(argv=None):
    """`pulse ...`. Returns the exit status; the console script and `python -m pulse`
    both pass it to sys.exit."""
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0
    if argv and argv[0] in ("-V", "--version"):
        from . import __version__
        print(f"pulse {__version__}")
        return 0

    if argv and argv[0] == "run":
        try:
            stream, cwd, script, script_args = _parse_run_args(argv[1:])
        except _UsageError as problem:
            if str(problem):
                print(f"pulse run: {problem}\n")
            print(USAGE)
            return 1 if str(problem) else 0
        try:
            return run_script(script, script_args, stream=stream, cwd=cwd)
        except KeyboardInterrupt:
            print("\n[Pulse] Interrupted.")
            return 130

    # Bare `pulse`, watch, attach, console, sessions -- and `pulse --model X` -- are the
    # console. Anything else that is not a command is a mistake worth showing usage for.
    if not argv or argv[0] in ("watch", "attach", "console", "sessions") or argv[0].startswith("-"):
        from .pulse_console import main as console_main
        return console_main(argv)

    print(USAGE)
    return 1


if __name__ == "__main__":
    sys.exit(main())
