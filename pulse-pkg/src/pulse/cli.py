# pulse/cli.py
import sys
import os
import ast
import json
import getpass
import tempfile
import subprocess


class PulseASTInjector(ast.NodeTransformer):
    """Inject Pulse auto-tracking into the user's training script."""

    def __init__(self):
        self.has_main_block = False

    @staticmethod
    def _make_auto_track_call():
        return ast.Expr(
            value=ast.Call(
                func=ast.Name(id="auto_track", ctx=ast.Load()),
                args=[],
                keywords=[
                    ast.keyword(
                        arg="mode",
                        value=ast.Constant(
                            value="cli",
                        ),
                    )
                ],
            )
        )

    @staticmethod
    def _find_import_insert_idx(body):
        """
        Insert after the last top-level import.

        If there are no imports, preserve a module docstring by inserting
        after it rather than before it.
        """
        idx = 0

        for i, child in enumerate(body):
            if isinstance(
                child,
                (
                    ast.Import,
                    ast.ImportFrom,
                ),
            ):
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

        left = test.left
        right = test.comparators[0]

        def is_dunder_name(node):
            return (
                isinstance(node, ast.Name)
                and node.id == "__name__"
            )

        def is_main_string(node):
            return (
                isinstance(node, ast.Constant)
                and node.value == "__main__"
            )

        return (
            (
                is_dunder_name(left)
                and is_main_string(right)
            )
            or
            (
                is_dunder_name(right)
                and is_main_string(left)
            )
        )

    def visit_If(self, node):
        self.generic_visit(node)

        if self._is_name_main_guard(node.test):
            self.has_main_block = True

            # auto_track() must be the first thing executed inside
            # the training script's __main__ block.
            node.body.insert(
                0,
                self._make_auto_track_call(),
            )

        return node

    def visit_Module(self, node):
        # Visit children first so we can detect a __main__ guard.
        self.generic_visit(node)

        insert_idx = self._find_import_insert_idx(
            node.body
        )

        pulse_import = ast.ImportFrom(
            module="pulse",
            names=[
                ast.alias(
                    name="auto_track",
                    asname=None,
                )
            ],
            level=0,
        )

        # Inject the import into the TRAINING SCRIPT.
        node.body.insert(
            insert_idx,
            pulse_import,
        )

        # If there is no __main__ guard, the training script executes
        # top-to-bottom, so start tracking immediately after imports.
        if not self.has_main_block:
            node.body.insert(
                insert_idx + 1,
                self._make_auto_track_call(),
            )

        return node


def _already_uses_pulse(tree):
    """
    Do not instrument a script that already imports Pulse.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if (
                    alias.name == "pulse"
                    or alias.name.startswith("pulse.")
                ):
                    return True

        elif isinstance(node, ast.ImportFrom):
            if (
                node.module
                and (
                    node.module == "pulse"
                    or node.module.startswith("pulse.")
                )
            ):
                return True

    return False


def _write_instrumented_script(tree, script_path):
    """
    Write the transformed training script to a temporary .py file in the
    SAME DIRECTORY as the original training script.

    Keeping it beside the original preserves normal Python import behavior
    for the training project.
    """
    fd, temp_path = tempfile.mkstemp(
        prefix=".pulse_instrumented_",
        suffix=".py",
        dir=os.path.dirname(script_path),
        text=True,
    )

    try:
        os.close(fd)

        ast.fix_missing_locations(tree)

        # Validate the transformed tree before writing it.
        compile(
            tree,
            filename=script_path,
            mode="exec",
        )

        # Convert the transformed AST back into source.
        #
        # This path is ONLY used when the original source successfully
        # parsed. Syntax-error files never reach this function.
        instrumented_source = ast.unparse(tree)

        with open(
            temp_path,
            "w",
            encoding="utf-8",
            newline="",
        ) as f:
            f.write(instrumented_source)
            f.write("\n")

        return temp_path

    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            pass

        raise


def _write_raw_instrumented_script(
    source_code,
    script_path,
):
    """
    Fallback instrumentation for a training script that contains
    a syntax error.

    IMPORTANT:
        This function intentionally does NOT parse or compile the
        user's source code.

    Pulse is placed at the very beginning of the temporary script
    so auto_track() starts before Python reaches the user's broken
    training code.

    The original training script is never modified.
    """
    fd, temp_path = tempfile.mkstemp(
        prefix=".pulse_instrumented_",
        suffix=".py",
        dir=os.path.dirname(script_path),
        text=True,
    )

    try:
        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
            newline="",
        ) as f:
            # Pulse MUST come first.
            f.write(
                "from pulse import auto_track\n"
            )
            f.write(
                'auto_track(mode="cli")\n\n'
            )

            # Then preserve the user's source exactly.
            f.write(source_code)

        return temp_path

    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            pass

        raise


def _run_training_script(
    script_path,
    script_dir,
    args,
):
    """
    Execute a training script as a normal Python subprocess.

    The training script itself owns the auto_track() call because
    instrumentation has already been injected into the temporary
    training script.
    """
    command = [
        sys.executable,
        script_path,
        *args,
    ]

    env = os.environ.copy()

    # Make the original training directory importable exactly as it
    # normally would be when executing:
    #
    #     python train.py
    #
    existing_pythonpath = env.get(
        "PYTHONPATH",
        "",
    )

    if existing_pythonpath:
        env["PYTHONPATH"] = (
            script_dir
            + os.pathsep
            + existing_pythonpath
        )
    else:
        env["PYTHONPATH"] = script_dir

    return subprocess.run(
        command,
        cwd=script_dir,
        env=env,
    )


def _bootstrap_pulse_config(script_path):
    """
    Interactively create pulse_config.json when Pulse needs an AI
    provider but no usable configuration exists.

    The configuration is stored beside the user's training script.
    The training script itself is never modified.
    """
    print()
    print("=" * 64)
    print("Pulse first-time setup")
    print("=" * 64)
    print()

    print(
        "Pulse needs an AI provider to repair this Python syntax error."
    )
    print(
        "Your API key will be stored in pulse_config.json in this"
    )
    print(
        "training project's directory."
    )
    print()

    # ------------------------------------------------------------
    # Terms of Service
    # ------------------------------------------------------------

    print(
        "Before continuing, you must agree to Pulse's Terms of Service."
    )
    print()

    try:
        tos_response = input(
            "Do you agree to the Pulse Terms of Service? [y/N]: "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        print(
            "[Pulse] Setup cancelled."
        )
        return False

    if tos_response not in (
        "y",
        "yes",
    ):
        print()
        print(
            "[Pulse] Terms of Service were not accepted."
        )
        print(
            "[Pulse] Cannot continue without accepting the Terms of Service."
        )
        return False

    # ------------------------------------------------------------
    # Provider
    # ------------------------------------------------------------

    print()
    print(
        "Supported provider examples:"
    )
    print(
        "  gemini"
    )
    print(
        "  openai"
    )
    print(
        "  anthropic"
    )
    print(
        "  openrouter"
    )
    print(
        "  local"
    )
    print()

    try:
        provider = input(
            "AI provider: "
        ).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        print(
            "[Pulse] Setup cancelled."
        )
        return False

    if not provider:
        print(
            "[Pulse] No provider selected."
        )
        return False

    # ------------------------------------------------------------
    # API key
    # ------------------------------------------------------------

    api_key = ""

    if provider.lower() not in (
        "local",
        "ollama",
        "lmstudio",
        "llama.cpp",
    ):
        print()

        try:
            api_key = getpass.getpass(
                "API key: "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print(
                "[Pulse] API key entry cancelled."
            )
            return False

        if not api_key:
            print(
                "[Pulse] No API key entered."
            )
            return False

    # ------------------------------------------------------------
    # Write configuration
    # ------------------------------------------------------------

    config_path = os.path.join(
        os.path.dirname(
            os.path.abspath(script_path)
        ),
        "pulse_config.json",
    )

    config = {
        "agent": {
            "provider": provider,
        },
        "api_key": api_key,
        "autofix": True,
        "tos_accepted": True,
    }

    try:
        with open(
            config_path,
            "w",
            encoding="utf-8",
        ) as config_file:
            json.dump(
                config,
                config_file,
                indent=2,
            )
            config_file.write("\n")

    except OSError as exc:
        print(
            f"[Pulse] Could not create "
            f"'{config_path}': {exc}"
        )
        return False

    print()
    print(
        f"[Pulse] Configuration saved to "
        f"{config_path}"
    )
    print(
        "[Pulse] This setup will be reused on future runs."
    )
    print()

    return True


def main():
    if len(sys.argv) < 3 or sys.argv[1] != "run":
        print(
            "Usage: pulse run <script.py> [args]"
        )
        sys.exit(1)

    script_path = sys.argv[2]

    if not os.path.exists(script_path):
        print(
            f"Error: Training script "
            f"'{script_path}' not found."
        )
        sys.exit(1)

    script_path = os.path.abspath(
        script_path
    )

    if not os.path.isfile(script_path):
        print(
            f"Error: Training script "
            f"'{script_path}' is not a file."
        )
        sys.exit(1)

    script_dir = os.path.dirname(
        script_path
    )

    with open(
        script_path,
        "r",
        encoding="utf-8",
    ) as f:
        source_code = f.read()

    temp_path = None

    try:
        # ============================================================
        # PATH 1: NORMAL / VALID PYTHON
        # ============================================================
        #
        # If the training script is valid Python, use the deterministic
        # AST instrumentation path.
        #
        # This preserves the existing behavior.
        #
        try:
            tree = ast.parse(
                source_code,
                filename=script_path,
            )

        except SyntaxError as exc:
            # ========================================================
            # PATH 2: SYNTAX ERROR REPAIR
            # ========================================================
            #
            # IMPORTANT:
            #
            # Python parses the ENTIRE file before executing line 1.
            # Therefore putting auto_track() before broken Python does
            # NOT work.
            #
            # Instead:
            #
            #   broken source
            #       ->
            #   temporary repair copy
            #       ->
            #   existing PulseCLI repair machinery
            #       ->
            #   valid source
            #       ->
            #   AST instrumentation
            #       ->
            #   training process
            #
            # The original training script is NEVER modified.
            # ========================================================

            print(
                f"[Pulse] Syntax error detected in "
                f"{os.path.basename(script_path)} "
                f"at line {exc.lineno}."
            )

            # --------------------------------------------------------
            # Create a temporary working copy.
            # --------------------------------------------------------

            fd, repair_path = tempfile.mkstemp(
                prefix=".pulse_repair_",
                suffix=".py",
                dir=script_dir,
                text=True,
            )

            try:
                with os.fdopen(
                    fd,
                    "w",
                    encoding="utf-8",
                    newline="",
                ) as f:
                    f.write(source_code)

                # ----------------------------------------------------
                # Use Pulse's EXISTING repair engine.
                #
                # Import locally so valid-script behavior and startup
                # remain unchanged.
                # ----------------------------------------------------

                from pulse.pulse_cli import PulseCLI

                pulse = PulseCLI()

                # _apply_code_fix() writes to self.script_path.
                # Therefore point it at the temporary copy.
                pulse.set_code_text(
                    source_code,
                    script_path=repair_path,
                )

                # ----------------------------------------------------
                # First-run setup.
                #
                # If no usable AI agent is configured, interactively
                # create pulse_config.json and then reload it.
                # ----------------------------------------------------

                if not getattr(
                    pulse,
                    "agent_provider",
                    None,
                ) or not getattr(
                    pulse,
                    "agent_key",
                    None,
                ):
                    if not _bootstrap_pulse_config(
                        script_path
                    ):
                        raise RuntimeError(
                            "Pulse setup was not completed."
                        )

                    pulse._load_config()

                # Initialize the configured provider using Pulse's
                # existing provider-selection machinery.
                if not pulse._select_agent_provider_and_key(
                    initial=True
                ):
                    raise RuntimeError(
                        "Pulse could not initialize the configured "
                        "AI provider."
                    )

                error_text = (
                    f"SyntaxError: {exc.msg} "
                    f"(line {exc.lineno}"
                )

                if exc.offset is not None:
                    error_text += (
                        f", column {exc.offset}"
                    )

                error_text += ")"

                question = (
                    "The training script cannot start because Python "
                    "reports this syntax error:\n\n"
                    f"{error_text}\n\n"
                    "Diagnose and fix ONLY this syntax error.\n\n"
                    "IMPORTANT:\n"
                    "- Do NOT rewrite the entire file.\n"
                    "- Do NOT return the entire script.\n"
                    "- Do NOT change unrelated code.\n"
                    "- Make the smallest possible surgical edit.\n"
                    "- Return the normal Pulse old/new code-fix format."
                )

                print(
                    "[Pulse] Sending the syntax error through "
                    "the existing repair pipeline..."
                )

                result = pulse.ask_agent(
                    question,
                    include_code=True,
                )

                if not getattr(
                    pulse,
                    "_fix_applied_this_turn",
                    False,
                ):
                    raise RuntimeError(
                        "Pulse's existing repair pipeline did not "
                        "apply a syntax-error fix.\n\n"
                        f"Agent response:\n{result}"
                    )

                # ----------------------------------------------------
                # Read the ACTUAL repaired temporary file.
                # ----------------------------------------------------

                with open(
                    repair_path,
                    "r",
                    encoding="utf-8",
                ) as f:
                    repaired_source = f.read()

                # ----------------------------------------------------
                # Hard validation: the agent's output must really
                # produce syntactically-valid Python.
                # ----------------------------------------------------

                compile(
                    repaired_source,
                    filename=repair_path,
                    mode="exec",
                )

                print(
                    "[Pulse] Syntax error repaired successfully."
                )

                # ----------------------------------------------------
                # Parse the repaired source.
                # ----------------------------------------------------

                repaired_tree = ast.parse(
                    repaired_source,
                    filename=repair_path,
                )

                # ----------------------------------------------------
                # Now use the SAME deterministic AST instrumentation
                # used for normal valid Python.
                # ----------------------------------------------------

                transformer = PulseASTInjector()

                modified_tree = transformer.visit(
                    repaired_tree
                )

                ast.fix_missing_locations(
                    modified_tree
                )

                print(
                    f"[Pulse] Automatically instrumenting "
                    f"{script_path}"
                )

                instrumented_path = _write_instrumented_script(
                    modified_tree,
                    repair_path,
                )

                try:
                    print(
                        f"[Pulse] Starting "
                        f"{os.path.basename(script_path)}..."
                    )

                    result = _run_training_script(
                        instrumented_path,
                        script_dir,
                        sys.argv[3:],
                    )

                finally:
                    try:
                        os.unlink(
                            instrumented_path
                        )
                    except OSError:
                        pass

                if result.returncode != 0:
                    print(
                        f"[Pulse] Training script exited "
                        f"with code {result.returncode}"
                    )

                sys.exit(
                    result.returncode
                )

            finally:
                try:
                    os.unlink(
                        repair_path
                    )
                except OSError:
                    pass

        # ============================================================
        # VALID PYTHON: CHECK WHETHER PULSE IS ALREADY IMPORTED
        # ============================================================

        if _already_uses_pulse(tree):
            print(
                f"[Pulse] {script_path} already imports Pulse. "
                f"Running it normally."
            )

            result = _run_training_script(
                script_path,
                script_dir,
                sys.argv[3:],
            )

            sys.exit(
                result.returncode
            )

        # ============================================================
        # VALID PYTHON: AST INSTRUMENTATION
        # ============================================================

        transformer = PulseASTInjector()

        modified_tree = transformer.visit(
            tree
        )

        ast.fix_missing_locations(
            modified_tree
        )

        print(
            f"[Pulse] Automatically instrumenting "
            f"{script_path}"
        )

        temp_path = _write_instrumented_script(
            modified_tree,
            script_path,
        )

        print(
            f"[Pulse] Starting "
            f"{os.path.basename(script_path)}..."
        )

        result = _run_training_script(
            temp_path,
            script_dir,
            sys.argv[3:],
        )

        if result.returncode != 0:
            print(
                f"[Pulse] Training script exited "
                f"with code {result.returncode}"
            )

        sys.exit(
            result.returncode
        )

    except KeyboardInterrupt:
        print(
            "\n[Pulse] Interrupted."
        )
        sys.exit(130)

    except Exception as exc:
        print(
            f"\n[Pulse Error] Failed to run "
            f"'{script_path}': {exc}"
        )
        raise

    finally:
        if temp_path:
            try:
                os.unlink(
                    temp_path
                )
            except OSError:
                pass


if __name__ == "__main__":
    main()
