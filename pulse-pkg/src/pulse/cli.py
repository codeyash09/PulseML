# pulse/cli.py
import sys
import os
import ast
import json
import getpass
import tempfile
import subprocess
import time


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


# How long to wait before asking the provider again when a repair request comes back
# rate-limited. The script has not started yet, so there is nothing to keep alive and
# nothing racing: waiting is free, and giving up leaves the user with a broken file.
_REPAIR_RETRY_DELAYS = (5, 15, 45)


def _bootstrap_pulse_config(script_path):
    """
    Interactively create pulse_config.json when Pulse needs an AI
    provider but no usable configuration exists.

    IMPORTANT:
        If pulse_config.json already exists beside the training script,
        this function DOES NOT prompt, modify, or overwrite it.

    The provider list comes directly from Pulse's real PROVIDERS registry
    in pulse_cli.py.

    The configuration format matches PulseCLI._load_config(), which
    expects top-level keys such as:

        {
            "agent": "Google AI Studio (Gemini 3.5 Flash-Lite)",
            "api_key": "...",
            "autofix": true,
            "tos_accepted": true
        }

    The training script itself is never modified.
    """
    from pulse.pulse_cli import PROVIDERS

    config_path = os.path.join(
        os.path.dirname(
            os.path.abspath(script_path)
        ),
        "pulse_config.json",
    )

    # ------------------------------------------------------------
    # EXISTING CONFIGURATION
    #
    # This check MUST happen before any setup prompts.
    # ------------------------------------------------------------

    if os.path.isfile(config_path):
        print()
        print(
            f"[Pulse] Existing configuration found: "
            f"{config_path}"
        )
        print(
            "[Pulse] Reusing existing configuration."
        )
        print()

        return True

    # ------------------------------------------------------------
    # FIRST-TIME SETUP
    # ------------------------------------------------------------

    print()
    print("=" * 64)
    print("Pulse first-time setup")
    print("=" * 64)
    print()

    print(
        "Pulse needs an AI provider to repair this Python syntax error."
    )
    print(
        "Your configuration will be stored in pulse_config.json in "
        "this training project's directory."
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
    #
    # Use Pulse's ACTUAL provider registry.
    # Do not maintain a second provider list here.
    # ------------------------------------------------------------

    print()
    print(
        "Available AI providers:"
    )
    print()

    provider_names = list(
        PROVIDERS.keys()
    )

    for index, provider_name in enumerate(
        provider_names,
        start=1,
    ):
        info = PROVIDERS[
            provider_name
        ]

        suffix = ""

        if info.get("local"):
            suffix = " [local]"

        elif info.get("openrouter"):
            suffix = " [OpenRouter]"

        elif info.get("custom"):
            suffix = " [custom]"

        print(
            f"  {index}. {provider_name}{suffix}"
        )

    print()

    try:
        provider_choice = input(
            "AI provider: "
        ).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        print(
            "[Pulse] Setup cancelled."
        )
        return False

    try:
        provider_index = int(
            provider_choice
        ) - 1
    except ValueError:
        print()
        print(
            "[Pulse] Invalid provider selection."
        )
        return False

    if (
        provider_index < 0
        or provider_index >= len(provider_names)
    ):
        print()
        print(
            "[Pulse] Invalid provider selection."
        )
        return False

    provider = provider_names[
        provider_index
    ]

    provider_info = PROVIDERS[
        provider
    ]

    # ------------------------------------------------------------
    # Build the config using Pulse's existing config semantics.
    # ------------------------------------------------------------

    config = {
        "agent": provider,
        "autofix": True,
        "tos_accepted": True,
    }

    # ------------------------------------------------------------
    # LOCAL PROVIDER
    # ------------------------------------------------------------

    if provider_info.get("local"):
        print()

        model_hint = provider_info.get(
            "model_hint",
            "model name",
        )

        default_api_base = provider_info.get(
            "default_api_base",
            "",
        )

        try:
            model_name = input(
                f"Model ({model_hint}): "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print(
                "[Pulse] Setup cancelled."
            )
            return False

        if not model_name:
            print(
                "[Pulse] No local model name entered."
            )
            return False

        try:
            api_base = input(
                f"API base [{default_api_base}]: "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print(
                "[Pulse] Setup cancelled."
            )
            return False

        if not api_base:
            api_base = default_api_base

        config["model"] = model_name
        config["api_base"] = api_base

    # ------------------------------------------------------------
    # OPENROUTER
    # ------------------------------------------------------------

    elif provider_info.get("openrouter"):
        print()

        try:
            model_name = input(
                "OpenRouter model slug "
                "(e.g. deepseek/deepseek-v4-flash): "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print(
                "[Pulse] Setup cancelled."
            )
            return False

        if not model_name:
            print(
                "[Pulse] No OpenRouter model entered."
            )
            return False

        config["model"] = model_name

        try:
            api_key = getpass.getpass(
                "OpenRouter API key: "
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

        config["api_key"] = api_key

    # ------------------------------------------------------------
    # CUSTOM PROVIDER
    # ------------------------------------------------------------

    elif provider_info.get("custom"):
        print()

        try:
            provider_model = input(
                "Provider/model: "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print(
                "[Pulse] Setup cancelled."
            )
            return False

        if not provider_model:
            print(
                "[Pulse] No provider/model entered."
            )
            return False

        # Pulse's custom-provider branch receives the provider/model
        # through the agent configuration.
        config["agent"] = provider_model

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

        if api_key:
            config["api_key"] = api_key

    # ------------------------------------------------------------
    # STANDARD CLOUD PROVIDER
    # ------------------------------------------------------------

    else:
        env_key = provider_info.get(
            "env_key"
        )

        if not env_key:
            print(
                f"[Pulse] Provider '{provider}' does not define "
                "an API-key environment variable."
            )
            return False

        print()

        try:
            api_key = getpass.getpass(
                f"{env_key}: "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print(
                "[Pulse] API key entry cancelled."
            )
            return False

        if not api_key:
            print(
                f"[Pulse] No {env_key} entered."
            )
            return False

        config["api_key"] = api_key

    # ------------------------------------------------------------
    # WRITE CONFIGURATION
    #
    # Use exclusive creation so an existing file can NEVER be
    # overwritten accidentally, even in a race between processes.
    # ------------------------------------------------------------

    try:
        with open(
            config_path,
            "x",
            encoding="utf-8",
        ) as config_file:
            json.dump(
                config,
                config_file,
                indent=2,
            )
            config_file.write("\n")

    except FileExistsError:
        # Another process created the configuration after our
        # initial existence check. Never overwrite it.
        print()
        print(
            f"[Pulse] Configuration already exists: "
            f"{config_path}"
        )
        print(
            "[Pulse] Reusing the existing configuration."
        )
        print()

        return True

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

                # Applying a fix normally restarts the training loop so the
                # running process picks it up. There is no training loop yet
                # here -- the script has never started -- and that restart
                # re-executed the temporary repair copy directly, taking over
                # from this function: the validation, instrumentation and run
                # below were all skipped, and the user's own file was left
                # broken with the fix stranded in a temp file. This repair
                # decides for itself when the run starts.
                pulse._suppress_auto_restart = True

                # ----------------------------------------------------
                # First-run setup.
                #
                # IMPORTANT:
                #
                # _load_config() is called by set_code_text().
                # Therefore, if a config already exists, Pulse should
                # already have loaded it and this setup is skipped.
                #
                # If no usable agent exists, create the config and
                # then explicitly reload it.
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
                    # With no terminal there is nobody to answer the setup
                    # prompts below, so a CI job or a piped shell stopped here
                    # with "setup was not completed" and the syntax error was
                    # never fixed. Pulse already has a documented headless
                    # path -- PULSE_PROVIDER plus that provider's API key env
                    # var -- so try that first when there is no tty. In a
                    # terminal nothing changes: setup runs exactly as below.
                    if not sys.stdin or not sys.stdin.isatty():
                        pulse.non_interactive = True
                        try:
                            pulse._select_agent_provider_and_key(
                                initial=True
                            )
                        except (EOFError, KeyboardInterrupt):
                            pass

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

                    # The config was just created, so reload it.
                    #
                    # If it already existed, _bootstrap_pulse_config()
                    # did not modify it; reloading it is harmless and
                    # guarantees the current config is authoritative.
                    pulse._load_config()

                # ----------------------------------------------------
                # Initialize the configured provider using Pulse's
                # EXISTING provider-selection machinery.
                # ----------------------------------------------------

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

                # A rate limit is not a failed repair. The crash handler in
                # pulse.py already retries these rather than giving up on the
                # run, and without the same treatment here one busy minute at
                # the provider turned into "the repair pipeline did not apply
                # a fix" and the script was left broken.
                for delay in _REPAIR_RETRY_DELAYS:
                    if not getattr(
                        pulse,
                        "_last_call_failed_transiently",
                        False,
                    ):
                        break
                    print(
                        f"[Pulse] The agent was unreachable "
                        f"(rate limit or a transient error) -- "
                        f"retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    result = pulse.ask_agent(
                        question,
                        include_code=True,
                    )

                if not getattr(
                    pulse,
                    "_fix_applied_this_turn",
                    False,
                ):
                    if getattr(
                        pulse,
                        "_last_call_failed_transiently",
                        False,
                    ):
                        raise RuntimeError(
                            "Pulse could not reach the AI provider to fix "
                            "the syntax error (rate limited or unavailable "
                            "after retries). The script is unchanged -- "
                            "try again shortly."
                        )

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

                # The repair happened on a temporary copy, which is what gets
                # instrumented and run. Leaving it there means the user never
                # receives the fix: `python train.py` still fails, and the next
                # `pulse run` pays the agent to find the same missing bracket
                # again. Every other fix Pulse makes is written to the real
                # file and logged for /revert, so this one is too.
                try:
                    with open(
                        script_path,
                        "w",
                        encoding="utf-8",
                    ) as original:
                        original.write(repaired_source)

                    print(
                        f"[Pulse] Wrote the fix to "
                        f"{os.path.basename(script_path)} "
                        f"(/log to see it, /revert to undo)."
                    )

                except OSError as write_error:
                    # A read-only checkout is a good reason not to run, but not
                    # a good reason to refuse to train: the repaired copy is
                    # still perfectly runnable.
                    print(
                        f"[Pulse] Could not write the fix back to "
                        f"{script_path}: {write_error}\n"
                        f"[Pulse] Running the repaired copy instead -- "
                        f"the file on disk is still broken."
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
