"""Remembered choices: ~/.pulse/settings.json (and the API keys, in their own private file).

Setup asks for the agent's model and key, the workspace and the project. What was chosen is
kept here, so the next start uses it without asking; `/config` in the app (or `pulse config`
in a shell) shows the settings and changes them, and `/config reset` forgets them so the
next start asks again.

The settings file holds no secrets. A pasted API key goes to keys.json beside it, readable
by its owner only -- unless `remember_keys` is off, and then no key is ever written. The
environment always wins over a saved key.

A pulse_config.json beside a script (unattended runs) is something else and still wins over
all of this: it is read by PulseCLI._load_config.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# name -> (what it is, the values it takes ("" = free text), its default)
KNOWN: Dict[str, Tuple[str, Tuple[str, ...], Any]] = {
    "agent": ("the agent's provider and model -- /config agent opens the picker", (), None),
    "workspace": ("the Pulse Cloud workspace runs go to -- /config workspace opens the picker", (), None),
    "project": ("the project inside it -- /config project opens the picker", (), None),
    "approver": ("who answers the y/N for flagged commands and reviews changes: \"same\" (the agent's own "
                 "model), a model id, or \"off\" (you are asked)", (), "same"),
    "review": ("ask before applying a change (a reviewer model answers when one is set)", ("on", "off"), "on"),
    "mouse": ("the app reads the mouse: drag to copy text, click to open folded lines, the wheel scrolls; off "
              "leaves the mouse to your terminal", ("on", "off"), "on"),
    "audits": ("the agent checks open runs on its own schedule", ("on", "off"), "on"),
    "remember_keys": ("keep pasted API keys in ~/.pulse/keys.json (owner-only) so you are not asked again",
                      ("on", "off"), "on"),
}


def _home() -> Path:
    # Beside credentials.json, wherever that lives (PULSE_CACHE_DIR, or ~/.pulse). Read from
    # pulse_supabase at call time, so a test that points it elsewhere moves this too.
    try:
        from . import pulse_supabase as cloud
        return Path(cloud.CACHE_PATH).parent
    except Exception:
        return Path(os.environ.get("PULSE_CACHE_DIR", str(Path.home() / ".pulse")))


def path() -> Path:
    return _home() / "settings.json"


def keys_path() -> Path:
    return _home() / "keys.json"


def _read(file: Path) -> Dict[str, Any]:
    try:
        with open(file, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(file: Path, data: Dict[str, Any], private: bool = False) -> bool:
    try:
        file.parent.mkdir(parents=True, exist_ok=True)
        tmp = file.with_name(file.name + ".tmp")
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        fd = os.open(str(tmp), flags, 0o600 if private else 0o644)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
        if private:
            try:
                os.chmod(tmp, 0o600)
            except OSError:
                pass
        os.replace(tmp, file)
        return True
    except OSError:
        return False


def load() -> Dict[str, Any]:
    return _read(path())


def get(name: str, default: Any = None) -> Any:
    value = load().get(name)
    if value is None:
        return KNOWN[name][2] if name in KNOWN and default is None else default
    return value


def is_set(name: str) -> bool:
    return load().get(name) is not None


def on(name: str) -> bool:
    return str(get(name) or "").strip().lower() in ("on", "true", "yes", "1")


def set(name: str, value: Any) -> bool:  # noqa: A001 -- settings.set reads well
    data = load()
    if value is None:
        data.pop(name, None)
    else:
        data[name] = value
    return _write(path(), data)


def unset(name: str) -> bool:
    return set(name, None)


def reset() -> None:
    """Forget every setting and every saved key: the next start asks again."""
    for file in (path(), keys_path()):
        try:
            file.unlink()
        except OSError:
            pass


def check(name: str, value: str) -> Optional[str]:
    """Why `value` is not a valid value for `name`, or None."""
    if name not in KNOWN:
        return f"there is no setting called {name!r} (there are: {', '.join(KNOWN)})"
    choices = KNOWN[name][1]
    if choices and value.lower() not in choices:
        return f"{name} is {' or '.join(choices)}"
    return None


# ---------------------------------------------------------------------------------- keys

def saved_key(env_var: str) -> str:
    return str(_read(keys_path()).get(env_var) or "")


def save_key(env_var: Optional[str], key: Optional[str]) -> bool:
    """Keep `key` for `env_var` (e.g. DEEPSEEK_API_KEY) -- if keys are remembered at all."""
    if not env_var or not key or key == "local" or not on("remember_keys"):
        return False
    keys = _read(keys_path())
    if keys.get(env_var) == key:
        return True
    keys[env_var] = key
    return _write(keys_path(), keys, private=True)


def forget_keys() -> None:
    try:
        keys_path().unlink()
    except OSError:
        pass


def load_keys_into_environment() -> List[str]:
    """Saved keys become the environment variables the providers read, where those are not
    already set (the environment wins). Returns the names that were filled in."""
    filled = []
    for env_var, key in _read(keys_path()).items():
        if key and not os.environ.get(env_var):
            os.environ[env_var] = str(key)
            filled.append(env_var)
    return filled


# ---------------------------------------------------------------------------------- the agent

def remember_agent(cli: Any) -> None:
    """What the agent is now, so the next start uses it without asking (and its key, if keys
    are remembered). Called after the person picked one."""
    from .pulse_cli import PROVIDERS
    name = getattr(cli, "agent_provider", None)
    if not name or name not in PROVIDERS:
        return
    info = PROVIDERS[name]
    record: Dict[str, Any] = {"provider": name}
    model = info.get("model")
    if info.get("local"):
        record["local_model"] = (getattr(cli, "agent_model_string", "") or "").split("/", 1)[-1]
        record["api_base"] = getattr(cli, "agent_api_base", None)
    elif model and (str(model).startswith("openrouter/") or name.startswith("Custom:")):
        record["model"] = model
        if info.get("env_key"):
            record["env_key"] = info["env_key"]
    set("agent", record)
    save_key(info.get("env_key"), getattr(cli, "agent_key", None))


def describe_agent(record: Any) -> str:
    if not isinstance(record, dict):
        return str(record or "")
    if record.get("local_model"):
        return f"{record.get('provider')} ({record['local_model']} @ {record.get('api_base')})"
    return str(record.get("model") or record.get("provider") or "")


def apply_agent(cli: Any) -> bool:
    """Set the agent from the settings, without a question. False when there is no saved
    agent, or it cannot be used now (its key is missing) -- then setup asks as before."""
    record = get("agent")
    if not isinstance(record, dict) or not record.get("provider"):
        return False
    load_keys_into_environment()
    if record.get("local_model"):
        want = record["provider"]
        extra = {"PULSE_LOCAL_MODEL": record["local_model"], "PULSE_LOCAL_API_BASE": record.get("api_base") or ""}
    elif record.get("model") and str(record["model"]).startswith("openrouter/"):
        want, extra = record["model"], {}
    elif record.get("model"):
        want, extra = record["model"], {"PULSE_API_KEY_ENV": record.get("env_key") or ""}
    else:
        want, extra = record["provider"], {}
    saved = {k: os.environ.get(k) for k in ["PULSE_PROVIDER", *extra]}
    os.environ["PULSE_PROVIDER"] = want
    for k, v in extra.items():
        if v:
            os.environ[k] = v
    was = getattr(cli, "non_interactive", False)
    cli.non_interactive = True
    try:
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            ok = bool(cli._select_agent_provider_and_key(initial=True))
    except Exception:
        ok = False
    finally:
        cli.non_interactive = was
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return ok and bool(getattr(cli, "agent_provider", None))


def remembered_id(name: str) -> Optional[str]:
    """The id kept for "workspace" or "project" ({"id", "name"}; a bare id also works)."""
    value = get(name)
    if isinstance(value, dict):
        return str(value.get("id") or "") or None
    return str(value) if value else None


def rows() -> List[Tuple[str, str, str]]:
    """(name, value as shown, what it is) for every setting."""
    data = load()
    out = []
    for name, (what, _choices, default) in KNOWN.items():
        value = data.get(name)
        if name == "agent":
            shown = describe_agent(value) if value else "not set (asked at start)"
        elif name in ("workspace", "project"):
            shown = (str(value.get("name") or value.get("id")) if isinstance(value, dict) else str(value)) \
                if value else "not set (asked at start)"
        else:
            shown = str(value if value is not None else default) + ("" if value is not None else " (default)")
        out.append((name, shown, what))
    keys = sorted(_read(keys_path()))
    out.append(("keys", ", ".join(keys) if keys else "none saved", f"saved in {keys_path()} (owner-only)"))
    return out


def main(argv: List[str]) -> int:
    """`pulse config`: show the settings; `pulse config <name> <value>` sets one;
    `pulse config <name> --unset`, `pulse config reset`."""
    if not argv:
        width = max(len(name) for name, _v, _w in rows())
        print(f"Pulse settings ({path()}):\n")
        for name, shown, what in rows():
            print(f"  {name.ljust(width)}  {shown}")
            print(f"  {' ' * width}  {what}")
        print("\nChange one: pulse config <name> <value>   (or /config inside pulse; /config agent opens the picker)")
        print("Forget them all: pulse config reset")
        return 0
    if argv[0] == "reset":
        reset()
        print("Settings and saved keys forgotten: the next start asks again.")
        return 0
    name = argv[0]
    if name not in KNOWN:
        print(f"pulse config: {check(name, '')}")
        return 2
    if len(argv) == 1:
        print(f"{name} = {dict((n, v) for n, v, _w in rows())[name]}")
        return 0
    if argv[1] == "--unset":
        unset(name)
        print(f"{name} unset: " + ("asked at the next start." if KNOWN[name][2] is None else f"back to {KNOWN[name][2]}."))
        return 0
    if name in ("agent", "workspace", "project"):
        print(f"{name} is chosen from a picker: run `pulse` and type /config {name}.")
        return 2
    value = " ".join(argv[1:]).strip()
    problem = check(name, value)
    if problem:
        print(f"pulse config: {problem}")
        return 2
    set(name, value.lower() if KNOWN[name][1] else value)
    if name == "remember_keys" and value.lower() == "off":
        forget_keys()
    print(f"{name} = {get(name)}")
    return 0
