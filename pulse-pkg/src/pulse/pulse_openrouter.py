"""
OpenRouter sign-in: get an API key by signing in -- or signing up -- in the browser.

Pulse's agent needs a model, and for someone trying Pulse for the first time the hard part
is having an API key at all. OpenRouter serves every major model behind one key, has free
models, and lets an app ask for a key on the user's behalf (OAuth with PKCE):

    1. Pulse makes a random secret (the verifier) and sends its hash (the challenge) along
       with the person to https://openrouter.ai/auth.
    2. They sign in or create an account there and click Authorize.
    3. OpenRouter hands back a one-time code -- by redirecting the browser to a tiny server
       Pulse runs on localhost, or, when there is no browser on this machine (SSH, a
       container), by showing the code for the person to paste.
    4. Pulse trades code + verifier for an API key. The code alone is useless to anyone
       else: only this process knows the verifier.

After sign-in, Pulse asks whether to save the key in its owner-only device key store;
`pulse openrouter logout` removes the saved key. A key that is *pasted* is never written
to disk without the user's consent.

Nothing here talks to a model. The only requests are the code exchange and, for
`pulse openrouter status`, a read of the key's own usage.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import queue
import secrets
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

AUTH_URL = "https://openrouter.ai/auth"
EXCHANGE_URL = "https://openrouter.ai/api/v1/auth/keys"
KEY_INFO_URL = "https://openrouter.ai/api/v1/key"
KEYS_PAGE = "https://openrouter.ai/settings/keys"
CREDITS_PAGE = "https://openrouter.ai/credits"
ENV_KEY = "OPENROUTER_API_KEY"
KEY_LABEL = "Pulse"
# OpenRouter's codes expire after 10 minutes; waiting longer than that for the browser is
# waiting for a code that can no longer be exchanged.
WAIT_SECONDS = 540.0
_CALLBACK_PATH = "/callback"

_DONE_PAGE = """<!doctype html><meta charset="utf-8"><title>Pulse</title>
<body style="font:16px/1.5 system-ui,sans-serif;max-width:32em;margin:15vh auto;padding:0 1em">
<h2>Signed in to OpenRouter</h2>
<p>Pulse has the sign-in. You can close this tab and go back to your terminal.</p>"""


class SignInError(Exception):
    """The sign-in did not produce a key. The message is written for the person."""


# ---------------------------------------------------------------------------------------
# PKCE and the two requests
# ---------------------------------------------------------------------------------------

def new_pkce() -> Tuple[str, str]:
    """(verifier, challenge): a random secret and the base64url SHA-256 of it."""
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return verifier, base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def authorize_url(challenge: str, callback_url: Optional[str] = None) -> str:
    """Where the person signs in. Without a callback_url OpenRouter shows the code on
    screen instead of redirecting (for machines with no browser of their own)."""
    params = {"code_challenge": challenge, "code_challenge_method": "S256", "key_label": KEY_LABEL}
    if callback_url:
        params = dict(callback_url=callback_url, **params)
    return AUTH_URL + "?" + urllib.parse.urlencode(params)


def _request_json(url: str, payload: Optional[Dict[str, Any]] = None,
                  headers: Optional[Dict[str, str]] = None, timeout: float = 30.0
                  ) -> Tuple[int, Dict[str, Any]]:
    """(status, parsed body). Raises SignInError only when OpenRouter can't be reached."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url, data=data, method="POST" if data is not None else "GET",
        headers={"Content-Type": "application/json", "User-Agent": "pulseml", **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, raw = error.code, error.read()
    except (urllib.error.URLError, OSError) as error:
        raise SignInError(f"could not reach OpenRouter ({getattr(error, 'reason', error)})") from None
    try:
        body = json.loads(raw.decode("utf-8", "replace") or "{}")
    except ValueError:
        body = {}
    return status, body if isinstance(body, dict) else {}


def _error_text(body: Dict[str, Any]) -> str:
    error = body.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or "")
    return str(error or "")


def exchange(code: str, verifier: str) -> str:
    """Trade the one-time code for an API key."""
    code = (code or "").strip()
    if not code:
        raise SignInError("no code was given")
    status, body = _request_json(EXCHANGE_URL, {
        "code": code, "code_verifier": verifier, "code_challenge_method": "S256"})
    key = body.get("key")
    if status == 200 and isinstance(key, str) and key.strip():
        return key.strip()
    detail = _error_text(body) or f"HTTP {status}"
    raise SignInError(
        f"OpenRouter did not accept that code ({detail}). A code works once, for 10 minutes, "
        "and only for the sign-in that was started from this terminal.")


def key_info(key: str) -> Optional[Dict[str, Any]]:
    """What OpenRouter says about this key (label, usage, limit, is_free_tier), or None
    when it can't be read -- an unreachable server and a rejected key both land here."""
    try:
        status, body = _request_json(KEY_INFO_URL, headers={"Authorization": f"Bearer {key}"}, timeout=15)
    except SignInError:
        return None
    data = body.get("data")
    return data if status == 200 and isinstance(data, dict) else None


def key_is_rejected(key: str) -> bool:
    """True only when OpenRouter answered and said this key is not valid (deleted, mistyped).
    Unreachable is not rejected: a run on a plane should not lose its sign-in."""
    try:
        status, _body = _request_json(KEY_INFO_URL, headers={"Authorization": f"Bearer {key}"}, timeout=15)
    except SignInError:
        return False
    return status in (401, 403)


# ---------------------------------------------------------------------------------------
# The localhost callback
# ---------------------------------------------------------------------------------------

class _Callback:
    """A one-purpose web server: the browser lands on /callback?code=... and the code is
    queued. Bound to loopback only; anything else it is asked for is a 404."""

    def __init__(self) -> None:
        self.codes: "queue.Queue[str]" = queue.Queue()
        codes = self.codes

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):                                      # noqa: N802 (http.server API)
                parsed = urllib.parse.urlparse(self.path)
                code = (urllib.parse.parse_qs(parsed.query).get("code") or [""])[0].strip()
                if parsed.path != _CALLBACK_PATH or not code:
                    self.send_error(404)
                    return
                codes.put(code)
                page = _DONE_PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(page)

            def log_message(self, *args):                          # keep the terminal clean
                pass

        self._servers: List[ThreadingHTTPServer] = []
        first = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = first.server_address[1]
        self._servers.append(first)
        # "localhost" resolves to ::1 first in some browsers; answer there too when we can.
        try:
            import socket

            class V6(ThreadingHTTPServer):
                address_family = socket.AF_INET6

            self._servers.append(V6(("::1", self.port), Handler))
        except Exception:
            pass
        for server in self._servers:
            server.daemon_threads = True
            threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1},
                             daemon=True, name="pulse-openrouter-callback").start()
        self.url = f"http://localhost:{self.port}{_CALLBACK_PATH}"

    def close(self) -> None:
        for server in self._servers:
            try:
                server.shutdown()
                server.server_close()
            except Exception:
                pass
        self._servers = []


def can_open_browser() -> bool:
    """Is there a browser on this machine that the person is sitting at? Not over SSH, and
    not on a Linux box with no display. A wrong 'yes' costs nothing: the link is printed
    either way and Ctrl+C switches to pasting the code."""
    if os.environ.get("PULSE_OPENROUTER_NO_BROWSER", "").strip().lower() in ("1", "true", "yes", "on"):
        return False
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY") or os.environ.get("SSH_CLIENT"):
        return False
    if sys.platform.startswith("linux") and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or os.environ.get("BROWSER")
            or "microsoft" in os.uname().release.lower()):
        return False
    try:
        import webbrowser
        webbrowser.get()
    except Exception:
        return False
    return True


def code_from(text: str) -> str:
    """The code out of whatever was pasted: the bare code, or the address of the page the
    browser was sent to (http://localhost:PORT/callback?code=...)."""
    text = (text or "").strip().strip("'\"")
    if "code=" in text:
        query = urllib.parse.urlparse(text).query or text.split("?", 1)[-1]
        found = (urllib.parse.parse_qs(query).get("code") or [""])[0].strip()
        if found:
            return found
    return text


# ---------------------------------------------------------------------------------------
# The sign-in itself
# ---------------------------------------------------------------------------------------

def sign_in(*, ask: Optional[Callable[[str], str]] = None, say: Callable[[str], None] = print,
            open_browser: Optional[Callable[[str], Any]] = None, browser: Optional[bool] = None,
            wait_seconds: float = WAIT_SECONDS) -> str:
    """Run the sign-in and return the new API key (not saved -- see save_key).

    `ask(prompt)` reads a pasted code from the person; without it (a GUI) only the browser
    redirect can finish the sign-in. `browser` forces the choice between the redirect flow
    (True) and the paste flow (False); by default it is whatever can_open_browser() says.
    Raises SignInError with a message for the person when no key comes out of it.
    """
    verifier, challenge = new_pkce()
    paste_url = authorize_url(challenge)
    use_browser = can_open_browser() if browser is None else browser
    if not use_browser and ask is None:
        raise SignInError("there is no browser on this machine and nowhere to paste a code")

    if use_browser:
        try:
            callback = _Callback()
        except OSError as error:
            if ask is None:
                raise SignInError(f"could not listen on localhost for the sign-in ({error})") from None
            callback = None
        if callback is not None:
            url = authorize_url(challenge, callback.url)
            say("Sign in to OpenRouter, or create an account, in your browser:")
            say(f"  {url}")
            say("Waiting for you to click Authorize there..."
                + ("  (Ctrl+C to paste a code instead)" if ask is not None else ""))
            try:
                if open_browser is None:
                    import webbrowser
                    open_browser = webbrowser.open
                open_browser(url)
            except Exception:
                pass                      # the link is on screen; opening it was a courtesy
            interrupted = False
            try:
                deadline = time.monotonic() + wait_seconds
                while time.monotonic() < deadline:
                    try:
                        code = callback.codes.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    try:
                        return exchange(code, verifier)
                    except SignInError as error:
                        if "could not reach" in str(error):
                            raise
                        # Anything on this machine can hit a localhost port. A code that
                        # OpenRouter rejects is not the person's sign-in: keep waiting.
                        continue
            except KeyboardInterrupt:
                interrupted = True
            finally:
                callback.close()
            if ask is None:
                raise SignInError("cancelled" if interrupted else
                                  "nothing came back from the browser in time")
            say("" if interrupted else "Nothing came back from the browser.")
            say("If the browser ended on a page that would not load, paste that page's address.")
            say("Or open this link on any device and paste the code it shows:")
            say(f"  {paste_url}")
    else:
        say("Open this link on any device, sign in to OpenRouter or create an account, and")
        say("click Authorize. OpenRouter then shows a code:")
        say(f"  {paste_url}")

    try:
        pasted = ask("Code (Enter to cancel)")
    except (EOFError, KeyboardInterrupt):
        pasted = ""
    code = code_from(pasted)
    if not code:
        raise SignInError("cancelled")
    return exchange(code, verifier)


# ---------------------------------------------------------------------------------------
# Staying signed in
# ---------------------------------------------------------------------------------------

def _store_path() -> Path:
    from . import pulse_supabase as cloud
    return cloud.CACHE_PATH.parent / "openrouter.json"


def _register(key: str) -> None:
    try:                                  # scrubbed wherever it shows up, whatever its shape
        from . import pulse_supabase as cloud
        cloud.register_secret(key)
    except Exception:
        pass


def save_key(key: str) -> bool:
    """Keep a signed-in key for next time, readable by the owner only."""
    try:
        from . import pulse_settings
        if not pulse_settings.on("remember_keys"):
            return False
    except ImportError:
        pass
    try:
        from . import pulse_supabase as cloud
        path = _store_path()
        cloud._ensure_private_dir(path.parent)
        cloud._write_private(path, json.dumps({"key": key, "saved_at": time.time()}, indent=2))
    except Exception:
        return False
    _register(key)
    return True


def saved_key() -> Optional[str]:
    """The key from an earlier sign-in, or None. PULSE_OPENROUTER_SAVED_KEY=off ignores it."""
    if os.environ.get("PULSE_OPENROUTER_SAVED_KEY", "").strip().lower() in ("off", "0", "false", "no"):
        return None
    try:
        data = json.loads(_store_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    key = data.get("key") if isinstance(data, dict) else None
    if not isinstance(key, str) or not key.strip():
        return None
    _register(key.strip())
    return key.strip()


def forget_key() -> bool:
    """Sign out on this machine. True when there was a saved key to remove."""
    try:
        _store_path().unlink()
        return True
    except OSError:
        return False


def key_for(model: Optional[str]) -> Optional[str]:
    """The key to use for an OpenRouter model when none was given: the environment's, else
    the saved sign-in. None for any other provider's model."""
    if not str(model or "").lower().startswith("openrouter/"):
        return None
    return os.environ.get(ENV_KEY, "").strip() or saved_key()


def offer_sign_in_for(model: Optional[str], *, ask: Optional[Callable[[str], str]] = None,
                      say: Callable[[str], None] = print) -> Optional[str]:
    """An OpenRouter model was asked for and there is no key anywhere: offer the sign-in.
    Returns the key now available (None for other providers' models, when the person
    declines, or with no terminal to ask on -- then only says how to sign in)."""
    if not str(model or "").lower().startswith("openrouter/"):
        return None
    key = key_for(model)
    if key:
        return key
    interactive = ask is not None or (sys.stdin is not None and sys.stdin.isatty())
    if not interactive:
        say(f"No OpenRouter key on this machine for {model}: run `pulse openrouter` to sign in "
            f"(or set {ENV_KEY}).")
        return None
    ask = ask or (lambda prompt: input(prompt + " > "))
    try:
        answer = ask(f"No OpenRouter key on this machine for {model}. Sign in, or create an "
                     "account, now? (Y/n)").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None
    if answer not in ("", "y", "yes"):
        return None
    try:
        key = sign_in(ask=ask, say=say)
    except SignInError as error:
        say(f"OpenRouter sign-in did not finish: {error}")
        return None
    try:
        keep = ask("Save this API key in Pulse's private device key store so you stay signed in? (y/N)").strip().lower()
    except (EOFError, KeyboardInterrupt):
        keep = ""
    kept = save_key(key) if keep in ("y", "yes") else False
    say(f"Signed in to OpenRouter (key {tail(key)})"
        + (" -- saved on this machine; `pulse openrouter logout` removes it." if kept
           else " -- not saved; it is available for this session only."))
    os.environ[ENV_KEY] = key
    return key


def tail(key: str) -> str:
    """A key, shown safely: its last four characters."""
    return "…" + key[-4:] if key and len(key) > 8 else "…"


def describe(info: Optional[Dict[str, Any]]) -> str:
    """One line on what a key can do, from key_info()."""
    if not info:
        return "could not check it with OpenRouter just now"
    parts = []
    usage, limit = info.get("usage"), info.get("limit")
    if isinstance(usage, (int, float)):
        parts.append(f"${usage:.2f} used")
    if isinstance(limit, (int, float)):
        parts.append(f"${limit:.2f} limit")
    if info.get("is_free_tier"):
        parts.append(f"no credits bought yet -- the free models work; add credits at {CREDITS_PAGE} for the rest")
    return ", ".join(parts) or "working"


# ---------------------------------------------------------------------------------------
# `pulse openrouter [login|logout|status]`
# ---------------------------------------------------------------------------------------

USAGE = """\
pulse openrouter            sign in to OpenRouter (or create an account) and keep the key
pulse openrouter status     whether this machine is signed in, and what the key has used
pulse openrouter logout     forget the saved key on this machine
"""


def main(argv: List[str]) -> int:
    action = (argv[0] if argv else "login").lower()
    if action in ("-h", "--help", "help"):
        print(USAGE)
        return 0
    if action == "status":
        env = os.environ.get(ENV_KEY, "").strip()
        key = saved_key()
        if key:
            print(f"Signed in to OpenRouter on this machine (key {tail(key)}): {describe(key_info(key))}.")
        else:
            print("Not signed in to OpenRouter on this machine. `pulse openrouter` signs in.")
        if env and env != key:
            print(f"{ENV_KEY} is also set in this shell (key {tail(env)}); Pulse asks which to use.")
        return 0
    if action == "logout":
        if forget_key():
            print("Signed out: the saved OpenRouter key is removed from this machine.")
            print(f"The key itself still exists on your account -- delete it at {KEYS_PAGE}")
        else:
            print("There was no saved OpenRouter key on this machine.")
        return 0
    if action != "login":
        print(f"pulse openrouter: unknown command {action!r}\n")
        print(USAGE)
        return 1
    try:
        key = sign_in(ask=lambda prompt: input(prompt + " > "))
    except SignInError as error:
        print(f"OpenRouter sign-in did not finish: {error}")
        return 1
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130
    try:
        keep = input("Save this API key in Pulse's private device key store so you stay signed in? (y/N) > ").strip().lower()
    except EOFError:
        print("\nNot saved.")
        return 1
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130
    if keep not in ("y", "yes"):
        print(f"Signed in to OpenRouter, but the key was not saved. Key {tail(key)} is available only for this session.")
        print(f"Save it later by running `pulse openrouter` again or use `pulse config remember_keys on`.")
        return 0
    if not save_key(key):
        print(f"Signed in, but the key could not be saved under {_store_path().parent} "
              "(is that folder writable?).")
        print(f"Fix that and run `pulse openrouter` again; the unused key can be deleted at {KEYS_PAGE}")
        return 1
    print(f"Signed in to OpenRouter. Key {tail(key)} is saved for Pulse on this machine "
          f"({_store_path()}).")
    print(f"It is ready to use: {describe(key_info(key))}.")
    return 0
