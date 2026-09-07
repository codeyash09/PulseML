"""
pulse_supabase.py
==================
Thin Supabase (PostgREST) client for the Pulse CLI's cloud features:
account (profiles), team collaboration (Teams, join codes), and live debug
session logging (Debug_Sessions: agent_logs / error_tracebacks / telemetry).

Deliberately dependency-free (uses `urllib`, not `requests` or the
`supabase-py` SDK) so this doesn't add a new install requirement on top of
Pulse's existing deps. Every network call is wrapped so a bad/offline
connection degrades gracefully (prints a warning once, never raises into
the training loop) -- same robustness philosophy as the rest of Pulse.

SECURITY NOTE: Passwords are now managed by Supabase Auth (auth.users table)
rather than in this application. The profiles table stores only non-sensitive
user metadata (email, personal_plan, etc.). RLS policies on the profiles
table should be configured to prevent unauthorized access. Supabase Auth
provides proper password hashing, session management, and account recovery.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import platform
import queue
import re
import secrets
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
import base64
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

SUPABASE_URL = os.environ.get(
    "PULSE_SUPABASE_URL", "https://wlvcvbaoxqgryarasjvw.supabase.co"
)
SUPABASE_KEY = os.environ.get(
    "PULSE_SUPABASE_KEY", "sb_publishable_o1djx0_qUWbPrYZXu9XiVw_PU1Wdx6U"
)

_REST_URL = f"{SUPABASE_URL.rstrip('/')}/rest/v1"
_AUTH_URL = f"{SUPABASE_URL.rstrip('/')}/auth/v1"

CACHE_PATH = Path(os.environ.get("PULSE_CACHE_DIR", str(Path.home() / ".pulse"))) / "credentials.json"
# Local-only run profile: the non-secret bits of "how this machine last ran
# Pulse" -- which provider it used (never the API key itself, see
# save_cached_profile's docstring), the last git commit sha Pulse synced,
# and the last-detected GPU info. Kept separate from credentials.json so
# clearing one (e.g. /logout) never silently wipes the other.
PROFILE_PATH = Path(os.environ.get("PULSE_CACHE_DIR", str(Path.home() / ".pulse"))) / "profile.json"

_TIMEOUT_INTERACTIVE = 8   # auth/team calls -- user is actively waiting
_TIMEOUT_BACKGROUND = 5    # session-log syncs -- best effort

# How long a cached login (credentials.json) is trusted before Pulse asks
# for a password again. Nothing here is a real session token (see the
# SECURITY NOTE above and _auth_flow in pulse_cli.py) -- re-prompting
# periodically is a partial mitigation against a stale/leaked
# credentials.json being usable indefinitely, not a substitute for real
# server-side session validation. Override with PULSE_SESSION_TTL_DAYS.
try:
    SESSION_TTL_DAYS = float(os.environ.get("PULSE_SESSION_TTL_DAYS", "30"))
except ValueError:
    SESSION_TTL_DAYS = 30.0


# ----------------------------------------------------------------------------
# Secret scrubbing -- applied to anything that leaves this machine (cloud
# telemetry/agent_logs/error_tracebacks) or gets written to a local log the
# user might paste/share (.pulse_history), so a captured API key, Supabase
# key, or generic bearer token doesn't ride along inside a stack trace or
# an agent transcript. Best-effort pattern matching, not a guarantee --
# this is defense in depth, not a substitute for not logging secrets in
# the first place.
# ----------------------------------------------------------------------------

_SECRET_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9\-_]{10,}"),          # Anthropic
    re.compile(r"sk-(?!ant-)[A-Za-z0-9]{20,}"),          # OpenAI-style
    re.compile(r"sk-proj-[A-Za-z0-9\-_]{10,}"),          # OpenAI project keys
    re.compile(r"AIza[0-9A-Za-z\-_]{20,}"),              # Gemini/Google
    re.compile(r"sb_(publishable|secret)_[A-Za-z0-9_\-]{10,}"),  # Supabase
    re.compile(r"gsk_[A-Za-z0-9]{20,}"),                 # Groq
    re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"),        # Slack tokens
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),                 # GitHub PAT
    re.compile(r"(?i)bearer\s+[A-Za-z0-9\-_.=]{10,}"),
    re.compile(r"(?i)(api[_-]?key|api[_-]?secret|access[_-]?token|password)\s*[:=]\s*['\"]?[A-Za-z0-9\-_./+=]{8,}['\"]?"),
]


def scrub_secrets(text: Optional[str]) -> Optional[str]:
    """Redact anything that looks like an API key/token/password out of
    `text`. Also scrubs the literal value of every *_API_KEY / *_TOKEN /
    *_SECRET environment variable currently set, so a key that doesn't
    match one of the known vendor formats above still gets caught. Safe to
    call on None/non-strings (passes through unchanged); never raises."""
    if not isinstance(text, str) or not text:
        return text
    scrubbed = text
    for pattern in _SECRET_PATTERNS:
        scrubbed = pattern.sub("[REDACTED_SECRET]", scrubbed)
    for env_name, env_val in os.environ.items():
        if not env_val or len(env_val) < 8:
            continue
        if re.search(r"(?i)(api[_-]?key|token|secret|password)", env_name) and env_val in scrubbed:
            scrubbed = scrubbed.replace(env_val, "[REDACTED_SECRET]")
    return scrubbed


def scrub_secrets_deep(obj: Any) -> Any:
    """Recursively apply scrub_secrets to every string inside a dict/list
    (telemetry payloads, agent log entries, etc. are usually nested)."""
    if isinstance(obj, str):
        return scrub_secrets(obj)
    if isinstance(obj, dict):
        return {k: scrub_secrets_deep(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub_secrets_deep(v) for v in obj]
    return obj


# ----------------------------------------------------------------------------
# Compression codec -- every entry pushed into agent_logs / telemetry /
# error_tracebacks is compressed before it leaves this process, and
# transparently decompressed on the way back in (fetch_recent_sessions
# callers should run entries through decode_entry/decode_entries). This
# cuts the bytes actually sent over the wire (Supabase's smallest compute
# tier chokes on large/frequent payloads) and is self-describing -- any
# Pulse install, old or new, can tell a compressed entry apart from a
# legacy plain one and read either.
# ----------------------------------------------------------------------------

_CODEC_PREFIX = "PZ1:"  # "Pulse Zip", format version 1: zlib + base64


def encode_entry(obj: Any) -> str:
    """Compress one array element (a dict or a plain traceback string) into
    a single self-describing string. Falls back to an uncompressed value
    if compression ever fails for some reason -- never raises.

    Every entry is scrubbed for secrets (see scrub_secrets_deep) before
    compression, since this is the one chokepoint every agent_logs/
    error_tracebacks/telemetry entry passes through on its way to the
    cloud -- catching it here means callers don't each have to remember to
    scrub individually."""
    try:
        obj = scrub_secrets_deep(obj)
        raw = json.dumps(obj, separators=(",", ":"), default=str).encode("utf-8")
        comp = zlib.compress(raw, level=9)
        return _CODEC_PREFIX + base64.b64encode(comp).decode("ascii")
    except Exception:
        return obj if isinstance(obj, str) else json.dumps(obj, default=str)


def decode_entry(value: Any) -> Any:
    """Inverse of encode_entry. Values that don't carry the PZ1: prefix are
    passed through as-is, so older, pre-compression rows already sitting
    in the table stay readable."""
    if isinstance(value, str) and value.startswith(_CODEC_PREFIX):
        try:
            comp = base64.b64decode(value[len(_CODEC_PREFIX):])
            raw = zlib.decompress(comp)
            return json.loads(raw)
        except Exception:
            return value
    return value


def decode_entries(values: Optional[List[Any]]) -> List[Any]:
    return [decode_entry(v) for v in (values or [])]


# ----------------------------------------------------------------------------
# Low-level REST helpers
# ----------------------------------------------------------------------------

class SupabaseError(Exception):
    pass


class SupabaseEmailConfirmationRequired(SupabaseError):
    """Raised by sign_up() when the account was actually created but this
    project's Auth settings require the user to confirm their email
    before a session can be issued. A subclass of SupabaseError (not a
    separate flag) so any existing `except SupabaseError` call site keeps
    working unchanged, while a call site that wants to show a "go check
    your email" message instead of a generic "sign up failed" message can
    catch this specifically."""
    pass


# ----------------------------------------------------------------------------
# Auth session state -- the real Supabase Auth JWT for the signed-in user,
# as distinct from the app-level "session_token"/session_token_hash pair
# further down (which only proves "this device previously logged in", not
# "this request is authenticated as this user" to PostgREST). Every REST
# call made after a successful sign_up/log_in/refresh_session automatically
# carries this JWT as its Authorization bearer instead of the anon
# SUPABASE_KEY (see _headers below) -- which is what lets Postgres RLS
# policies like `auth.uid() = id` actually pass. Calls made with only the
# anon key are unauthenticated as far as Postgres is concerned and get zero
# rows back from any RLS-protected table, even ones that objectively exist
# -- that mismatch, not a real "profile missing"/"wrong password" problem,
# was the cause of sign-up and login both failing.
# ----------------------------------------------------------------------------
_ACCESS_TOKEN: Optional[str] = None
_REFRESH_TOKEN: Optional[str] = None


def set_session(access_token: Optional[str], refresh_token: Optional[str] = None) -> None:
    """Make every subsequent REST call in this process authenticate as the
    user this token belongs to, instead of the anon key. Called internally
    by sign_up/log_in/refresh_session on success."""
    global _ACCESS_TOKEN, _REFRESH_TOKEN
    _ACCESS_TOKEN = access_token or None
    if refresh_token:
        _REFRESH_TOKEN = refresh_token


def clear_session() -> None:
    """Drop back to the anon key for all requests (e.g. on logout/account
    deletion). Does not revoke the refresh token server-side -- see
    revoke_session_token for the app-level equivalent of that."""
    global _ACCESS_TOKEN, _REFRESH_TOKEN
    _ACCESS_TOKEN = None
    _REFRESH_TOKEN = None


def has_active_session() -> bool:
    return _ACCESS_TOKEN is not None


def get_refresh_token() -> Optional[str]:
    """The current user's Supabase Auth refresh token, if any -- callers
    (see save_cached_credentials in pulse_cli.py's _auth_flow) persist this
    locally so a future run can call refresh_session() to re-establish a
    real authenticated context without asking for the password again."""
    return _REFRESH_TOKEN


def refresh_session(refresh_token: Optional[str]) -> bool:
    """Exchange a previously-issued refresh_token for a fresh access_token
    without needing the password again. Meant to be called once at the
    start of a run when there's a cached login, BEFORE any profile/team/
    session call, so those calls run authenticated instead of falling back
    to the anon key. Returns False (never raises) on a missing/expired/
    invalid refresh token -- caller should fall back to a fresh interactive
    login in that case, same graceful-degradation pattern as the rest of
    this module."""
    if not refresh_token:
        return False
    body = {"refresh_token": refresh_token}
    try:
        url = f"{_AUTH_URL}/token?grant_type=refresh_token"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST", headers=_headers())
        with urllib.request.urlopen(req, timeout=_TIMEOUT_INTERACTIVE) as resp:
            auth_response = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError):
        return False
    if not auth_response or "access_token" not in auth_response:
        return False
    set_session(auth_response.get("access_token"), auth_response.get("refresh_token"))
    return True


def _headers(prefer: Optional[str] = None) -> Dict[str, str]:
    h = {
        "apikey": SUPABASE_KEY,
        # apikey identifies the project to the gateway; Authorization is
        # what Postgres/PostgREST actually evaluate `auth.uid()` from for
        # RLS. Use the signed-in user's real JWT once we have one instead
        # of the anon key for every request.
        "Authorization": f"Bearer {_ACCESS_TOKEN or SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    if prefer:
        h["Prefer"] = prefer
    return h


def _request(
    method: str,
    path: str,
    params: Optional[Dict[str, str]] = None,
    body: Optional[Any] = None,
    prefer: Optional[str] = None,
    timeout: float = _TIMEOUT_INTERACTIVE,
) -> Any:
    """Raw PostgREST call. Raises SupabaseError on any failure (network,
    HTTP status, or bad JSON) -- callers decide whether that's fatal
    (auth/team setup) or just skipped (background session sync)."""
    url = f"{_REST_URL}/{path.lstrip('/')}"
    if params:
        # PostgREST operator prefixes (eq., cs., select list commas, and
        # the {..} array-literal syntax) need to survive unescaped; only
        # the actual values are percent-encoded.
        qs = "&".join(f"{k}={urllib.parse.quote(v, safe='.,{}()=*')}" for k, v in params.items())
        url = f"{url}?{qs}"

    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=_headers(prefer))

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            if not raw:
                return None
            return json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SupabaseError(f"{method} {path} -> HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise SupabaseError(f"{method} {path} -> network error: {exc.reason}") from exc
    except (json.JSONDecodeError, TimeoutError) as exc:
        raise SupabaseError(f"{method} {path} -> {exc}") from exc


# ----------------------------------------------------------------------------
# Input validation -- specifically for the two user-typed values that flow
# straight into a PostgREST `eq.<value>` filter (email at sign-up/login,
# join codes when joining a team): _request's URL-building (see its
# "safe='.,{}()=*'" quoting) deliberately leaves PostgREST's own structural
# characters (`.`, `,`, `{}`, `()`) unescaped so operator prefixes and
# array-literal syntax that CALLERS construct still work -- which means an
# attacker-supplied value containing those same characters could otherwise
# ride along unescaped into the query string and change what the filter
# actually matches. Rather than rework every call site to separately quote
# prefix vs. value, the practical fix for a email/password app with no
# custom backend of its own is to constrain these two fields to a safe
# character set at the one point they're ever created (sign_up,
# create_team's join code generator), and reject anything else on every
# subsequent lookup too -- so even a pre-existing row that somehow doesn't
# conform can't be used to smuggle a filter-breaking value through login.
# ----------------------------------------------------------------------------
_email_RE = re.compile(
    r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
)
_JOIN_CODE_RE = re.compile(r"^[A-Za-z0-9]{4,24}$")


def is_valid_email(email: str) -> bool:
    return bool(email) and bool(_email_RE.match(email))


def is_valid_join_code(code: str) -> bool:
    return bool(code) and bool(_JOIN_CODE_RE.match(code))


# ----------------------------------------------------------------------------
# Session tokens -- the real fix for the "cached user_id alone is trusted
# forever" gap noted in the SECURITY NOTE above. On a successful
# sign_up/log_in, a random per-device token is generated; only its SHA-256
# hash is ever sent to Supabase (stored in profiles.session_token_hash), the
# raw token stays local (in credentials.json, chmod 600 -- see
# save_cached_credentials). A cached login is only trusted on a future run
# if the locally-held raw token still hashes to what the server has on
# file, which a stolen credentials.json alone (without ALSO reading the
# token out of it) can't satisfy, and which a revoked/rotated token (see
# revoke_session_token) invalidates immediately server-side.
#
# This degrades gracefully on any deployment that hasn't added the
# session_token_hash column yet: attach/verify both detect the "unknown
# column" response from PostgREST and return a clear "unsupported"
# signal (rather than raising), and callers fall back to the old
# TTL-only trust model in that case -- see _auth_flow in pulse_cli.py.
#
# To enable this, add the column once in the Supabase SQL editor:
#   ALTER TABLE "profiles" ADD COLUMN session_token_hash text;
# ----------------------------------------------------------------------------

class SessionTokenUnsupported(Exception):
    """Raised internally when the deployment doesn't have
    profiles.session_token_hash yet -- callers catch this specifically and
    fall back, rather than treating it like any other SupabaseError."""


def _hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _is_unknown_column_error(exc: SupabaseError) -> bool:
    msg = str(exc)
    return "PGRST204" in msg or "column" in msg.lower() and ("does not exist" in msg.lower() or "not found" in msg.lower())


def attach_session_token(user_id: str) -> Optional[str]:
    """Generate a new session token for `user_id`, store its hash
    server-side, and return the raw token for the caller to cache
    locally. Returns None (never raises) if the deployment doesn't
    support this yet -- see the migration note above -- so a fresh
    install without the column keeps working exactly as before."""
    token = secrets.token_hex(32)
    try:
        _request(
            "PATCH", "profiles",
            params={"id": f"eq.{user_id}"},
            body={"session_token_hash": _hash_session_token(token)},
            prefer="return=minimal",
        )
        return token
    except SupabaseError as exc:
        if _is_unknown_column_error(exc):
            return None
        return None  # any other failure -- also non-fatal, just means no session token this run


def verify_session_token(user_id: str, token: Optional[str]) -> Optional[bool]:
    """True if `token` matches what's stored for `user_id`, False if it
    doesn't (cached login should NOT be trusted -- force a fresh login),
    or None if this deployment doesn't support session tokens at all
    (caller should fall back to the old TTL-only trust model rather than
    treating None as a mismatch)."""
    if not token:
        return None
    try:
        rows = _request(
            "GET", "profiles",
            params={"id": f"eq.{user_id}", "select": "session_token_hash"},
        )
    except SupabaseError as exc:
        if _is_unknown_column_error(exc):
            return None
        return None  # network hiccup etc. -- don't lock the user out, just skip verification this run
    if not rows or "session_token_hash" not in (rows[0] or {}):
        return None
    stored_hash = rows[0].get("session_token_hash")
    if not stored_hash:
        return None  # column exists but no token attached yet (e.g. pre-upgrade account)
    return hmac.compare_digest(_hash_session_token(token), stored_hash)


def revoke_session_token(user_id: str) -> None:
    """Invalidate any cached login for this user everywhere (e.g. on
    /logout, or if the user suspects a leaked credentials.json) by
    clearing the server-side hash -- no future verify_session_token call
    for any copy of the old token can succeed after this. Best-effort;
    never raises."""
    try:
        _request(
            "PATCH", "profiles",
            params={"id": f"eq.{user_id}"},
            body={"session_token_hash": None},
            prefer="return=minimal",
        )
    except SupabaseError:
        pass


# ----------------------------------------------------------------------------
# Local credential cache (~/.pulse/credentials.json)
# ----------------------------------------------------------------------------

def load_cached_credentials() -> Optional[Dict[str, str]]:
    """Returns the cached login, or None if there's no cache, it's
    malformed, or it's aged past SESSION_TTL_DAYS. The TTL is a partial
    mitigation, not real session security -- see SESSION_TTL_DAYS above."""
    try:
        if not CACHE_PATH.exists():
            return None
        data = json.loads(CACHE_PATH.read_text())
        if not (data.get("user_id") and data.get("email")):
            return None
        cached_at = data.get("cached_at")
        if cached_at and SESSION_TTL_DAYS > 0:
            try:
                age_days = (time.time() - float(cached_at)) / 86400.0
                if age_days > SESSION_TTL_DAYS:
                    return None  # expired -- caller falls through to a fresh login
            except (TypeError, ValueError):
                pass
        return data
    except Exception:
        pass
    return None


def save_cached_credentials(
    user_id: str,
    email: str,
    team_id: Optional[str] = None,
    session_token: Optional[str] = None,
    refresh_token: Optional[str] = None,
) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {"user_id": user_id, "email": email, "team_id": team_id, "cached_at": time.time()}
        # refresh_token defaults to the current in-memory session's if the
        # caller didn't pass one explicitly -- covers call sites that just
        # want to re-save (e.g. to bump team_id) without re-deriving it.
        refresh_token = refresh_token or get_refresh_token()
        if session_token:
            payload["session_token"] = session_token
        if refresh_token:
            payload["refresh_token"] = refresh_token
        if not session_token or not refresh_token:
            # Preserve whichever of session_token/refresh_token wasn't just
            # passed in, rather than dropping it, when there's an existing
            # cache on disk to preserve it from.
            if CACHE_PATH.exists():
                try:
                    existing = json.loads(CACHE_PATH.read_text())
                    if not session_token and existing.get("session_token"):
                        payload["session_token"] = existing["session_token"]
                    if not refresh_token and existing.get("refresh_token"):
                        payload["refresh_token"] = existing["refresh_token"]
                except Exception:
                    pass
        CACHE_PATH.write_text(json.dumps(payload, indent=2))
        try:
            os.chmod(CACHE_PATH, 0o600)  # never store even a hash for anyone else to read
        except OSError:
            pass
    except Exception:
        pass  # local caching is a convenience, never a hard requirement


def clear_cached_credentials() -> None:
    clear_session()
    try:
        if CACHE_PATH.exists():
            CACHE_PATH.unlink()
    except Exception:
        pass


# ----------------------------------------------------------------------------
# Local run profile (~/.pulse/profile.json) -- remembers the *non-secret*
# shape of the last run so the CLI doesn't re-prompt/re-detect from
# scratch every time: which provider was used (never the API key itself --
# that stays in the env var / OS keychain-adjacent process env only, and
# is still re-read from PULSE_*_API_KEY / re-entered via getpass), the
# last-synced git commit sha, and the last-detected GPU. Deliberately a
# separate file from credentials.json (auth) so /logout or a credentials
# TTL expiry doesn't also wipe these unrelated preferences.
# ----------------------------------------------------------------------------

def load_cached_profile() -> Dict[str, Any]:
    try:
        if not PROFILE_PATH.exists():
            return {}
        data = json.loads(PROFILE_PATH.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_cached_profile(**fields: Any) -> None:
    """Merge `fields` into the existing profile cache and persist it.
    Pass only the keys that changed -- existing keys are preserved. NEVER
    pass an API key/secret here; this file is not chmod-restricted any
    more tightly than credentials.json and is meant purely for
    non-sensitive defaults (provider name, env var *name*, commit sha,
    GPU description)."""
    try:
        current = load_cached_profile()
        current.update({k: v for k, v in fields.items() if v is not None})
        PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROFILE_PATH.write_text(json.dumps(current, indent=2))
        try:
            os.chmod(PROFILE_PATH, 0o600)
        except OSError:
            pass
    except Exception:
        pass  # local caching is a convenience, never a hard requirement


# ----------------------------------------------------------------------------
# Telemetry opt-out
# ----------------------------------------------------------------------------

def telemetry_enabled() -> bool:
    """PULSE_TELEMETRY=off/0/false/no disables environment-info collection
    and reporting entirely (checked at every call site that would gather
    or send it, not just once at startup, so /telemetry off takes effect
    immediately mid-run too)."""
    return os.environ.get("PULSE_TELEMETRY", "on").strip().lower() not in ("off", "0", "false", "no")


# ----------------------------------------------------------------------------
# Users
# ----------------------------------------------------------------------------

def find_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Fetch user profile by email from the profiles table."""
    if not is_valid_email(email):
        return None
    rows = _request(
        "GET", "profiles",
        params={"email": f"eq.{email}", "select": "id,email,personal_plan"},
    )
    return rows[0] if rows else None


_DEFAULT_PERSONAL_PLAN = "free"

def sign_up(email: str, password: str, plan: Optional[str] = None) -> Dict[str, Any]:
    """Sign up a new user via Supabase Auth. Email is generated from email.
    The profile is automatically created by the Supabase trigger.
    Returns the user profile from the profiles table."""
    if not is_valid_email(email):
        raise SupabaseError(
            "Enter a valid email address (min. 8 characters), e.g. name@example.com."
        )
    if len(password) < 8:
        raise SupabaseError("Password must be at least 8 characters.")
    if find_user_by_email(email):
        raise SupabaseError(f"Email '{email}' is already taken.")
    
    # Store the original email input, then map it to the placeholder format
    original_email = email
    auth_email = f"{email}"
    
    body = {
        "email": auth_email,
        "password": password,
        "data": {"email": original_email},  # <-- FIX: Changed "user_metadata" to "data"
    }
    
    try:
        url = f"{_AUTH_URL}/signup"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST", headers=_headers())
        with urllib.request.urlopen(req, timeout=_TIMEOUT_INTERACTIVE) as resp:
            auth_response = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SupabaseError(f"POST /auth/v1/signup -> HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise SupabaseError(f"POST /auth/v1/signup -> network error: {exc.reason}") from exc
    
    if not auth_response or "user" not in auth_response:
        raise SupabaseError("Sign up succeeded but no user was returned.")
    
    access_token = auth_response.get("access_token")
    refresh_token = auth_response.get("refresh_token")

    if not access_token:
        # This project's Auth settings require email confirmation, so
        # there's no session yet. The trigger has already created the
        # profiles row at this point, but it can't be read back until the
        # user confirms their email and logs in for real -- surface that
        # plainly instead of the misleading "profile was not created"
        # below (which is what happens if you try to fetch it anyway).
        raise SupabaseEmailConfirmationRequired(
            "Confirm your account: check your email for a confirmation link, then log in."
        )

    # Authenticate every subsequent request in this process (starting with
    # the profile fetch right below) as this user instead of the anon key.
    # Without this, the SELECT below is unauthenticated as far as RLS is
    # concerned and comes back empty even though the row exists -- that's
    # what used to surface as "Profile was not created after signup."
    set_session(access_token, refresh_token)

    # Wait a brief moment for the trigger to create the profile
    time.sleep(0.5)
    
    # Fetch the created profile by email -- NOT by matching auth id to
    # profiles.id. This deployment's signup trigger doesn't guarantee
    # profiles.id == auth.users.id, so email is the one column that
    # reliably ties the two together.
    profile = find_user_by_email(original_email)
    if not profile:
        raise SupabaseError(
            "Profile was not created after signup. If a profiles row does exist "
            "for this email, check for a missing/incorrect Row Level Security "
            "SELECT policy on 'profiles' (e.g. `create policy \"select own "
            "profile\" on profiles for select using (auth.uid() = id);` -- or a "
            "more permissive policy if profiles.id doesn't match the auth id)."
        )
    
    return profile

def log_in(email: str, password: str) -> Dict[str, Any]:
    """Log in by email/password via Supabase Auth. Returns the user profile.

    Authenticates FIRST, then looks the profile up by email using the
    resulting JWT -- not the other way around, and not by id. The
    previous version looked the profile up (via find_user_by_email, an
    anon-key GET on the RLS-protected profiles table) before ever
    checking the password, which always came back empty under RLS
    regardless of the password. It's looked up by email rather than by
    matching the auth id to profiles.id because this deployment's signup
    trigger doesn't guarantee those two ids are the same value."""
    if not is_valid_email(email):
        raise SupabaseError("Invalid email or password.")  # Don't leak format rules

    body = {"email": email, "password": password}

    try:
        # Use urllib directly for Auth API token endpoint
        url = f"{_AUTH_URL}/token?grant_type=password"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST", headers=_headers())
        with urllib.request.urlopen(req, timeout=_TIMEOUT_INTERACTIVE) as resp:
            auth_response = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError):
        raise SupabaseError("Invalid email or password.")

    if not auth_response or "user" not in auth_response or "access_token" not in auth_response:
        raise SupabaseError("Invalid email or password.")

    # Authenticate every subsequent request in this process (starting with
    # the profile fetch right below) as this user instead of the anon key
    # -- see set_session/_headers.
    set_session(auth_response.get("access_token"), auth_response.get("refresh_token"))

    auth_email = auth_response["user"].get("email") or email
    profile = find_user_by_email(auth_email)
    if not profile:
        raise SupabaseError(
            "Signed in, but no matching profile was found for this account. If a "
            "profiles row does exist for this user, this is almost always a "
            "missing/incorrect Row Level Security SELECT policy on 'profiles' -- "
            "e.g. `create policy \"select own profile\" on profiles for select "
            "using (auth.uid() = id);`, or the trigger that creates the row isn't "
            "setting profiles.id to the same uuid as the auth user's id."
        )

    return profile


def change_password(user_id: str, current_password: str, new_password: str) -> None:
    """Change password via Supabase Auth. Requires the current password for verification.
    Also revokes all other session tokens after password change."""
    if not is_valid_uuid(user_id):
        raise SupabaseError("Invalid account.")
    if len(new_password) < 8:
        raise SupabaseError("New password must be at least 8 characters.")
    
    # Fetch the profile to get the email
    profile = fetch_user(user_id)
    if not profile:
        raise SupabaseError("User not found.")
    
    email = profile.get("email")
    if not email:
        raise SupabaseError("Invalid account.")
    
    # Verify current password by attempting to authenticate
    email = f"{email}"
    body = {"email": email, "password": current_password}
    
    try:
        url = f"{_AUTH_URL}/token?grant_type=password"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST", headers=_headers())
        with urllib.request.urlopen(req, timeout=_TIMEOUT_INTERACTIVE) as resp:
            _ = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError):
        raise SupabaseError("Current password is incorrect.")
    
    # Update password in Auth (requires an access token, which we don't have in this context)
    # For CLI usage, you may need to use the Admin API endpoint instead:
    # PATCH /auth/v1/admin/users/{id} with the admin JWT, or implement password reset via email
    # For now, raise an error noting this limitation
    try:
        # Attempt to update via admin API if we had access
        # This is a limitation of the current architecture
        raise SupabaseError(
            "Password change requires an authenticated session. "
            "Implement this via email verification link or admin API with proper JWT."
        )
    except SupabaseError:
        raise
    finally:
        # Revoke all session tokens
        revoke_session_token(user_id)


def fetch_user_with_password(user_id: str) -> Optional[Dict[str, Any]]:
    """Fetch user profile by ID. Note: passwords are managed by Supabase Auth,
    not stored in the profiles table."""
    if not is_valid_uuid(user_id):
        return None
    rows = _request(
        "GET", "profiles",
        params={"id": f"eq.{user_id}", "select": "id,email"},
    )
    return rows[0] if rows else None


# ----------------------------------------------------------------------------
# Account recovery -- there's no email/SMS infrastructure here (no SMTP
# config, no phone verification), so a normal "email me a reset link"
# flow isn't available. The standard fallback used by tools in the same
# position (GitHub 2FA, Signal PINs, etc.) is a recovery code: a single
# high-entropy secret shown to the user exactly once, at sign-up, that
# they're told to store somewhere safe (a password manager) -- possession
# of it substitutes for the forgotten password exactly once, after which
# it's invalidated and a new one is generated (so it can't be reused if
# it was ever exposed during that one recovery).
#
# Needs its own column -- add it once in the Supabase SQL editor:
#   ALTER TABLE "profiles" ADD COLUMN recovery_code_hash text;
# Degrades gracefully (returns None) on a deployment without it yet, same
# pattern as attach_session_token/session_token_hash above.
# ----------------------------------------------------------------------------

def _generate_recovery_code() -> str:
    # e.g. "7F3K-9QRT-2LXP" -- grouped for readability when copying it down
    raw = secrets.token_hex(6).upper()
    return "-".join(raw[i:i + 4] for i in range(0, len(raw), 4))


def attach_recovery_code(user_id: str) -> Optional[str]:
    """Generate (or regenerate) this user's recovery code, return it ONCE
    for the caller to show/save. None if this deployment doesn't have
    the column yet (see migration note above) -- sign-up/change-password
    still succeed either way, there's just no recovery code offered."""
    code = _generate_recovery_code()
    try:
        _request(
            "PATCH", "profiles",
            params={"id": f"eq.{user_id}"},
            body={"recovery_code_hash": _hash_session_token(code)},
            prefer="return=minimal",
        )
        return code
    except SupabaseError as exc:
        if _is_unknown_column_error(exc):
            return None
        return None


def recover_account(email: str, recovery_code: str, new_password: str) -> Optional[str]:
    """Reset `email`'s password using a previously-issued recovery code.
    Requires implementation of Auth API password reset or admin endpoint.
    Single-use: a new recovery code is generated after a successful recovery."""
    if not is_valid_email(email):
        raise SupabaseError("Invalid email or recovery code.")
    if len(new_password) < 8:
        raise SupabaseError("New password must be at least 8 characters.")
    
    # Fetch user profile to get user_id
    profile = find_user_by_email(email)
    if not profile:
        raise SupabaseError("Invalid email or recovery code.")
    
    user_id = profile.get("id")
    if not user_id:
        raise SupabaseError("Invalid email or recovery code.")
    
    # Check for recovery_code_hash in profile
    try:
        rows = _request(
            "GET", "profiles",
            params={"id": f"eq.{user_id}", "select": "id,recovery_code_hash"},
        )
    except SupabaseError:
        raise SupabaseError("Account recovery isn't available on this deployment yet.")
    
    if not rows or "recovery_code_hash" not in (rows[0] or {}):
        raise SupabaseError(
            "Account recovery isn't available on this deployment yet, or the email is wrong."
        )
    
    stored_hash = rows[0].get("recovery_code_hash")
    normalized = recovery_code.strip().upper()
    if not stored_hash or not hmac.compare_digest(_hash_session_token(normalized), stored_hash):
        raise SupabaseError("Invalid email or recovery code.")
    
    # Password change must be done via Auth API
    # This is a limitation: would need email verification link or admin API access
    raise SupabaseError(
        "Password reset requires verification via email link. "
        "Implement via Supabase Auth password recovery endpoint."
    )
    # After implementation, generate new recovery code:
    # return attach_recovery_code(user_id)


# ----------------------------------------------------------------------------
# Terms of Service / Privacy Policy acceptance -- most jurisdictions and
# most payment processors expect a record of WHEN a user agreed to your
# terms, not just that a checkbox existed in the UI at some point. This
# only records a timestamp against the account; see
# pulse_cli.py's _prompt_tos_acceptance for the actual prompt, and swap
# in your real ToS/Privacy Policy URLs via PULSE_TOS_URL/PULSE_PRIVACY_URL
# -- Pulse doesn't (and shouldn't) generate legal text for you.
#
# Needs its own column -- add it once in the Supabase SQL editor:
#   ALTER TABLE "profiles" ADD COLUMN tos_accepted_at timestamptz;
# Degrades gracefully (silently does nothing) on a deployment without it.
# ----------------------------------------------------------------------------

def record_tos_acceptance(user_id: str) -> None:
    if not is_valid_uuid(user_id):
        return
    try:
        _request(
            "PATCH", "profiles",
            params={"id": f"eq.{user_id}"},
            body={"tos_accepted_at": datetime.now(timezone.utc).isoformat()},
            prefer="return=minimal",
        )
    except SupabaseError:
        pass  # missing column or any other issue -- never block sign-up over this


# ----------------------------------------------------------------------------
# Account deletion -- required in most jurisdictions selling to teams
# (GDPR's right to erasure and equivalents elsewhere) to have SOME way
# for a user to delete their own account and data, not just an admin
# doing it by hand in the Supabase dashboard. Removes the user from every
# team's members/admin_ids (so a deleted user doesn't linger as a ghost
# member), revokes tokens/sessions, then deletes the profiles row itself.
# Does NOT delete Debug_Sessions rows the user created for a team they
# were on -- those are the TEAM's training history, not solely the
# individual's, and other members/admins still have a legitimate reason
# to see that history after this person leaves. If full erasure of
# authored content is a hard requirement for you, that's a product
# decision to make deliberately, not a side effect of account deletion.
# ----------------------------------------------------------------------------

def delete_account(user_id: str, password: str) -> None:
    """Delete user account via Auth. Requires password for confirmation.
    Profile will cascade delete when auth user is deleted.
    Removes user from all teams first."""
    if not is_valid_uuid(user_id):
        raise SupabaseError("Invalid account.")
    
    # Verify password by attempting authentication
    profile = fetch_user(user_id)
    if not profile:
        raise SupabaseError("User not found.")
    
    email = profile.get("email")
    if not email:
        raise SupabaseError("Invalid account.")
    
    # Verify password directly against the Auth API (this is NOT a
    # PostgREST table, so it must NOT go through _request/_REST_URL --
    # see log_in() above for the same pattern).
    email = f"{email}"
    body = {"email": email, "password": password}

    try:
        url = f"{_AUTH_URL}/token?grant_type=password"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST", headers=_headers())
        with urllib.request.urlopen(req, timeout=_TIMEOUT_INTERACTIVE) as resp:
            _ = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError):
        raise SupabaseError("Password is incorrect.")

    # Remove user from all teams
    for team in find_teams_for_user(user_id):
        new_members = [m for m in (team.get("members") or []) if m != user_id]
        new_admins = [a for a in (team.get("admin_ids") or []) if a != user_id]
        try:
            _request(
                "PATCH", "Teams",
                params={"team_id": f"eq.{team['team_id']}"},
                body={"members": new_members, "admin_ids": new_admins},
                prefer="return=minimal",
            )
        except SupabaseError:
            pass  # best-effort -- still proceed to delete the account below

    # Delete profile (profile->auth.users cascade will handle auth deletion)
    try:
        _request("DELETE", "profiles", params={"id": f"eq.{user_id}"}, prefer="return=minimal")
    except SupabaseError:
        # If profile delete fails, try to delete the auth user directly via admin API
        # This requires proper JWT admin token setup
        pass


def fetch_user(user_id: str) -> Optional[Dict[str, Any]]:
    if not is_valid_uuid(user_id):
        return None
    rows = _request("GET", "profiles", params={"id": f"eq.{user_id}", "select": "id,email,personal_plan"})
    return rows[0] if rows else None


# ----------------------------------------------------------------------------
# Teams
# ----------------------------------------------------------------------------

def _generate_join_code() -> str:
    return secrets.token_hex(4).upper()  # e.g. "A1B2C3D4"


def git_remote_url(cwd: Optional[str] = None) -> Optional[str]:
    """Best-effort `git remote get-url origin`, run in `cwd` (the tracked
    script's own directory, not necessarily the process's cwd) so this
    works when Pulse is launched from outside the repo."""
    try:
        out = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=3, cwd=cwd,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


_TEAM_SELECT = "team_id,members,admin_ids,join_code,plan,owner_id,repo"


def find_team_by_join_code(code: str) -> Optional[Dict[str, Any]]:
    if not is_valid_join_code(code):
        return None
    rows = _request(
        "GET", "Teams",
        params={"join_code": f"eq.{code}", "select": _TEAM_SELECT},
    )
    return rows[0] if rows else None


_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


def is_valid_uuid(value: str) -> bool:
    return bool(value) and bool(_UUID_RE.match(value))


def find_teams_for_user(user_id: str) -> List[Dict[str, Any]]:
    """Teams whose members array contains this user_id."""
    if not is_valid_uuid(user_id):
        return []
    rows = _request(
        "GET", "Teams",
        params={
            "members": f"cs.{{{user_id}}}",  # Postgres array "contains" filter
            "select": _TEAM_SELECT,
        },
    )
    return rows or []


def create_team(
    owner_id: str,
    plan: Optional[str] = None,
    repo: Optional[str] = None,
    cwd: Optional[str] = None,
) -> Dict[str, Any]:
    body = {
        "members": [owner_id],
        "admin_ids": [owner_id],  # the creator is always an admin of their own team
        "join_code": _generate_join_code(),
        "plan": plan or _DEFAULT_PERSONAL_PLAN,
        "owner_id": owner_id,
        "repo": repo or git_remote_url(cwd) or "unknown",
    }
    rows = _request("POST", "Teams", body=body, prefer="return=representation")
    if not rows:
        raise SupabaseError("Team creation succeeded but no row was returned.")
    return rows[0]


def update_team_repo(team_id: str, repo: str) -> None:
    _request(
        "PATCH", "Teams",
        params={"team_id": f"eq.{team_id}"},
        body={"repo": repo, "updated_at": datetime.now(timezone.utc).isoformat()},
        prefer="return=minimal",
    )


def update_team_saved_vars(team_id: str, saved_vars: List[str]) -> None:
    """Sync the team's shared GPU-tracked variable list (Teams.saved_vars)
    so teammates see the same set. `/gputrack` and the agent's
    GPUTRACK:/GPUUNTRACK: directives call this after changing
    gpu_tracked_vars locally.
    """
    _request(
        "PATCH", "Teams",
        params={"team_id": f"eq.{team_id}"},
        body={"saved_vars": list(saved_vars), "updated_at": datetime.now(timezone.utc).isoformat()},
        prefer="return=minimal",
    )


def join_team(join_code: str, user_id: str) -> Dict[str, Any]:
    team = find_team_by_join_code(join_code)
    if not team:
        raise SupabaseError(f"No team found with join code '{join_code}'.")
    members = list(team.get("members") or [])
    if user_id not in members:
        members.append(user_id)
        _request(
            "PATCH", "Teams",
            params={"team_id": f"eq.{team['team_id']}"},
            body={"members": members, "updated_at": datetime.now(timezone.utc).isoformat()},
            prefer="return=minimal",
        )
        team["members"] = members
    return team


def is_team_admin(team: Dict[str, Any], user_id: str) -> bool:
    return user_id in (team.get("admin_ids") or [])


def add_team_admin(team_id: str, user_id: str, members: Optional[List[str]] = None) -> Dict[str, Any]:
    """Grant `user_id` admin rights on the team. Enforces that admins must
    already be team members -- add them as a member first (join_team)
    before promoting them.
    """
    team = _request(
        "GET", "Teams", params={"team_id": f"eq.{team_id}", "select": _TEAM_SELECT}
    )
    team = team[0] if team else None
    if not team:
        raise SupabaseError(f"No team found with id '{team_id}'.")

    current_members = members if members is not None else list(team.get("members") or [])
    if user_id not in current_members:
        raise SupabaseError("Can't make an admin out of someone who isn't a team member yet -- add them to the team first.")

    admin_ids = list(team.get("admin_ids") or [])
    if user_id not in admin_ids:
        admin_ids.append(user_id)
        _request(
            "PATCH", "Teams",
            params={"team_id": f"eq.{team_id}"},
            body={"admin_ids": admin_ids, "updated_at": datetime.now(timezone.utc).isoformat()},
            prefer="return=minimal",
        )
    team["admin_ids"] = admin_ids
    return team


def remove_team_admin(team_id: str, user_id: str) -> Dict[str, Any]:
    """Revoke `user_id`'s admin rights on the team. Leaves their regular
    membership untouched -- this only removes them from admin_ids."""
    team = _request(
        "GET", "Teams", params={"team_id": f"eq.{team_id}", "select": _TEAM_SELECT}
    )
    team = team[0] if team else None
    if not team:
        raise SupabaseError(f"No team found with id '{team_id}'.")

    admin_ids = [uid for uid in (team.get("admin_ids") or []) if uid != user_id]
    _request(
        "PATCH", "Teams",
        params={"team_id": f"eq.{team_id}"},
        body={"admin_ids": admin_ids, "updated_at": datetime.now(timezone.utc).isoformat()},
        prefer="return=minimal",
    )
    team["admin_ids"] = admin_ids
    return team


# ----------------------------------------------------------------------------
# Debug_Sessions
# ----------------------------------------------------------------------------

def current_git_commit_sha(cwd: Optional[str] = None) -> Optional[str]:
    """Best-effort `git rev-parse HEAD`, run in `cwd` (the tracked script's
    own directory) so the right repo's sha gets logged even when Pulse is
    launched from elsewhere (a different cwd, a notebook, etc.)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=3, cwd=cwd,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def create_debug_session(
    team_id: Optional[str], user_id: Optional[str], git_commit_sha: Optional[str] = None,
) -> Optional[str]:
    body = {
        "team_id": team_id,
        "user_id": user_id,
        "git_commit_sha": git_commit_sha or "unknown",
        "agent_logs": [],
        "error_tracebacks": [],
        "telemetry": [],
        "incidents": [],
        "uptime_seconds": 0,
        "downtime_seconds": 0,
    }
    rows = _request("POST", "Debug_Sessions", body=body, prefer="return=representation")
    if not rows:
        return None
    return rows[0]["id"]


_SESSION_SELECT = "id,created_at,git_commit_sha,agent_logs,error_tracebacks,telemetry,incidents,uptime_seconds,downtime_seconds"
_SESSION_SELECT_NO_CREATED_AT = "id,git_commit_sha,agent_logs,error_tracebacks,telemetry,incidents,uptime_seconds,downtime_seconds"
_SESSION_SELECT_LEGACY = "id,created_at,git_commit_sha,agent_logs,error_tracebacks,telemetry"  # pre-dashboard-columns deployments


def fetch_recent_sessions(
    team_id: Optional[str], user_id: Optional[str], max_rows: int = 50,
) -> List[Dict[str, Any]]:
    """Debug_Sessions rows for this team (falling back to just this user
    if there's no team), most recent first via created_at (server-side
    order + limit, so this scales instead of pulling every row ever
    logged). Falls back through progressively older select lists so this
    keeps working on a deployment that hasn't added the uptime_seconds/
    downtime_seconds/incidents columns yet (see the Teams.admin_ids-style
    manual column addition needed for the web dashboard).
    """
    if team_id:
        params = {"team_id": f"eq.{team_id}"}
    elif user_id:
        params = {"user_id": f"eq.{user_id}"}
    else:
        return []
    params["order"] = "created_at.desc"
    params["limit"] = str(max_rows)
    for select in (_SESSION_SELECT, _SESSION_SELECT_LEGACY):
        try:
            params["select"] = select
            return _request("GET", "Debug_Sessions", params=params, timeout=_TIMEOUT_INTERACTIVE) or []
        except SupabaseError:
            continue
    # Last resort: no created_at either (caller sorts client-side by the
    # `t` timestamps stamped into agent_logs/telemetry instead).
    params.pop("order", None)
    params["select"] = _SESSION_SELECT_NO_CREATED_AT
    try:
        return _request("GET", "Debug_Sessions", params=params, timeout=_TIMEOUT_INTERACTIVE) or []
    except SupabaseError:
        params["select"] = "id,git_commit_sha,agent_logs,error_tracebacks,telemetry"
        return _request("GET", "Debug_Sessions", params=params, timeout=_TIMEOUT_INTERACTIVE) or []


def fetch_debug_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a single Debug_Sessions row by id (fields still
    compressed -- caller decodes with decode_entries).

    Used when resuming an EXISTING debug session -- currently only after
    an auto-fix restart, which carries the same session id into the new
    process via PULSE_AUTO_SESSION_ID -- so the resumed process can
    preload its local agent_logs/error_tracebacks/telemetry/incidents
    lists with whatever's already saved server-side. Without this, a
    freshly-started process's lists start empty and, since a PATCH
    replaces the whole array column (PostgREST has no array "append"
    verb -- see _build_cloud_patch_body), its very first flush after
    the restart would silently overwrite the pre-restart history with
    just the handful of entries logged since. Falls back to the legacy
    select list for deployments that don't have the incidents/uptime/
    downtime columns yet.
    """
    try:
        rows = _request(
            "GET", "Debug_Sessions",
            params={"id": f"eq.{session_id}", "select": _SESSION_SELECT},
        )
    except SupabaseError:
        try:
            rows = _request(
                "GET", "Debug_Sessions",
                params={"id": f"eq.{session_id}", "select": _SESSION_SELECT_LEGACY},
            )
        except SupabaseError:
            return None
    if not rows:
        return None
    return rows[0]


def patch_debug_session(session_id: str, fields: Dict[str, Any], timeout: float = _TIMEOUT_BACKGROUND) -> None:
    _request(
        "PATCH", "Debug_Sessions",
        params={"id": f"eq.{session_id}"},
        body=fields,
        prefer="return=minimal",
        timeout=timeout,
    )


# ----------------------------------------------------------------------------
# Environment info -- gathered once at session start, pushed as the first
# Debug_Sessions.telemetry entry (see PulseCLI._cloud_setup).
# ----------------------------------------------------------------------------

def _gpu_info() -> Dict[str, Any]:
    """GPU name/count/CUDA-or-ROCm version, detected without assuming any
    one array library is installed. Order of preference:
      1. `nvidia-smi` / `rocm-smi` -- these see the hardware directly and
         work no matter which (if any) of Pulse's five supported array
         backends (torch, tensorflow, jax, cupy, mlx) the project uses.
      2. Each backend's own device API, purely as a fallback/cross-check
         for environments where the CLI tools aren't on PATH (some
         containers) -- every one of the five is tried independently and
         none is required or treated as primary.
    """
    info: Dict[str, Any] = {"gpu_name": None, "gpu_count": 0, "cuda_version": None, "vendor": None}

    # -- 1a. NVIDIA, via nvidia-smi --
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            names = [n.strip() for n in out.stdout.strip().splitlines() if n.strip()]
            if names:
                info["gpu_count"] = len(names)
                info["gpu_name"] = names[0]
                info["vendor"] = "nvidia"
    except Exception:
        pass

    if info["gpu_name"]:
        try:
            out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
            if out.returncode == 0:
                m = re.search(r"CUDA Version:\s*([\d.]+)", out.stdout)
                if m:
                    info["cuda_version"] = m.group(1)
        except Exception:
            pass

    # -- 1b. AMD, via rocm-smi (only if nvidia-smi found nothing) --
    if not info["gpu_name"]:
        try:
            out = subprocess.run(["rocm-smi", "--showproductname"], capture_output=True, text=True, timeout=5)
            if out.returncode == 0 and out.stdout.strip():
                names = re.findall(r"Card series:\s*(.+)", out.stdout)
                if names:
                    info["gpu_count"] = len(names)
                    info["gpu_name"] = names[0].strip()
                    info["vendor"] = "amd"
        except Exception:
            pass
        if info["gpu_name"] and not info["cuda_version"]:
            try:
                out = subprocess.run(["rocm-smi", "--showdriverversion"], capture_output=True, text=True, timeout=5)
                if out.returncode == 0:
                    m = re.search(r"Driver Version:\s*([\w.]+)", out.stdout)
                    if m:
                        info["cuda_version"] = f"rocm-{m.group(1)}"
            except Exception:
                pass

    # -- 2. Backend-specific fallback/cross-check, tried independently --
    def _merge(name: Optional[str], count: int, vendor: Optional[str] = None) -> None:
        info["gpu_count"] = max(info["gpu_count"], count)
        if not info["gpu_name"] and name:
            info["gpu_name"] = name
        if not info["vendor"] and vendor:
            info["vendor"] = vendor

    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            _merge(torch.cuda.get_device_name(0) if n else None, n, "nvidia")
            if not info["cuda_version"]:
                info["cuda_version"] = torch.version.cuda
    except Exception:
        pass

    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            _merge(gpus[0].name, len(gpus))
    except Exception:
        pass

    try:
        import jax
        devs = [d for d in jax.devices() if getattr(d, "platform", "") in ("gpu", "tpu")]
        if devs:
            _merge(f"{devs[0].platform}:{getattr(devs[0], 'device_kind', '?')}", len(devs))
    except Exception:
        pass

    try:
        import cupy
        n = cupy.cuda.runtime.getDeviceCount()
        if n:
            name = None
            try:
                props = cupy.cuda.runtime.getDeviceProperties(0)
                raw_name = props.get("name")
                name = raw_name.decode() if isinstance(raw_name, bytes) else raw_name
            except Exception:
                pass
            _merge(name, n, "nvidia")
    except Exception:
        pass

    try:
        import mlx.core as mx
        if mx.metal.is_available():
            _merge("Apple Metal (MLX)", 1, "apple")
    except Exception:
        pass

    return info


def _framework_versions() -> Dict[str, str]:
    """Reports whichever of Pulse's five supported array backends (plus
    numpy) are actually importable -- not every project has all, or any,
    of them."""
    versions: Dict[str, str] = {}
    for mod_name in ("torch", "tensorflow", "jax", "cupy", "mlx", "numpy"):
        try:
            mod = __import__(mod_name)
        except Exception:
            continue
        v = getattr(mod, "__version__", None)
        if v:
            versions[mod_name] = str(v)
    return versions


def collect_environment_info() -> Dict[str, Any]:
    """Gathered once, at the start of a session. Every field is
    best-effort -- no GPU, or a missing library, just leaves that field
    None/empty rather than raising."""
    gpu = _gpu_info()
    return {
        "type": "env_info",
        "t": time.time(),
        "gpu_name": gpu.get("gpu_name"),
        "gpu_count": gpu.get("gpu_count"),
        "cuda_version": gpu.get("cuda_version"),
        "python_version": platform.python_version(),
        "framework_versions": _framework_versions(),
    }


# ----------------------------------------------------------------------------
# Background sync worker
# ----------------------------------------------------------------------------

class BackgroundSync:
    """Runs Debug_Sessions PATCHes on a single daemon thread so agent-log,
    traceback, and telemetry syncing never blocks the training loop or the
    interactive prompt on network latency. Best-effort: failures are
    reported at most once per kind, then swallowed."""

    def __init__(self, on_error=None):
        self._q: "queue.Queue" = queue.Queue()
        self._on_error = on_error
        self._warned = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, session_id: str, fields: Dict[str, Any], wait: bool = False):
        done = threading.Event() if wait else None
        self._q.put((session_id, fields, done))
        return done

    def _run(self) -> None:
        while True:
            session_id, fields, done = self._q.get()
            try:
                if fields:
                    patch_debug_session(session_id, fields)
            except SupabaseError as exc:
                if not self._warned and self._on_error:
                    self._warned = True
                    self._on_error(str(exc))
            finally:
                if done is not None:
                    done.set()
