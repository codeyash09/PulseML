"""Fake LLM for the end-to-end sweep tests: never touches the network.

Put this directory on PYTHONPATH and litellm.completion answers from a script:
  FAKE_LLM_LOG    JSONL file every call is appended to (model + messages)
  FAKE_LLM_RULES  JSON list of [substring, reply]; the first rule whose substring is in
                  the last user message wins
  FAKE_LLM_REPLY  the reply when no rule matches (default: a benign 'all fine')
  FAKE_LLM_DELAY  seconds to sleep before answering (simulates provider latency)
  FAKE_LLM_SHORTCUT=1  answer with a hand-made response object instead of going through
                  litellm.completion(mock_response=...) (the default, which runs litellm's
                  real code path -- lazy imports included -- without any network)
"""
import json
import os
import time

if os.environ.get("FAKE_DUMP_AFTER"):
    import faulthandler, sys
    import atexit, traceback

    def _dump_threads_at_exit():
        for tid, fr in sys._current_frames().items():
            sys.__stderr__.write("--- thread %s at last atexit ---\n" % tid)
            sys.__stderr__.write("".join(traceback.format_stack(fr)[:14]))
    atexit.register(_dump_threads_at_exit)

if os.environ.get("FAKE_LLM_LOG"):
    # Patched lazily, the moment litellm finishes importing, so the fake adds no import time
    # of its own to the runs being timed.
    import importlib.util
    import sys as _sys

    class _Usage:
        prompt_tokens, completion_tokens, total_tokens = 10, 5, 15

    class _Msg:
        def __init__(self, text):
            self.content = text

    class _Choice:
        def __init__(self, text):
            self.message = _Msg(text)

    class _Resp:
        def __init__(self, text):
            self.choices = [_Choice(text)]
            self.usage = _Usage()

    def _fake_completion(**kw):
        messages = kw.get("messages") or []
        last = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last = str(m.get("content") or "")
                break
        reply = os.environ.get(
            "FAKE_LLM_REPLY",
            "VERDICT: ok\nSENSITIVITY: medium\nNEXTCHECK: 100000\nNo problem found; nothing to change.")
        try:
            for sub, rep in json.loads(os.environ.get("FAKE_LLM_RULES", "[]")):
                if sub in last:
                    reply = rep
                    break
        except Exception:
            pass
        delay = float(os.environ.get("FAKE_LLM_DELAY", "0") or 0)
        if delay:
            time.sleep(delay)
        try:
            with open(os.environ["FAKE_LLM_LOG"], "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"t": time.time(), "pid": os.getpid(), "model": kw.get("model"),
                                     "last_user": last[:20000], "reply": reply,
                                     "all": "\n".join(str(m.get("content") or "") for m in messages)[-60000:]}) + "\n")
        except Exception:
            pass
        real = _REAL.get("completion")
        if real is not None and os.environ.get("FAKE_LLM_SHORTCUT") != "1":
            # Through litellm's real completion() with mock_response: the same provider
            # resolution, lazy imports and response objects as a real call, minus the network.
            kw = dict(kw)
            kw["mock_response"] = reply
            return real(**kw)
        return _Resp(reply)

    _REAL = {}

    class _PatchLitellm:
        _busy = False

        def find_spec(self, fullname, path=None, target=None):
            if fullname != "litellm" or self._busy:
                return None
            self._busy = True
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                self._busy = False
            if spec is None or spec.loader is None:
                return None
            orig = spec.loader.exec_module

            def exec_module(module, _orig=orig):
                _orig(module)
                _REAL["completion"] = module.completion
                module.completion = _fake_completion

            spec.loader.exec_module = exec_module
            return spec

    _sys.meta_path.insert(0, _PatchLitellm())
