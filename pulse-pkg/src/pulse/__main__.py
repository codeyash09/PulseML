"""`python -m pulse ...` -- the same thing the `pulse` command does.

The installed console script is `pulse`, defined in pyproject as `pulse.cli:main`.
Anyone who has the package but not its scripts on PATH -- a venv that was not
activated, a CI image, `pip install --target` -- reaches for `python -m pulse`, and
this makes that spelling work rather than fail with "cannot be directly executed".

There is deliberately no second implementation here: one entry point, one behaviour.
"""
from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
