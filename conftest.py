"""Repository-wide pytest configuration.

Bytecode-cache policy
---------------------
This repository keeps ALL Python bytecode caches in ONE location:
``<repo>/.pycache`` (git-ignored). ``sys.pycache_prefix`` (Python >= 3.8)
redirects every ``__pycache__`` the interpreter would otherwise scatter next to
each module into that single tree. The one exception is this conftest file
itself: its bytecode is written before this module executes, so the leftover
root ``__pycache__`` directory is removed when the pytest session finishes.

For runs outside pytest (e.g. ``python -c "import veritas"``), set the
equivalent environment variable once per shell:

    $env:PYTHONPYCACHEPREFIX = "<repo>\\.pycache"      # PowerShell
    export PYTHONPYCACHEPREFIX="<repo>/.pycache"       # bash
"""

import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
CACHE_ROOT = REPO_ROOT / ".pycache"

sys.pycache_prefix = str(CACHE_ROOT)


def pytest_sessionfinish(session, exitstatus):
    """Remove the conftest-only root ``__pycache__`` left by this session."""
    shutil.rmtree(REPO_ROOT / "__pycache__", ignore_errors=True)
