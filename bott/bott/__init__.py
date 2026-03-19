"""
Compatibility shim for incorrect `sys.path` entries.

Some notebooks (or local execution setups) insert the directory:
`.../bott/bott` into `sys.path` instead of the repo root `.../bott`.

In that case, `import bott` would fail because the expected package
directory is `.../bott/bott/bott/`.

This shim makes `import bott.*` work by extending the package search path
to include the parent directory where the real modules (e.g. `io.py`,
`problem.py`, etc.) live.
"""

from __future__ import annotations

import os
import pkgutil

__version__ = "2025.04.15"

# Directory containing the "real" modules (io.py, utils.py, ...).
_real_pkg_dir = os.path.dirname(os.path.dirname(__file__))

# Allow `bott.io`, `bott.utils`, etc. to be found in the real directory.
__path__ = pkgutil.extend_path(__path__, __name__)
if _real_pkg_dir not in list(__path__):
    __path__.append(_real_pkg_dir)

