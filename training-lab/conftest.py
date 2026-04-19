"""Top-level conftest for training-lab tests.

macOS DYLD fix — permanent solution
------------------------------------
``tarok_engine`` is a PyO3 extension linked against libtorch
(``@rpath/libtorch_cpu.dylib``).  On macOS, the dynamic linker needs
the torch shared-library directory on ``DYLD_LIBRARY_PATH`` *before*
the extension is loaded.

Importing ``torch`` here (before any test module is collected) causes
Python to ``dlopen`` libtorch and register its path with the dynamic
linker, so that the subsequent ``import tarok_engine`` succeeds without
any extra shell-level environment variables.

This file is discovered automatically by pytest before collection.  It
works whether you run::

    pytest training-lab/tests/
    make test-lab
    uv run pytest ...

No manual ``DYLD_LIBRARY_PATH`` juggling required.
"""

import torch  # noqa: F401  — must be imported before tarok_engine
