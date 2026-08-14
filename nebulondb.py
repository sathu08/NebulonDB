#!/usr/bin/env python3
"""NebulonDB entry point.

All server-operations logic and the interactive TUI live in
``ndb_host.tui``. This module is only a thin shim so the PyPI entry
point ``nebulondb = "nebulondb:main"`` keeps working.
"""

from ndb_host.tui.app import main


if __name__ == "__main__":
    main()