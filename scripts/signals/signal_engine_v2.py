#!/usr/bin/env python3
"""Signal Engine v2 entrypoint.

This currently runs the active signal engine implementation from signal_engine.py.
"""

from scripts.signals.signal_engine import main


if __name__ == "__main__":
    raise SystemExit(main())
