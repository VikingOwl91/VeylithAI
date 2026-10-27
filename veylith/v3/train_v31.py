#!/usr/bin/env python3
"""Entry point for veylith v3.1-A training."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
