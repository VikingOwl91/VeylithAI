#!/usr/bin/env python3
"""Entry point for the veylith v3 data pipeline.

Run from the repository root:

    python3 veylith/v3/pipeline.py --help
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_pipeline.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
