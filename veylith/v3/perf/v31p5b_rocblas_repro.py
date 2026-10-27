"""PERF-5B: the historical model reproducer with rocBLAS instead of hipBLASLt.

Only used by the matrix when the positive control establishes that the model
reproducer still faults. Everything except the BLAS library choice is the
reproducer's own behaviour: the arguments are passed through unchanged.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "debug"))

for _name in ("hipblas", "rocblas"):
    try:
        torch.backends.cuda.preferred_blas_library(_name)
        print(f"blas_backend={_name}", flush=True)
        break
    except RuntimeError:
        continue
else:
    raise SystemExit("no rocBLAS backend name accepted by this build")

import reproducer  # noqa: E402

if __name__ == "__main__":
    reproducer.main()
