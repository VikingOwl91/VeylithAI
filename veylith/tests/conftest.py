"""Force the test session onto the CPU.

Same reasoning as veylith/v2/tests/conftest.py: on this PyTorch/ROCm build a
plain CPU AdamW step can initialize the HIP context, and these tests must not
touch the GPU at all.
"""

import os

os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = ""
