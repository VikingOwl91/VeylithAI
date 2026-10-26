"""Force the whole test session onto the CPU.

Individual tests also pass ``--device cpu``; hiding the devices here is the
second lock. It matters on this PyTorch/ROCm build: a plain CPU ``AdamW.step``
initializes the HIP context even though nothing is computed on the GPU, so
"device cpu" alone does not keep the driver untouched.
"""

import os

os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = ""
