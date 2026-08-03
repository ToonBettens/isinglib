from __future__ import annotations

import numpy as np

__all__ = ("ATOL", "DEFAULT_FLOAT_DTYPE", "RTOL")

# Tolerances for equality checks.
RTOL = 1e-12
ATOL = 1e-12

# Authoritative default dtype. Everything numeric resolves to this unless told otherwise.
DEFAULT_FLOAT_DTYPE = np.dtype(np.float64)
