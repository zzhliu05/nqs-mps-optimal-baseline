"""Consistent autodiff treatment of compensated MPS contraction scales."""
import os
from jax import lax


def contraction_scale(value):
    # Use this same value for division and log-scale compensation.
    if os.environ.get("NQS_MPS_SCALE_STOP_GRADIENT", "1") == "1":
        return lax.stop_gradient(value)
    return value
