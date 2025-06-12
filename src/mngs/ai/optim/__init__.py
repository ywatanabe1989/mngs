#!/usr/bin/env python3
"""Optimizer module for MNGS AI."""

# New API
from ._optimizers import get_optimizer, set_optimizer, RANGER_AVAILABLE

# Legacy API (deprecated)
from ._get_set import get, set

__all__ = ["get_optimizer", "set_optimizer", "get", "set", "RANGER_AVAILABLE"]
