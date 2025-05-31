#!/usr/bin/env python3
"""Loss functions for MNGS AI."""

from .multi_task_loss import MultiTaskLoss
from ._L1L2Losses import *

__all__ = ['MultiTaskLoss']