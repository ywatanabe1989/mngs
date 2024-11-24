#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-24 22:54:44 (ywatanabe)"
# File: ./mngs_repo/src/mngs/db/_BaseMixins/_BaseConnectionMixin.py

__file__ = "/home/ywatanabe/proj/mngs_repo/src/mngs/db/_BaseMixins/_BaseConnectionMixin.py"

# _BaseDB_modules/_BaseConnectionMixin.py
import threading
from typing import Optional

class _BaseConnectionMixin:
    def __init__(self):
        self.lock = threading.Lock()
        self._maintenance_lock = threading.Lock()
        self.conn = None
        self.cursor = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def connect(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def reconnect(self):
        raise NotImplementedError

    def execute(self):
        raise NotImplementedError

    def executemany(self):
        raise NotImplementedError


# EOF
