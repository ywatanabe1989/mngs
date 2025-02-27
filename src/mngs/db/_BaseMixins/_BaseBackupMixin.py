#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-24 22:15:06 (ywatanabe)"
# File: ./mngs_repo/src/mngs/db/_Basemodules/_BaseBackupMixin.py

__file__ = "/home/ywatanabe/proj/mngs_repo/src/mngs/db/_Basemodules/_BaseBackupMixin.py"

from typing import Optional

class _BaseBackupMixin:
    def backup_table(self, table: str, file_path: str):
        raise NotImplementedError

    def restore_table(self, table: str, file_path: str):
        raise NotImplementedError

    def backup_database(self, file_path: str):
        raise NotImplementedError

    def restore_database(self, file_path: str):
        raise NotImplementedError

    def copy_table(self, source_table: str, target_table: str, where: Optional[str] = None):
        raise NotImplementedError


# EOF
