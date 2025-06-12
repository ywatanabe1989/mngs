#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-24 22:20:26 (ywatanabe)"
# File: ./mngs_repo/src/mngs/db/_Basemodules/_BaseIndexMixin.py

THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/db/_Basemodules/_BaseIndexMixin.py"


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List


class _BaseIndexMixin:
    def create_index(
        self,
        table_name: str,
        column_names: List[str],
        index_name: str = None,
        unique: bool = False,
    ) -> None:
        raise NotImplementedError

    def drop_index(self, index_name: str) -> None:
        raise NotImplementedError


# EOF
