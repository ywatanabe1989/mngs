#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-15 11:34:05 (ywatanabe)"
# File: ./mngs_repo/src/mngs/dev/_CodeAnalyzer.py

__file__ = "/home/ywatanabe/proj/mngs_repo/src/mngs/dev/_CodeAnalyzer.py"



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-15 09:52:42 (ywatanabe)"

import ast

import matplotlib.pyplot as plt
import mngs


class CodeAnalyzer:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.execution_flow = []
        self.sequence = 1
        self.skip_functions = {
            "__init__",
            "__main__",
            # Python built-ins
            "len",
            "min",
            "max",
            "sum",
            "enumerate",
            "eval",
            "print",
            "str",
            "int",
            "float",
            "list",
            "dict",
            "set",
            "tuple",
            "any",
            "all",
            "map",
            "filter",
            "zip",
            # Common DataFrame operations
            "apply",
            "unique",
            "tolist",
            "to_list",
            "rename",
            "merge",
            "set_index",
            "reset_index",
            "groupby",
            "sort_values",
            "iloc",
            "loc",
            "where",
            # NumPy operations
            "reshape",
            "squeeze",
            "stack",
            "concatenate",
            "array",
            "zeros",
            "ones",
            "full",
            "empty",
            "frombuffer",
            # Common attributes/methods
            "shape",
            "dtype",
            "size",
            "index",
            "columns",
            "values",
            "name",
            "names",
            # File operations
            "open",
            "read",
            "write",
            "close",
            # String operations
            "join",
            "split",
            "strip",
            "replace",
        }
        # self.seen_calls = set()  # Track unique function calls

    def _trace_calls(self, node, depth=0):
        if isinstance(node, ast.FunctionDef):

            if node.name in self.skip_functions:
                return

            # Track all function definitions
            self.execution_flow.append((depth, node.name, self.sequence))
            self.sequence += 1

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id

                if func_name in self.skip_functions:
                    return

                if func_name not in self.skip_functions:
                    self.execution_flow.append(
                        (depth, func_name, self.sequence)
                    )
                    self.sequence += 1

            elif isinstance(node.func, ast.Attribute):
                parts = []
                current = node.func
                while isinstance(current, ast.Attribute):
                    parts.insert(0, current.attr)
                    current = current.value
                if isinstance(current, ast.Name):
                    parts.insert(0, current.id)
                func_name = ".".join(parts)
                if not any(skip in func_name for skip in self.skip_functions):
                    self.execution_flow.append(
                        (depth, func_name, self.sequence)
                    )
                    self.sequence += 1

        for child in ast.iter_child_nodes(node):
            self._trace_calls(child, depth + 1)

    def _get_func_name(self, node):
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            parts = []
            current = node.func
            while isinstance(current, ast.Attribute):
                parts.insert(0, current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.insert(0, current.id)
            func_name = ".".join(parts)
            return (
                func_name
                if not any(skip in func_name for skip in self.skip_functions)
                else None
            )
        return None

    def _format_output(self):
        output = ["Execution Flow:"]
        last_depth = 1
        skip_until_depth = None

        filtered_flow = []
        seen_classes = set()

        for depth, call, seq in self.execution_flow:

            # Add class definition only once
            if (
                "." not in call
                and call[0].isupper()
                and call not in seen_classes
            ):  # Likely a class
                filtered_flow.append((depth, f"class {call}:", seq - 0.5))
                seen_classes.add(call)

            # Start skipping when encountering private method
            if call.startswith(("_", "self._")):
                skip_until_depth = depth
                continue

            # Skip all nested calls within private methods
            if skip_until_depth is not None and depth > skip_until_depth:
                continue
            else:
                skip_until_depth = None

            # # Skip standalone load calls as they're now under PACLoader
            # if call == "load":
            #     continue

            filtered_flow.append((depth, call, seq))
            last_depth = depth

        for depth, call, seq in filtered_flow:
            prefix = "    " * depth
            output.append(
                f"{prefix}[{int(seq) if isinstance(seq, float) else seq}] └── {call}"
            )

        return "\n".join(output)

    # # Class definition is not handled
    # def _format_output(self):
    #     output = ["Execution Flow:"]
    #     last_depth = 0
    #     skip_until_depth = None

    #     filtered_flow = []
    #     for depth, call, seq in self.execution_flow:
    #         # Start skipping when encountering private method
    #         if call.startswith(("_", "self._")):
    #             skip_until_depth = depth
    #             continue

    #         # Skip all nested calls within private methods
    #         if skip_until_depth is not None and depth > skip_until_depth:
    #             continue
    #         else:
    #             skip_until_depth = None

    #         filtered_flow.append((depth, call, seq))
    #         last_depth = depth

    #     for depth, call, seq in filtered_flow:
    #         prefix = "    " * depth
    #         output.append(f"{prefix}[{seq}] └── {call}")

    #     return "\n".join(output)

    # # Not handling nested call after private functions
    # def _format_output(self):
    #     output = ["Execution Flow:"]
    #     last_depth = 0

    #     filtered_flow = [
    #         (depth, call, seq)
    #         for depth, call, seq in self.execution_flow
    #         if not (call.startswith("_") or call.startswith("self._"))
    #         and call not in self.skip_functions
    #     ]

    #     for depth, call, seq in filtered_flow:
    #         if depth > last_depth + 1:
    #             depth = last_depth + 1
    #         last_depth = depth

    #         prefix = "    " * depth
    #         output.append(f"{prefix}[{seq}] └── {call}")

    #     return "\n".join(output)

    # def _format_output(self):
    #     output = ["Execution Flow:"]
    #     last_depth = 0

    #     for depth, call, seq in self.execution_flow:

    #         if call in self.skip_functions:
    #             continue

    #         if call.startswith("_") or call.startswith("self._"):
    #             __import__("ipdb").set_trace()

    #         # Maintain hierarchy but limit maximum depth difference
    #         if depth > last_depth + 1:
    #             depth = last_depth + 1
    #             last_depth = depth

    #         prefix = "    " * depth
    #         output.append(f"{prefix}[{seq}] └── {call}")

    #     return "\n".join(output)

    def analyze(self):
        with open(self.file_path, "r") as file:
            content = file.read()

            # Find main guard position and truncate content
            if "if __name__" in content:
                main_guard_pos = content.find("if __name__")
                content = content[:main_guard_pos].strip()

            tree = ast.parse(content)
        self._trace_calls(tree)
        return self._format_output()

    # def analyze(self):
    #     # First pass to find if main guard exists
    #     with open(self.file_path, "r") as file:
    #         content = file.read()
    #         __import__("ipdb").set_trace()
    #         tree = ast.parse(content)

    #         # Check for main guard
    #         has_main_guard = any(
    #             isinstance(node, ast.If)
    #             and isinstance(node.test, ast.Compare)
    #             and any(
    #                 "__main__" in getattr(n, "id", "")
    #                 for n in ast.walk(node.test)
    #             )
    #             for node in ast.walk(tree)
    #         )

    #         if not has_main_guard:
    #             self._trace_calls(tree)
    #             return self._format_output()

    #     # Second pass to read complete file if main guard exists
    #     with open(self.file_path, "r") as file:
    #         lines = file.readlines()
    #         # Remove main guard section
    #         content = "".join(
    #             line
    #             for line in lines
    #             if not line.strip().startswith("if __name__")
    #         )
    #         tree = ast.parse(content)

    #     self._trace_calls(tree)
    #     return self._format_output()


def analyze_current_file():
    analyzer = CodeAnalyzer()
    return analyzer.analyze()


def analyze_code(lpath):
    return CodeAnalyzer(lpath).analyze()


def main(args):
    diagram = analyze_current_file()
    print(diagram)
    return 0


def parse_args():
    import argparse

    import mngs

    is_script = mngs.gen.is_script()

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--var",
        "-v",
        type=int,
        choices=None,
        default=1,
        help="(default: %%(default)s)",
    )
    parser.add_argument(
        "--flag",
        "-f",
        action="store_true",
        default=False,
        help="(default: %%(default)s)",
    )
    args = parser.parse_args()
    mngs.str.printc(args, c="yellow")

    return args


if __name__ == "__main__":
    import sys

    import matplotlib.pyplot as plt

    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        verbose=False,
        agg=True,
    )

    exit_status = main(parse_args())

    mngs.gen.close(
        CONFIG,
        verbose=False,
        notify=False,
        message="",
        exit_status=exit_status,
    )


# EOF
