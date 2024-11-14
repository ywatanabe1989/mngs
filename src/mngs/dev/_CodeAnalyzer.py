#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-15 10:37:38 (ywatanabe)"
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
        self.seen_calls = set()  # Track unique function calls

    def _trace_calls(self, node, depth=0):
        if isinstance(node, ast.FunctionDef):
            # Track all function definitions
            self.execution_flow.append((depth, node.name, self.sequence))
            self.sequence += 1

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
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
        last_depth = 0

        for depth, call, seq in self.execution_flow:
            # Maintain hierarchy but limit maximum depth difference
            if depth > last_depth + 1:
                depth = last_depth + 1
                last_depth = depth

            prefix = "    " * depth
            output.append(f"{prefix}[{seq}] └── {call}")

        return "\n".join(output)

    def analyze(self):
        with open(self.file_path, "r") as file:
            tree = ast.parse(file.read())
        self._trace_calls(tree)
        return self._format_output()

    # # def _trace_calls(self, node, depth=0):
    # #     if isinstance(node, ast.Call):
    # #         if isinstance(node.func, ast.Name):
    # #             func_name = node.func.id
    # #             if func_name not in self.skip_functions:
    # #                 self.execution_flow.append(
    # #                     (depth, func_name, self.sequence)
    # #                 )
    # #                 self.sequence += 1
    # #         elif isinstance(node.func, ast.Attribute):
    # #             parts = []
    # #             current = node.func
    # #             while isinstance(current, ast.Attribute):
    # #                 parts.insert(0, current.attr)
    # #                 current = current.value
    # #             if isinstance(current, ast.Name):
    # #                 parts.insert(0, current.id)
    # #             func_name = ".".join(parts)
    # #             if not any(skip in func_name for skip in self.skip_functions):
    # #                 self.execution_flow.append(
    # #                     (depth, func_name, self.sequence)
    # #                 )
    # #                 self.sequence += 1

    # #     for child in ast.iter_child_nodes(node):
    # #         self._trace_calls(child, depth + 1)

    # def _format_output(self):
    #     output = ["Execution Flow:"]
    #     for depth, call, seq in self.execution_flow:
    #         prefix = "    " * depth
    #         output.append(f"{prefix}[{seq}] └── {call}")
    #     return "\n".join(output)

    # def analyze(self):
    #     with open(self.file_path, "r") as file:
    #         tree = ast.parse(file.read())
    #     self._trace_calls(tree)
    #     return self._format_output()


# class CodeAnalyzer:
#     def __init__(
#         self,
#         file_path: Optional[str] = None,
#         ignore_private: bool = True,
#         max_depth: int = 3,
#     ):
#         self.file_path = file_path or inspect.getfile(inspect.currentframe())
#         self.classes: Dict[str, List[str]] = {}
#         self.dependencies: Dict[str, Dict[str, Any]] = {}
#         self.function_defs: Set[str] = set()
#         self.variables: Dict[str, Set[str]] = {}
#         self.execution_order: List[Tuple[int, str, int]] = []
#         self.imports: Set[str] = set()
#         self.max_depth = max_depth
#         self.current_depth = 0
#         self.current_scope = "global"
#         self.ignore_private = ignore_private
#         self.graph = nx.DiGraph()

#     def _process_call(self, node: ast.Call) -> None:
#         if isinstance(node.func, ast.Name):
#             func_name = node.func.id
#         elif isinstance(node.func, ast.Attribute):
#             parts = []
#             current = node.func
#             while isinstance(current, ast.Attribute):
#                 parts.insert(0, current.attr)
#                 current = current.value
#             if isinstance(current, ast.Name):
#                 parts.insert(0, current.id)
#             func_name = ".".join(parts)
#         else:
#             return

#         self.execution_order.append(
#             (len(self.execution_order), func_name, self.current_depth)
#         )

#         if func_name not in self.dependencies:
#             self.dependencies[func_name] = {"inputs": set(), "outputs": set()}

#         for arg in node.args:
#             if isinstance(arg, ast.Name):
#                 self.dependencies[func_name]["inputs"].add(arg.id)
#                 self.graph.add_edge(arg.id, func_name, type="input")

#     def _trace_node(self, node: ast.AST) -> None:
#         if self.current_depth > self.max_depth:
#             return

#         if isinstance(node, ast.Assign):
#             for target in node.targets:
#                 if isinstance(target, ast.Name):
#                     if self.current_scope not in self.variables:
#                         self.variables[self.current_scope] = set()
#                     self.variables[self.current_scope].add(target.id)
#                     self.graph.add_node(
#                         target.id, type="variable", scope=self.current_scope
#                     )

#         if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
#             prev_scope = self.current_scope
#             self.current_scope = node.name
#             self.graph.add_node(
#                 node.name,
#                 type="class" if isinstance(node, ast.ClassDef) else "function",
#                 scope=prev_scope,
#             )

#             for child in ast.iter_child_nodes(node):
#                 self._trace_node(child)
#             self.current_scope = prev_scope
#         else:
#             for child in ast.iter_child_nodes(node):
#                 if isinstance(child, ast.Call):
#                     self._process_call(child)
#                 self.current_depth += 1
#                 self._trace_node(child)
#                 self.current_depth -= 1

#     def analyze(self) -> tuple[str, str, nx.DiGraph]:
#         with open(self.file_path, "r") as file:
#             tree = ast.parse(file.read())

#         for node in ast.walk(tree):
#             if isinstance(node, ast.Import):
#                 for name in node.names:
#                     self.imports.add(name.name)
#                     self.graph.add_node(name.name, type="import")
#             elif isinstance(node, ast.ImportFrom):
#                 if node.module:
#                     self.imports.add(node.module)
#                     self.graph.add_node(node.module, type="import")
#             elif isinstance(node, ast.FunctionDef):
#                 if not (self.ignore_private and node.name.startswith("_")):
#                     self.function_defs.add(node.name)
#             elif isinstance(node, ast.ClassDef):
#                 methods = []
#                 for item in node.body:
#                     if isinstance(item, ast.FunctionDef):
#                         if not (
#                             self.ignore_private and item.name.startswith("_")
#                         ):
#                             methods.append(item.name)
#                             self.function_defs.add(f"{node.name}.{item.name}")
#                             self.graph.add_edge(
#                                 node.name, item.name, type="method"
#                             )
#                 self.classes[node.name] = methods

#         self._trace_node(tree)
#         return self.generate_tree(), self.generate_mermaid(), self.graph

#     def generate_tree(self) -> str:
#         lines = []

#         if self.imports:
#             lines.append("Imports")
#             for imp in sorted(self.imports):
#                 lines.append(f"├── {imp}")
#             lines.append("")

#         if self.variables:
#             lines.append("Variables")
#             for scope, vars in sorted(self.variables.items()):
#                 if scope == "global":
#                     for var in sorted(vars):
#                         lines.append(f"├── {var}")
#                 else:
#                     lines.append(f"├── {scope}")
#                     for var in sorted(vars):
#                         lines.append(f"│   ├── {var}")
#             lines.append("")

#         if self.execution_order:
#             lines.append("Execution Flow")
#             lines.append("* Script starts")
#             last_depth = 0
#             for _, func, depth in sorted(self.execution_order):
#                 if "." not in func:
#                     prefix = "│   " * min(depth, self.max_depth)
#                     if depth > last_depth:
#                         lines.append(f"{prefix}└── {func}")
#                     else:
#                         lines.append(f"{prefix}├── {func}")
#                     if func in self.dependencies and self.dependencies[
#                         func
#                     ].get("outputs"):
#                         lines.append(
#                             f"{prefix}    └── returns {', '.join(self.dependencies[func]['outputs'])}"
#                         )
#                     last_depth = depth
#             lines.append("")

#         if self.classes:
#             lines.append("Classes")
#             for class_name, methods in sorted(self.classes.items()):
#                 lines.append(f"├── {class_name}")
#                 for method in sorted(methods):
#                     lines.append(f"│   ├── {method}")
#                     if method in self.dependencies:
#                         deps = self.dependencies[method]
#                         if deps.get("inputs"):
#                             lines.append(
#                                 f"│   │   ├── inputs: {', '.join(deps['inputs'])}"
#                             )
#                         if deps.get("outputs"):
#                             lines.append(
#                                 f"│   │   └── outputs: {', '.join(deps['outputs'])}"
#                             )
#             lines.append("")

#         standalone_funcs = self.function_defs - set().union(
#             *self.classes.values()
#         )
#         if standalone_funcs:
#             lines.append("Functions")
#             for func in sorted(standalone_funcs):
#                 lines.append(f"├── {func}")
#                 if func in self.dependencies:
#                     deps = self.dependencies[func]
#                     if deps.get("inputs"):
#                         lines.append(
#                             f"│   ├── inputs: {', '.join(deps['inputs'])}"
#                         )
#                     if deps.get("outputs"):
#                         lines.append(
#                             f"│   └── outputs: {', '.join(deps['outputs'])}"
#                         )
#             lines.append("")

#         return "\n".join(lines)

#     def generate_mermaid(self) -> str:
#         lines = ["```mermaid", "graph TD"]

#         if self.imports:
#             lines.append("    subgraph Imports")
#             for imp in sorted(self.imports):
#                 lines.append(f"        {imp}[{imp}]:::import")
#             lines.append("    end")

#         if self.classes:
#             lines.append("    subgraph Classes")
#             for class_name, methods in sorted(self.classes.items()):
#                 lines.append(f"        {class_name}[{class_name}]:::class")
#                 for method in sorted(methods):
#                     method_id = f"{class_name}_{method}"
#                     lines.append(f"        {method_id}({method}):::method")
#                     lines.append(f"        {class_name} --> {method_id}")
#                     if method in self.dependencies:
#                         for dep in self.dependencies[method]["inputs"]:
#                             lines.append(
#                                 f"        {method_id} -.->|uses| {dep}"
#                             )
#             lines.append("    end")

#         standalone_funcs = self.function_defs - set().union(
#             *self.classes.values()
#         )
#         if standalone_funcs:
#             lines.append("    subgraph Functions")
#             for func in sorted(standalone_funcs):
#                 lines.append(f"        {func}({func}):::function")
#                 if func in self.dependencies:
#                     for dep in self.dependencies[func]["inputs"]:
#                         lines.append(f"        {func} -.->|uses| {dep}")
#             lines.append("    end")

#         lines.extend(
#             [
#                 "    classDef import fill:#f9f9f9,stroke:#999;",
#                 "    classDef class fill:#f9f,stroke:#333,stroke-width:2px;",
#                 "    classDef method fill:#bbf,stroke:#333;",
#                 "    classDef function fill:#bfb,stroke:#333;",
#             ]
#         )
#         lines.append("```")
#         return "\n".join(lines)


def analyze_current_file():
    analyzer = CodeAnalyzer()
    return analyzer.analyze()


def analyze_code(lpath):
    return CodeAnalyzer(lpath).analyze()


# def analyze_code(lpath):
#     analyzer = CodeAnalyzer(lpath)
#     tree, mermaid, graph = analyzer.analyze()

#     # Print only essential sections with clear separation
#     sections = tree.split("\n\n")
#     clean_output = []

#     for section in sections:
#         if any(
#             section.startswith(x)
#             for x in ["Classes", "Functions", "Execution Flow"]
#         ):
#             clean_output.append(section)

#     return "\n\n".join(clean_output)


# def main(args):
#     output = analyze_current_file()
#     print("\n=== Code Structure Analysis ===\n")
#     print(output)
#     return 0


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
