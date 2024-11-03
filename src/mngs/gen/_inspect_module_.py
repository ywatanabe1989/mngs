#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-06 01:01:19 (ywatanabe)"
# /home/ywatanabe/proj/_mngs_repo_openhands/src/mngs/gen/_inspect_module.py


import inspect
import sys
from typing import Optional, Set, List, Tuple
import pandas as pd

def inspect_module(
    module: object,
    prefix: str = "",
    max_depth: int = 5,
    visited: Optional[Set[str]] = None,
    docstring: bool = False,
    tree: bool = True,
    current_depth: int = 0,
    print_output: bool = False,
) -> pd.DataFrame:
    """
    List the contents of a module recursively and return as a DataFrame.

    Example
    -------
    >>> 
    >>> df = inspect_module(mngs)
    >>> print(df)
       Type           Name                    Docstring  Depth
    0    M            mngs  Module description              0
    1    F  mngs.some_function  Function description        1
    2    C  mngs.SomeClass  Class description               1
    ...

    Parameters
    ----------
    module : object
        The module to inspect
    prefix : str, optional
        Prefix for the current module, used for recursive calls
    max_depth : int, optional
        Maximum depth for recursion
    visited : Set[str], optional
        Set of visited module names to prevent infinite recursion
    docstring : bool, optional
        Whether to include docstrings in the output
    tree : bool, optional
        Whether to display the output in a tree-like structure
    current_depth : int, optional
        Current depth in the module hierarchy
    print_output : bool, optional
        Whether to print the output or not

    Returns
    -------
    pd.DataFrame
        DataFrame containing (Type, Name, Docstring, Depth) for each module content
    """
    if visited is None:
        visited = set()

    content_list = []

    if max_depth < 0 or module.__name__ in visited:
        return pd.DataFrame(content_list, columns=['Type', 'Name', 'Docstring', 'Depth'])

    visited.add(module.__name__)

    try:
        module_version = f" (v{module.__version__})" if hasattr(module, '__version__') else ""
        content_list.append(('M', f"{prefix}.{module.__name__}" if prefix else module.__name__, module_version, current_depth))
    except Exception:
        pass

    for name, obj in inspect.getmembers(module):
        if name.startswith("_"):
            continue

        full_name = f"{prefix}.{name}" if prefix else name

        if inspect.ismodule(obj) and obj.__name__.startswith("mngs"):
            content_list.append(('M', full_name, obj.__doc__ if docstring and obj.__doc__ else "", current_depth))
            sub_df = inspect_module(obj, full_name, max_depth - 1, visited, docstring, tree, current_depth + 1, print_output)
            content_list.extend(sub_df.values.tolist())
        elif inspect.isfunction(obj):
            content_list.append(('F', full_name, obj.__doc__ if docstring and obj.__doc__ else "", current_depth))
        elif inspect.isclass(obj):
            content_list.append(('C', full_name, obj.__doc__ if docstring and obj.__doc__ else "", current_depth))

    df = pd.DataFrame(content_list, columns=['Type', 'Name', 'Docstring', 'Depth'])

    if tree and current_depth == 0 and print_output:
        _print_module_contents(df)

    return df

def _print_module_contents(df: pd.DataFrame) -> None:
    df_sorted = df.sort_values(['Depth', 'Name'])
    depth_last = {}

    for index, row in df_sorted.iterrows():
        depth = row['Depth']
        is_last = index == len(df_sorted) - 1 or df_sorted.iloc[index + 1]['Depth'] <= depth

        prefix = ""
        for d in range(depth):
            if d == depth - 1:
                prefix += "└── " if is_last else "├── "
            else:
                prefix += "    " if depth_last.get(d, False) else "│   "

        print(f"{prefix}({row['Type']}) {row['Name']}{row['Docstring']}")

        depth_last[depth] = is_last

if __name__ == "__main__":
    sys.setrecursionlimit(10_000)
    df = inspect_module(mngs, docstring=True, print_output=False)
    print(mngs.pd.round(df))

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-10-06 00:53:42 (ywatanabe)"
# # /home/ywatanabe/proj/_mngs_repo_openhands/src/mngs/gen/_inspect_module.py

# 
# import inspect
# import sys
# from typing import Optional, Set

# def inspect_module(
#     module: object,
#     prefix: str = "",
#     max_depth: int = 5,
#     visited: Optional[Set[str]] = None,
#     docstring: bool = False,
#     tree: bool = True,
#     indent: str = ""
# ) -> None:
#     """
#     List the contents of a module recursively.

#     Example
#     -------
#     >>> 
#     >>> inspect_module(mngs)
#     Module: mngs
#     ├── Function: mngs.some_function
#     └── Class: mngs.SomeClass
#     ...

#     Parameters
#     ----------
#     module : object
#         The module to inspect
#     prefix : str, optional
#         Prefix for the current module, used for recursive calls
#     max_depth : int, optional
#         Maximum depth for recursion
#     visited : Set[str], optional
#         Set of visited module names to prevent infinite recursion
#     docstring : bool, optional
#         Whether to include docstrings in the output
#     tree : bool, optional
#         Whether to display the output in a tree-like structure
#     indent : str, optional
#         Current indentation for tree structure

#     Returns
#     -------
#     None
#     """

#     try:
#         print(f"{module.__name__}" f"{module.__version__}")
#     except Exception as e:
#         pass

#     if visited is None:
#         visited = set()

#     if max_depth == 0 or module.__name__ in visited:
#         return

#     visited.add(module.__name__)

#     for index, (name, obj) in enumerate(inspect.getmembers(module)):
#         if name.startswith("_"):
#             continue

#         full_name = f"{prefix}.{name}" if prefix else name

#         is_last = index == len(inspect.getmembers(module)) - 1
#         tree_prefix = "└── " if is_last else "├── " if tree else ""
#         next_indent = indent + ("    " if is_last else "│   ") if tree else indent

#         if inspect.ismodule(obj) and obj.__name__.startswith("mngs"):
#             print(f"{indent}{tree_prefix}Module: {full_name}")
#             inspect_module(obj, full_name, max_depth - 1, visited, docstring, tree, next_indent)
#         elif inspect.isfunction(obj):
#             print(f"{indent}{tree_prefix}Function: {full_name}")
#             if docstring and obj.__doc__:
#                 doc_lines = obj.__doc__.strip().split('\n')
#                 for i, line in enumerate(doc_lines):
#                     doc_prefix = "     " if i == 0 else "     "
#                     print(f"{next_indent}{doc_prefix}{line.strip()}")
#         elif inspect.isclass(obj):
#             print(f"{indent}{tree_prefix}Class: {full_name}")
#             if docstring and obj.__doc__:
#                 doc_lines = obj.__doc__.strip().split('\n')
#                 for i, line in enumerate(doc_lines):
#                     doc_prefix = "     " if i == 0 else "     "
#                     print(f"{next_indent}{doc_prefix}{line.strip()}")

# if __name__ == "__main__":
#     sys.setrecursionlimit(10_000)
#     inspect_module(mngs, docstring=True)
