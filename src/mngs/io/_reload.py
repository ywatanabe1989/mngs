#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-06-04 19:10:36 (ywatanabe)"


def reload(module_or_func, verbose=False):
    """
    Reloads the given module or the module containing the given function.

    This function attempts to reload a module directly if a module is passed.
    If a function is passed, it tries to reload the module that contains the function.
    This can be useful during development when changes are made to modules and you want
    those changes to be reflected without restarting the Python interpreter.

    Arguments:
        module_or_func (module|function): The module or function to reload. If a function is
                                          provided, its containing module is reloaded.

    Returns:
        None

    Note:
        Reloading modules can have unexpected side effects, especially for modules that
        maintain state or for complex imports. Use with caution.
    """
    import importlib
    import sys

    if module_or_func in sys.modules:
        del sys.modules[module_or_func]
        importlib.reload(module_or_func)

    if hasattr(module_or_func, "__module__"):
        # If the object has a __module__ attribute, it's likely a function or class.
        # Attempt to reload its module.
        module_name = module_or_func.__module__
        if module_name not in sys.modules:
            print(
                f"Module {module_name} not found in sys.modules. Cannot reload."
            )
            return
    elif (
        hasattr(module_or_func, "__name__")
        and module_or_func.__name__ in sys.modules
    ):
        # Otherwise, assume it's a module and try to get its name directly.
        module_name = module_or_func.__name__
    else:
        print(
            f"Provided object is neither a recognized module nor a function/class with a __module__ attribute."
        )
        return

    try:
        # Attempt to reload the module by name.
        importlib.reload(sys.modules[module_name])

    except KeyError:
        # The module is not found in sys.modules, likely due to it not being imported.
        print(f"Module {module_name} not found in sys.modules. Cannot reload.")
    except Exception as e:
        # Catch any other exceptions and print an error message.
        print(f"Failed to reload module {module_name}. Error: {e}")
