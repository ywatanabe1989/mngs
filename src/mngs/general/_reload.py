#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-07 20:16:08 (ywatanabe)"


def reload(module_or_func):
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

    if hasattr(module_or_func, "__module__"):
        # If the object has a __module__ attribute, it's likely a function or class.
        # Attempt to reload its module.
        module_name = module_or_func.__module__
    else:
        # Otherwise, assume it's a module and try to get its name directly.
        module_name = module_or_func.__name__

    try:
        # Attempt to reload the module by name.
        importlib.reload(sys.modules[module_name])
    except KeyError:
        # The module is not found in sys.modules, likely due to it not being imported.
        print(f"Module {module_name} not found in sys.modules. Cannot reload.")
    except Exception as e:
        # Catch any other exceptions and print an error message.
        print(f"Failed to reload module {module_name}. Error: {e}")


# def reload(module_or_func):
#     import importlib
#     import sys

#     try:
#         importlib.reload(module_or_func)
#     except Exception as e:
#         print(e)

#     try:
#         importlib.reload(sys.modules[module_or_func.__module__])
#     except:
#         pass
