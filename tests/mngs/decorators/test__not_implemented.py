#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-28 15:44:39 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/decorators/test__not_implemented.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/decorators/test__not_implemented.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import pytest
from mngs.decorators._not_implemented import not_implemented


def test_not_implemented_decorator_warning():
    """Test that not_implemented decorator issues a warning."""

    @not_implemented
    def unimplemented_function():
        return "This should not be executed"

    # Check that a warning is issued when the function is called
    with pytest.warns(
        FutureWarning,
        match="Attempt to use unimplemented method: 'unimplemented_function'",
    ):
        unimplemented_function()


def test_not_implemented_preserves_function_name():
    """Test that not_implemented preserves the original function's name."""

    @not_implemented
    def test_function():
        pass

    with pytest.warns(FutureWarning) as record:
        test_function()

    # Check that the function name is in the warning message
    assert "test_function" in str(record[0].message)


def test_not_implemented_with_arguments():
    """Test not_implemented decorator with functions that take arguments."""

    @not_implemented
    def function_with_args(xx, yy):
        return xx + yy

    # Should warn but not raise error when called with arguments
    with pytest.warns(FutureWarning):
        function_with_args(1, 2)

    # The function should not actually execute (return None)
    with pytest.warns(FutureWarning):
        result = function_with_args(1, 2)
        assert result is None


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-06-07 22:16:25 (ywatanabe)"
#
# import warnings
#
#
# def not_implemented(func):
#     """
#     Decorator to mark methods as not implemented, issue a warning, and prevent their execution.
#
#     Arguments:
#         func (callable): The function or method to decorate.
#
#     Returns:
#         callable: A wrapper function that issues a warning and raises NotImplementedError when called.
#     """
#
#     def wrapper(*args, **kwargs):
#         # Issue a warning before raising the error
#         warnings.warn(
#             f"Attempt to use unimplemented method: '{func.__name__}'. This method is not yet available.",
#             category=FutureWarning,
#             stacklevel=2,
#         )
#         # # Raise the NotImplementedError
#         # raise NotImplementedError(f"The method '{func.__name__}' is not implemented yet.")
#
#     return wrapper

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-06-07 22:16:25 (ywatanabe)"
#
# import warnings
#
#
# def not_implemented(func):
#     """
#     Decorator to mark methods as not implemented, issue a warning, and prevent their execution.
#
#     Arguments:
#         func (callable): The function or method to decorate.
#
#     Returns:
#         callable: A wrapper function that issues a warning and raises NotImplementedError when called.
#     """
#
#     def wrapper(*args, **kwargs):
#         # Issue a warning before raising the error
#         warnings.warn(
#             f"Attempt to use unimplemented method: '{func.__name__}'. This method is not yet available.",
#             category=FutureWarning,
#             stacklevel=2,
#         )
#         # # Raise the NotImplementedError
#         # raise NotImplementedError(f"The method '{func.__name__}' is not implemented yet.")
#
#     return wrapper

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-06-07 22:16:25 (ywatanabe)"
#
# import warnings
#
#
# def not_implemented(func):
#     """
#     Decorator to mark methods as not implemented, issue a warning, and prevent their execution.
#
#     Arguments:
#         func (callable): The function or method to decorate.
#
#     Returns:
#         callable: A wrapper function that issues a warning and raises NotImplementedError when called.
#     """
#
#     def wrapper(*args, **kwargs):
#         # Issue a warning before raising the error
#         warnings.warn(
#             f"Attempt to use unimplemented method: '{func.__name__}'. This method is not yet available.",
#             category=FutureWarning,
#             stacklevel=2,
#         )
#         # # Raise the NotImplementedError
#         # raise NotImplementedError(f"The method '{func.__name__}' is not implemented yet.")
#
#     return wrapper

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-06-07 22:16:25 (ywatanabe)"
#
# import warnings
#
#
# def not_implemented(func):
#     """
#     Decorator to mark methods as not implemented, issue a warning, and prevent their execution.
#
#     Arguments:
#         func (callable): The function or method to decorate.
#
#     Returns:
#         callable: A wrapper function that issues a warning and raises NotImplementedError when called.
#     """
#
#     def wrapper(*args, **kwargs):
#         # Issue a warning before raising the error
#         warnings.warn(
#             f"Attempt to use unimplemented method: '{func.__name__}'. This method is not yet available.",
#             category=FutureWarning,
#             stacklevel=2,
#         )
#         # # Raise the NotImplementedError
#         # raise NotImplementedError(f"The method '{func.__name__}' is not implemented yet.")
#
#     return wrapper

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-06-07 22:16:25 (ywatanabe)"
#
# import warnings
#
#
# def not_implemented(func):
#     """
#     Decorator to mark methods as not implemented, issue a warning, and prevent their execution.
#
#     Arguments:
#         func (callable): The function or method to decorate.
#
#     Returns:
#         callable: A wrapper function that issues a warning and raises NotImplementedError when called.
#     """
#
#     def wrapper(*args, **kwargs):
#         # Issue a warning before raising the error
#         warnings.warn(
#             f"Attempt to use unimplemented method: '{func.__name__}'. This method is not yet available.",
#             category=FutureWarning,
#             stacklevel=2,
#         )
#         # # Raise the NotImplementedError
#         # raise NotImplementedError(f"The method '{func.__name__}' is not implemented yet.")
#
#     return wrapper

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-06-07 22:16:25 (ywatanabe)"
#
# import warnings
#
#
# def not_implemented(func):
#     """
#     Decorator to mark methods as not implemented, issue a warning, and prevent their execution.
#
#     Arguments:
#         func (callable): The function or method to decorate.
#
#     Returns:
#         callable: A wrapper function that issues a warning and raises NotImplementedError when called.
#     """
#
#     def wrapper(*args, **kwargs):
#         # Issue a warning before raising the error
#         warnings.warn(
#             f"Attempt to use unimplemented method: '{func.__name__}'. This method is not yet available.",
#             category=FutureWarning,
#             stacklevel=2,
#         )
#         # # Raise the NotImplementedError
#         # raise NotImplementedError(f"The method '{func.__name__}' is not implemented yet.")
#
#     return wrapper

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-06-07 22:16:25 (ywatanabe)"
#
# import warnings
#
#
# def not_implemented(func):
#     """
#     Decorator to mark methods as not implemented, issue a warning, and prevent their execution.
#
#     Arguments:
#         func (callable): The function or method to decorate.
#
#     Returns:
#         callable: A wrapper function that issues a warning and raises NotImplementedError when called.
#     """
#
#     def wrapper(*args, **kwargs):
#         # Issue a warning before raising the error
#         warnings.warn(
#             f"Attempt to use unimplemented method: '{func.__name__}'. This method is not yet available.",
#             category=FutureWarning,
#             stacklevel=2,
#         )
#         # # Raise the NotImplementedError
#         # raise NotImplementedError(f"The method '{func.__name__}' is not implemented yet.")
#
#     return wrapper

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-06-07 22:16:25 (ywatanabe)"
#
# import warnings
#
#
# def not_implemented(func):
#     """
#     Decorator to mark methods as not implemented, issue a warning, and prevent their execution.
#
#     Arguments:
#         func (callable): The function or method to decorate.
#
#     Returns:
#         callable: A wrapper function that issues a warning and raises NotImplementedError when called.
#     """
#
#     def wrapper(*args, **kwargs):
#         # Issue a warning before raising the error
#         warnings.warn(
#             f"Attempt to use unimplemented method: '{func.__name__}'. This method is not yet available.",
#             category=FutureWarning,
#             stacklevel=2,
#         )
#         # # Raise the NotImplementedError
#         # raise NotImplementedError(f"The method '{func.__name__}' is not implemented yet.")
#
#     return wrapper

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-06-07 22:16:25 (ywatanabe)"
#
# import warnings
#
#
# def not_implemented(func):
#     """
#     Decorator to mark methods as not implemented, issue a warning, and prevent their execution.
#
#     Arguments:
#         func (callable): The function or method to decorate.
#
#     Returns:
#         callable: A wrapper function that issues a warning and raises NotImplementedError when called.
#     """
#
#     def wrapper(*args, **kwargs):
#         # Issue a warning before raising the error
#         warnings.warn(
#             f"Attempt to use unimplemented method: '{func.__name__}'. This method is not yet available.",
#             category=FutureWarning,
#             stacklevel=2,
#         )
#         # # Raise the NotImplementedError
#         # raise NotImplementedError(f"The method '{func.__name__}' is not implemented yet.")
#
#     return wrapper

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-06-07 22:16:25 (ywatanabe)"
#
# import warnings
#
#
# def not_implemented(func):
#     """
#     Decorator to mark methods as not implemented, issue a warning, and prevent their execution.
#
#     Arguments:
#         func (callable): The function or method to decorate.
#
#     Returns:
#         callable: A wrapper function that issues a warning and raises NotImplementedError when called.
#     """
#
#     def wrapper(*args, **kwargs):
#         # Issue a warning before raising the error
#         warnings.warn(
#             f"Attempt to use unimplemented method: '{func.__name__}'. This method is not yet available.",
#             category=FutureWarning,
#             stacklevel=2,
#         )
#         # # Raise the NotImplementedError
#         # raise NotImplementedError(f"The method '{func.__name__}' is not implemented yet.")
#
#     return wrapper

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/decorators/_not_implemented.py
# --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-06-07 22:16:25 (ywatanabe)"
# # /home/ywatanabe/proj/mngs/src/mngs/gen/_not_implemented.py
# 
# import warnings
# 
# 
# def not_implemented(func):
#     """
#     Decorator to mark methods as not implemented, issue a warning, and prevent their execution.
# 
#     Arguments:
#         func (callable): The function or method to decorate.
# 
#     Returns:
#         callable: A wrapper function that issues a warning and raises NotImplementedError when called.
#     """
# 
#     def wrapper(*args, **kwargs):
#         # Issue a warning before raising the error
#         warnings.warn(
#             f"Attempt to use unimplemented method: '{func.__name__}'. This method is not yet available.",
#             category=FutureWarning,
#             stacklevel=2,
#         )
#         # # Raise the NotImplementedError
#         # raise NotImplementedError(f"The method '{func.__name__}' is not implemented yet.")
# 
#     return wrapper

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/decorators/_not_implemented.py
# --------------------------------------------------------------------------------
