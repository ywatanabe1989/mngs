#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-05 01:01:08 (ywatanabe)"
# File: ./mngs_repo/src/mngs/dict/_DotDict.py

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-04 23:59:06 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/dict/_DotDict.py

# """
# Functionality:
#     * Provides a dictionary subclass with attribute-style (dot notation) access
#     * Supports nested dictionaries with recursive conversion
#     * Maintains all standard dictionary operations

# Input:
#     * Python dictionary objects to be converted to DotDict

# Output:
#     * DotDict objects with attribute-style access capabilities
#     * Regular dictionaries when converting back using to_dict()

# Prerequisites:
#     * Python's built-in json module
# """

# import json
# from typing import Any, Dict, ItemsView, Iterator, KeysView, ValuesView


# class DotDict:
#     """
#     A dictionary subclass that enables attribute-style access to dictionary keys.

#     Example
#     -------
#     >>> data = {'a': 1, 'b': {'c': 2}}
#     >>> dot_dict = DotDict(data)
#     >>> print(dot_dict.a)
#     1
#     >>> print(dot_dict.b.c)
#     2
#     >>> dot_dict.d = 3
#     >>> print(dot_dict.d)
#     3

#     Parameters
#     ----------
#     dictionary : Dict[str, Any]
#         Input dictionary to convert to DotDict

#     Returns
#     -------
#     DotDict
#         Dictionary with attribute-style access
#     """

#     def __init__(self, dictionary: Dict[str, Any]) -> None:
#         """Initializes DotDict with given dictionary."""
#         for key, value in dictionary.items():
#             if isinstance(value, dict):
#                 value = DotDict(value)
#             setattr(self, key, value)

#     def __getitem__(self, key: str) -> Any:
#         """Gets item using bracket notation."""
#         return getattr(self, key)

#     def __setitem__(self, key: str, value: Any) -> None:
#         """Sets item using bracket notation."""
#         setattr(self, key, value)

#     def __delitem__(self, key: str) -> None:
#         """Deletes item using bracket notation."""
#         delattr(self, key)

#     def get(self, key: str, default: Any = None) -> Any:
#         """Gets value for key, returns default if not found."""
#         return getattr(self, key, default)

#     def to_dict(self) -> Dict[str, Any]:
#         """Converts DotDict to regular dictionary recursively."""
#         result = {}
#         for key, value in self.__dict__.items():
#             if isinstance(value, DotDict):
#                 value = value.to_dict()
#             result[key] = value
#         return result

#     def copy(self) -> 'DotDict':
#         """Creates a deep copy."""
#         return DotDict(self.to_dict())

#     def keys(self) -> KeysView[str]:
#         """Returns dictionary keys view."""
#         return self.__dict__.keys()

#     def values(self) -> ValuesView[Any]:
#         """Returns dictionary values view."""
#         return self.__dict__.values()

#     def items(self) -> ItemsView[str, Any]:
#         """Returns dictionary items view."""
#         return self.__dict__.items()

#     def update(self, dictionary: Dict[str, Any]) -> None:
#         """Updates with key-value pairs from another dictionary."""
#         for key, value in dictionary.items():
#             if isinstance(value, dict):
#                 value = DotDict(value)
#             setattr(self, key, value)

#     def setdefault(self, key: str, default: Any = None) -> Any:
#         """Sets default value if key not present."""
#         if key not in self.__dict__:
#             self[key] = default
#         return self[key]

#     def pop(self, key: str, default: Any = None) -> Any:
#         """Removes and returns value for key."""
#         return self.__dict__.pop(key, default)

#     def __str__(self) -> str:
#         """String representation with JSON serialization."""
#         def default_handler(obj: Any) -> str:
#             return str(obj)
#         return json.dumps(self.to_dict(), indent=4, default=default_handler)

#     def __repr__(self) -> str:
#         """Developer representation."""
#         return self.__str__()

#     def __len__(self) -> int:
#         """Returns number of items."""
#         return len(self.__dict__)

#     def __contains__(self, key: str) -> bool:
#         """Implements 'in' operator."""
#         return key in self.__dict__

#     def __iter__(self) -> Iterator[str]:
#         """Returns iterator over keys."""
#         return iter(self.__dict__)


# # EOF

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-04 23:55:45 (ywatanabe)"
# File: ./mngs_repo/src/mngs/dict/_DotDict.py

import json


class DotDict:
    """
    A dictionary subclass that allows attribute-like access to keys.
    """

    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = DotDict(value)
            setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def to_dict(self):
        """
        Recursively converts DotDict and nested DotDict objects back to ordinary dictionaries.
        """
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, DotDict):
                value = value.to_dict()
            result[key] = value
        return result

    # def __str__(self):
    #     """
    #     Returns a string representation of the dotdict by converting it to a dictionary and pretty-printing it.
    #     """
    #     return json.dumps(self.to_dict(), indent=4)

    def __str__(self):
        """
        Returns a string representation, handling non-JSON-serializable objects.
        """
        def default_handler(obj):
            return str(obj)

        return json.dumps(self.to_dict(), indent=4, default=default_handler)

    def __repr__(self):
        """
        Returns a string representation of the dotdict for debugging and development.
        """
        return self.__str__()

    def __len__(self):
        """
        Returns the number of key-value pairs in the dictionary.
        """
        return len(self.__dict__)

    def keys(self):
        """
        Returns a view object displaying a list of all the keys in the dictionary.
        """
        return self.__dict__.keys()

    def values(self):
        """
        Returns a view object displaying a list of all the values in the dictionary.
        """
        return self.__dict__.values()

    def items(self):
        """
        Returns a view object displaying a list of all the items (key, value pairs) in the dictionary.
        """
        return self.__dict__.items()

    def update(self, dictionary):
        """
        Updates the dictionary with the key-value pairs from another dictionary.
        """
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = DotDict(value)
            setattr(self, key, value)

    def setdefault(self, key, default=None):
        """
        Returns the value of the given key. If the key does not exist, insert the key with the specified default value.
        """
        if key not in self.__dict__:
            self[key] = default
        return self[key]

    def pop(self, key, default=None):
        """
        Removes the specified key and returns the corresponding value.
        """
        return self.__dict__.pop(key, default)

    def __contains__(self, key):
        """
        Checks if the dotdict contains the specified key.
        """
        return key in self.__dict__

    def __iter__(self):
        """
        Returns an iterator over the keys of the dictionary.
        """
        return iter(self.__dict__)

    def copy(self):
        """
        Creates a deep copy of the DotDict object.
        """
        return DotDict(self.to_dict().copy())

    def __delitem__(self, key):
        """
        Deletes the specified key from the dictionary.
        """
        delattr(self, key)




# EOF


# EOF
