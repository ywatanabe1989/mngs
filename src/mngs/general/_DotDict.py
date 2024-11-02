#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-06 18:26:45 (ywatanabe)"
# /home/ywatanabe/proj/mngs/src/mngs/general/_dotdict.py

import json
from types import MappingProxyType
from typing import Any, Dict, Iterator, List, Union
from copy import deepcopy

# class DotDict:
#     def __init__(self, data=None):
#         self._data = {} if data is None else data
#         self._frozen = False

#     def __getattr__(self, key: str) -> Any:
#         try:
#             value = self._data[key]
#             if isinstance(value, dict) and not isinstance(value, DotDict):
#                 return DotDict(value)
#             return value
#         except KeyError:
#             raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

#     def __setattr__(self, key: str, value: Any) -> None:
#         if key == '_data':
#             super().__setattr__(key, value)
#         else:
#             self._data[key] = value

#     def __getitem__(self, key: str) -> Any:
#         return self.__getattr__(key)

#     def __setitem__(self, key: str, value: Any) -> None:
#         self.__setattr__(key, value)

#     def freeze(self) -> None:
#         """Freeze the DotDict, making it immutable."""
#         self._frozen = True
#         for value in self._data.values():
#             if isinstance(value, DotDict):
#                 value.freeze()
#         self._data = MappingProxyType(self._data)

#     def unfreeze(self) -> None:
#         """Unfreeze the DotDict, making it mutable again."""
#         if self._frozen:
#             self._data = dict(self._data)
#             for value in self._data.values():
#                 if isinstance(value, DotDict):
#                     value.unfreeze()
#             self._frozen = False

#     def get(self, key: str, default: Any = None) -> Any:
#         return self._data.get(key, default)

#     def to_dict(self) -> Dict[str, Any]:
#         """Recursively converts DotDict and nested DotDict objects back to ordinary dictionaries."""
#         return {k: v.to_dict() if isinstance(v, DotDict) else v for k, v in self._data.items()}

#     def __str__(self) -> str:
#         """Returns a string representation of the DotDict."""
#         return json.dumps(self.to_dict(), indent=4)

#     def __repr__(self) -> str:
#         """Returns a string representation of the DotDict for debugging and development."""
#         return self.__str__()

#     def __len__(self) -> int:
#         """Returns the number of key-value pairs in the dictionary."""
#         return len(self._data)

#     def __deepcopy__(self, memo):
#         return DotDict(deepcopy(self._data, memo))

#     def keys(self) -> List[str]:
#         """Returns a list of all the keys in the dictionary."""
#         return list(self._data.keys())

#     def values(self) -> List[Any]:
#         """Returns a list of all the values in the dictionary."""
#         return list(self._data.values())

#     def items(self) -> List[tuple]:
#         """Returns a list of all the items (key, value pairs) in the dictionary."""
#         return list(self._data.items())

#     def update(self, dictionary: Dict[str, Any]) -> None:
#         """Updates the dictionary with the key-value pairs from another dictionary."""
#         if self._frozen:
#             raise AttributeError("Cannot modify frozen DotDict")
#         for key, value in dictionary.items():
#             if isinstance(value, dict):
#                 value = DotDict(value)
#             self._data[key] = value

#     def setdefault(self, key: str, default: Any = None) -> Any:
#         """Returns the value of the given key. If the key does not exist, insert the key with the specified default value."""
#         if self._frozen:
#             raise AttributeError("Cannot modify frozen DotDict")
#         return self._data.setdefault(key, default)

#     def pop(self, key: str, default: Any = None) -> Any:
#         """Removes the specified key and returns the corresponding value."""
#         if self._frozen:
#             raise AttributeError("Cannot modify frozen DotDict")
#         return self._data.pop(key, default)

#     def __contains__(self, key: str) -> bool:
#         """Checks if the DotDict contains the specified key."""
#         return key in self._data

#     def __iter__(self) -> Iterator[str]:
#         """Returns an iterator over the keys of the dictionary."""
#         return iter(self._data)

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

    def __str__(self):
        """
        Returns a string representation of the dotdict by converting it to a dictionary and pretty-printing it.
        """
        return json.dumps(self.to_dict(), indent=4)

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
