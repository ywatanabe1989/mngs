#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-07-07 18:34:54 (ywatanabe)"
# /home/ywatanabe/proj/mngs/src/mngs/general/_dotdict.py

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


# class DotDict:
#     """
#     A dictionary subclass that allows attribute-like access to keys.
#     """

#     def __init__(self, dictionary):
#         for key, value in dictionary.items():
#             if isinstance(value, dict):
#                 value = DotDict(value)
#             setattr(self, key, value)

#     def __getitem__(self, key):
#         return getattr(self, key)

#     def __setitem__(self, key, value):
#         setattr(self, key, value)

#     def get(self, key, default=None):
#         return getattr(self, key, default)

#     def to_dict(self):
#         """
#         Recursively converts DotDict and nested DotDict objects back to ordinary dictionaries.
#         """
#         result = {}
#         for key, value in self.__dict__.items():
#             if isinstance(value, DotDict):
#                 value = value.to_dict()
#             result[key] = value
#         return result

#     def __str__(self):
#         """
#         Returns a string representation of the dotdict by converting it to a dictionary and pretty-printing it.
#         """
#         return json.dumps(self.to_dict(), indent=4)

#     def __repr__(self):
#         """
#         Returns a string representation of the dotdict for debugging and development.
#         """
#         return self.__str__()


# class ddict(dict):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def __getattr__(self, item):
#         return self[item]

#     def __setattr__(self, key, value):
#         self[key] = value

#     def __getstate__(self):
#         return self.__dict__

#     def __setstate__(self, state):
#         self.__dict__.update(state)
#         for key, value in state.items():
#             self[key] = value


# # class ddict(dict):
# #     def __init__(self, *args, **kwargs):
# #         super().__init__(*args, **kwargs)

# #     def __getattr__(self, item):
# #         return self[item]

# #     def __setattr__(self, key, value):
# #         self[key] = value

# import yaml


# def ddict_representer(dumper, data):
#     return dumper.represent_dict(data)


# def ddict_constructor(loader, node):
#     return ddict(loader.construct_pairs(node))


# yaml.add_representer(ddict, ddict_representer)
# yaml.add_constructor("tag:yaml.org,2002:map", ddict_constructor)


# class ddict(dict):
#     """Dot notation access to dictionary attributes"""

#     def __getattr__(self, item):
#         try:
#             return self[item]
#         except KeyError:
#             raise AttributeError(
#                 f"'{type(self).__name__}' object has no attribute '{item}'"
#             )

#     def __setattr__(self, key, value):
#         self[key] = value

#     def __delattr__(self, key):
#         try:
#             del self[key]
#         except KeyError:
#             raise AttributeError(
#                 f"'{type(self).__name__}' object has no attribute '{key}'"
#             )

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)


# class ddict(dict):
#     """dot.notation access to dictionary attributes"""

#     __getattr__ = dict.get
#     __setattr__ = dict.__setitem__
#     __delattr__ = dict.__delitem__
