#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-06-05 02:05:43 (ywatanabe)"
# /home/ywatanabe/proj/mngs/src/mngs/general/_dotdict.py


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
