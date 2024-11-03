# src from here --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-04-21 23:09:40"
# # Author: Yusuke Watanabe (ywata1989@gmail.com)
# 
# """
# This script provides detailed system information including system basics, boot time, CPU, memory, disk, network, and custom user environment variables.
# """
# 
# import platform as _platform
# import sys
# from datetime import datetime as _datetime
# from pprint import pprint
# 
# import matplotlib.pyplot as plt
# import psutil as _psutil
# import yaml as _yaml
# 
# 
# def get_specs(
#     system=True,
#     # boot_time=True,
#     cpu=True,
#     gpu=True,
#     disk=True,
#     network=True,
#     verbose=False,
#     yaml=False,
# ):
#     """
#     Collects and returns system specifications including system information, CPU, GPU, disk, and network details.
# 
#     This function gathers various pieces of system information based on the parameters provided. It can return the data in a dictionary format or print it out based on the verbose flag. Additionally, there's an option to format the output as YAML.
# 
#     Arguments:
#         system (bool): If True, collects system-wide information such as OS and node name. Default is True.
#         boot_time (bool): If True, collects system boot time. Currently commented out in the implementation. Default is True.
#         cpu (bool): If True, collects CPU-specific information including frequency and usage. Default is True.
#         gpu (bool): If True, collects GPU-specific information. Default is True.
#         disk (bool): If True, collects disk usage information for all partitions. Default is True.
#         network (bool): If True, collects network interface and traffic information. Default is True.
#         verbose (bool): If True, prints the collected information using pprint. Default is False.
#         yaml (bool): If True, formats the collected information as YAML. This modifies the return type to a YAML formatted string. Default is False.
# 
#     Returns:
#         dict or str: By default, returns a dictionary containing the collected system specifications. If `yaml` is True, returns a YAML-formatted string instead.
# 
#     Note:
#         - The actual collection of system, CPU, GPU, disk, and network information depends on the availability of corresponding libraries and access permissions.
#         - The `boot_time` argument is currently not used as its corresponding code is commented out.
#         - The function uses global variables and imports within its scope, which might affect its reusability and testability.
# 
#     Example:
#         >>> specs = get_specs(verbose=True)
#         This will print and return the system specifications based on the default parameters.
# 
#     Dependencies:
#         - This function depends on the `mngs` library for accessing system information and formatting output. Ensure this library is installed and properly configured.
#         - Python standard libraries: `datetime`, `platform`, `psutil`, `yaml` (optional for YAML output).
# 
#     Raises:
#         PermissionError: If the function lacks necessary permissions to access certain system information, especially disk and network details.
#     """
# 
#     import mngs
# 
#     # To prevent import errors, _SUPPLE_INFO is collected here.
#     global _SUPPLE_INFO
#     _SUPPLE_INFO = mngs.res._utils.get_env_info()._asdict()
# 
#     collected_info = {}  # OrderedDict()
# 
#     collected_info["Collected Time"] = _datetime.now().strftime(
#         "%Y-%m-%d %H:%M:%S"
#     )
#     if system:
#         collected_info["System Information"] = _system_info()
#     # if boot_time:
#     #     collected_info["Boot Time"] = _boot_time_info()
#     if cpu:
#         collected_info["CPU Info"] = _cpu_info()
#         collected_info["Memory Info"] = _memory_info()
#     if gpu:
#         collected_info["GPU Info"] = _supple_nvidia_info()
#         # mngs.gen.placeholder()
#     if disk:
#         collected_info["Disk Info"] = _disk_info()
#     if network:
#         collected_info["Network Info"] = _network_info()
# 
#     if yaml:
#         collected_info = _yaml.dump(collected_info, sort_keys=False)
# 
#     if verbose:
#         pprint(collected_info)
# 
#     return collected_info
# 
# 
# def _system_info():
#     uname = _platform.uname()
#     return {
#         "OS": _supple_os_info()["os"],
#         # "GCC version": _supple_os_info()["gcc_version"],
#         # "System": uname.system,
#         "Node Name": uname.node,
#         "Release": uname.release,
#         "Version": uname.version,
#         # "Machine": uname.machine,
#         # "Processor": uname.processor,
#     }
# 
# 
# # def _boot_time_info():
# #     boot_time_timestamp = _psutil.boot_time()
# #     bt = _datetime.fromtimestamp(boot_time_timestamp)
# #     return {
# #         "Boot Time": f"{bt.year}-{bt.month:02d}-{bt.day:02d} {bt.hour:02d}:{bt.minute:02d}:{bt.second:02d}"
# #     }
# 
# 
# def _cpu_info():
#     cpufreq = _psutil.cpu_freq()
#     cpu_usage_per_core = _psutil.cpu_percent(percpu=True, interval=1)
#     return {
#         "Physical cores": _psutil.cpu_count(logical=False),
#         "Total cores": _psutil.cpu_count(logical=True),
#         "Max Frequency": f"{cpufreq.max:.2f} MHz",
#         "Min Frequency": f"{cpufreq.min:.2f} MHz",
#         "Current Frequency": f"{cpufreq.current:.2f} MHz",
#         "CPU Usage Per Core": {
#             f"Core {i}": f"{percentage}%"
#             for i, percentage in enumerate(cpu_usage_per_core)
#         },
#         "Total CPU Usage": f"{_psutil.cpu_percent()}%",
#     }
# 
# 
# def _memory_info():
#     import mngs
# 
#     svmem = _psutil.virtual_memory()
#     swap = _psutil.swap_memory()
# 
#     return {
#         "Memory": {
#             "Total": mngs.gen.readable_bytes(svmem.total),
#             "Available": mngs.gen.readable_bytes(svmem.available),
#             "Used": mngs.gen.readable_bytes(svmem.used),
#             "Percentage": svmem.percent,
#         },
#         "SWAP": {
#             "Total": mngs.gen.readable_bytes(swap.total),
#             "Free": mngs.gen.readable_bytes(swap.free),
#             "Used": mngs.gen.readable_bytes(swap.used),
#             "Percentage": swap.percent,
#         },
#     }
# 
# 
# def _disk_info():
#     import mngs
# 
#     partitions_info = {}
#     partitions = _psutil.disk_partitions()
#     for partition in partitions:
#         try:
#             usage = _psutil.disk_usage(partition.mountpoint)
#             partitions_info[partition.device] = {
#                 "Mountpoint": partition.mountpoint,
#                 "File system type": partition.fstype,
#                 "Total Size": mngs.gen.readable_bytes(usage.total),
#                 "Used": mngs.gen.readable_bytes(usage.used),
#                 "Free": mngs.gen.readable_bytes(usage.free),
#                 "Percentage": usage.percent,
#             }
#         except PermissionError:
#             continue
# 
#     disk_io = _psutil.disk_io_counters()
#     return {
#         "Partitions": partitions_info,
#         "Total read": mngs.gen.readable_bytes(disk_io.read_bytes),
#         "Total write": mngs.gen.readable_bytes(disk_io.write_bytes),
#     }
# 
# 
# def _network_info():
#     import mngs
# 
#     if_addrs = _psutil.net_if_addrs()
#     interfaces = {}
#     for interface_name, interface_addresses in if_addrs.items():
#         interface_info = []
#         for address in interface_addresses:
#             interface_info.append(
#                 {
#                     # "Address Type": "IP" if address.family == _psutil.AF_INET else "MAC",
#                     "Address": address.address,
#                     "Netmask": address.netmask,
#                     "Broadcast": address.broadcast,
#                 }
#             )
#         interfaces[interface_name] = interface_info
# 
#     net_io = _psutil.net_io_counters()
#     return {
#         "Interfaces": interfaces,
#         "Total Sent": mngs.gen.readable_bytes(net_io.bytes_sent),
#         "Total Received": mngs.gen.readable_bytes(net_io.bytes_recv),
#     }
# 
# 
# def _python_info():
#     return _supple_python_info()
# 
# 
# def _supple_os_info():
#     _SUPPLE_OS_KEYS = [
#         "os",
#         "gcc_version",
#     ]
#     return {k: _SUPPLE_INFO[k] for k in _SUPPLE_OS_KEYS}
# 
# 
# def _supple_python_info():
#     _SUPPLE_PYTHON_KEYS = [
#         "python_version",
#         "torch_version",
#         "is_cuda_available",
#         "pip_version",
#         "pip_packages",
#         "conda_packages",
#     ]
# 
#     return {k: _SUPPLE_INFO[k] for k in _SUPPLE_PYTHON_KEYS}
# 
# 
# def _supple_nvidia_info():
#     _SUPPLE_NVIDIA_KEYS = [
#         "nvidia_gpu_models",
#         "nvidia_driver_version",
#         "cuda_runtime_version",
#         "cudnn_version",
#     ]
# 
#     def replace_key(key):
#         return key
# 
#     def replace_key(key):
#         key = key.replace("_", " ")
#         key = key.replace("nvidia", "NVIDIA")
#         key = key.replace("gpu", "GPU")
#         key = key.replace("cuda", "CUDA")
#         key = key.replace("cudnn", "cuDNN")
#         key = key.replace("driver", "Driver")
#         key = key.replace("runtime", "Runtime")
#         return key
# 
#     return {replace_key(k): _SUPPLE_INFO[k] for k in _SUPPLE_NVIDIA_KEYS}
# 
# 
# if __name__ == "__main__":
#     import mngs
# 
#     # Start
#     CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt)
# 
#     info = mngs.res.get_specs()
#     mngs.io.save(info, "specs.yaml")
# 
#     # Close
#     mngs.gen.close(CONFIG)
# 
# # EOF
# 
# """
# /home/ywatanabe/proj/entrance/mngs/res/_get_specs.py
# """

# test from here --------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import pytest
import numpy as np

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.mngs.resource/_get_specs.py import *

class Test_MainFunctionality:
    def setup_method(self):
        # Setup test fixtures
        pass
    
    def teardown_method(self):
        # Clean up after tests
        pass
    
    def test_basic_functionality(self):
        # Basic test case
        pass
    
    def test_edge_cases(self):
        # Edge case testing
        pass
    
    def test_error_handling(self):
        # Error handling testing
        pass
