<!-- ---
!-- title: README
!-- author: ywatanabe
!-- date: 2024-11-02 16:50:19
!-- --- -->


# [`mngs.resource`](https://github.com/ywatanabe1989/mngs/tree/main/src/mngs/resource/)

## Overview
The `mngs.resource` module provides comprehensive system resource monitoring and information gathering utilities. It offers an easy-to-use API for collecting detailed information about various system components, including CPU, memory, GPU, disk, and network.

## Installation
```bash
pip install mngs
```

## Features
- Comprehensive system information gathering
- Easy-to-use API for resource monitoring
- Support for CPU, memory, GPU, disk, and network information
- Output in easily readable and parsable formats (YAML, JSON)

## Quick Start
```python
import mngs

# Gather system information
info = mngs.resource.gather_info()

# Save the information to a file
mngs.io.save(info, "system_info.yaml")

# Print specific information
print(f"CPU Usage: {info['cpu']['usage']}%")
print(f"Total RAM: {info['memory']['total']} GB")
print(f"GPU Name: {info['gpu'][0]['name']}")

# Monitor system resources over time
for _ in range(10):
    cpu_usage = mngs.resource.get_cpu_usage()
    mem_usage = mngs.resource.get_memory_usage()
    print(f"CPU: {cpu_usage}%, Memory: {mem_usage}%")
    time.sleep(1)
```

## API Reference
- `mngs.resource.gather_info()`: Collects comprehensive system resource information
- `mngs.resource.get_cpu_info()`: Returns detailed CPU information
- `mngs.resource.get_memory_info()`: Returns memory usage statistics
- `mngs.resource.get_gpu_info()`: Returns GPU information (if available)
- `mngs.resource.get_disk_info()`: Returns disk usage and I/O statistics
- `mngs.resource.get_network_info()`: Returns network interface information
- `mngs.resource.get_cpu_usage()`: Returns current CPU usage percentage
- `mngs.resource.get_memory_usage()`: Returns current memory usage percentage

## Example Output
The `gather_info()` function returns a dictionary containing detailed system information. For a full example of the output, please refer to:
https://github.com/ywatanabe1989/mngs/tree/main/src/mngs/res/_gather_info/info.yaml

## Use Cases
- System monitoring and diagnostics
- Performance benchmarking
- Resource usage analysis
- Debugging hardware-related issues
- Generating system reports
- Automated system health checks

## Performance
The `mngs.resource` module is designed to be lightweight and efficient, with minimal impact on system performance during monitoring.

## Contributing
Contributions to improve `mngs.resource` are welcome. Please submit pull requests or open issues on the GitHub repository.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
Yusuke Watanabe (ywata1989@gmail.com)

For more information and updates, please visit the [mngs GitHub repository](https://github.com/ywatanabe1989/mngs).
