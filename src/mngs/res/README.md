# [`mngs.res`](https://github.com/ywatanabe1989/mngs/tree/main/src/mngs/res/)
A module to gather and manage system resource information.

## Overview
The `mngs.res` module provides utilities for collecting and analyzing system resource information, including CPU, memory, GPU, disk, and network details. This module is particularly useful for system monitoring, performance analysis, and debugging.

## Installation
```bash
$ pip install mngs
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
info = mngs.res.gather_info()

# Save the information to a file
mngs.io.save(info, "info.yaml")

# Print the gathered information
print(info)
```

## Example Output
The `gather_info()` function returns a dictionary containing detailed system information, including:

- System Information (OS, Node Name, Release, Version)
- CPU Info (Cores, Frequencies, Usage)
- Memory Info (RAM and SWAP usage)
- GPU Info (if available)
- Disk Info (Partitions, Read/Write statistics)
- Network Info (Interfaces, Data transfer)

For a full example of the output, please refer to:
https://github.com/ywatanabe1989/mngs/tree/main/src/mngs/res/_gather_info/info.yaml

## Use Cases
- System monitoring and diagnostics
- Performance benchmarking
- Resource usage analysis
- Debugging hardware-related issues
- Generating system reports

## API Reference
- `mngs.res.gather_info()`: Collects comprehensive system resource information

## Contributing
Contributions to improve `mngs.res` are welcome. Please submit pull requests or open issues on the GitHub repository.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions or feedback, please contact Yusuke Watanabe (ywata1989@gmail.com).

For more information and updates, visit the [mngs GitHub repository](https://github.com/ywatanabe1989/mngs).
