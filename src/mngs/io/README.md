# [`mngs.io`](https://github.com/ywatanabe1989/mngs/tree/main/src/mngs/io/)
Python input/output utilities.

## Installation
```bash
$ pip install mngs
```


## Quick Start
``` python
import mngs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# .npy
arr = np.array([1, 2, 3])
mngs.io.save(arr, "xxx.npy")
# arr = mngs.io.load("xxx.npy")

# .csv
df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
mngs.io.save(df, "xxx.csv")
# df = mngs.io.load("xxx.csv")

# .pth
tensor = torch.tensor([1, 2, 3])
mngs.io.save(obj, "xxx.pth")
# tensor = mngs.io.load("xxx.pth")

# .pkl
_dict = {"a": 1, "b": 2, "c": [3, 4, 5]} # serializable object like dict, list, ...
mngs.io.save(_dict, "xxx.pkl")
# _dict = mngs.io.load("xxx.pkl")

# .png or .tiff
plt.figure()
plt.plot(np.array([1, 2, 3]))
mngs.io.save(plt, "xxx.png") # or "xxx.tiff"

# .png or .tiff
fig, ax = plt.subplots()
mngs.io.save(plt, "xxx.png") # or "xxx.tiff"

# .yaml
mngs.io.save(_dict, "xxx.yaml")
# _dict = mngs.io.load("xxx.yaml")

# .json
mngs.io.save(_dict, "xxx.json")
# _dict = mngs.io.load("xxx.json")
```

## Contact
Yusuke Watanabe (ywata1989@gmail.com).
