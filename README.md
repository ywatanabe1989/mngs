## Installation
``` bash
$ pip install -y mngs
```

## mngs.general.save
``` python
import mngs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## numpy
arr = np.arange(10)
mngs.general.save(arr, 'spath.npy')

## pandas
df = pd.DataFrame(arr)
mngs.general.save(df, 'spath.csv')

## matplotlib
fig, ax = plt.subplots()
ax.plot(arr)
mngs.general.save(fig, 'spath.png)
```

## mngs.general.load
``` python
import mngs
arr = mngs.general.load('spath.npy')
arr = mngs.general.load('spath.mat')
df = mngs.general.load('spath.npy')
yaml_dict = mngs.general.load('spath.yaml')
hdf5_dict = mngs.general.load('spath.hdf5')
```

## mngs.general.fix_seeds

``` python
import mngs
import os
import random
import numpy as np
import torch

mngs.general.fix_seeds(os=os, random=random, np=np, torch=torch, tf=None, seed=42)
```

## mngs.general.tee
``` python
import sys
sys.stdout, sys.stderr = tee(sys)
print("abc")  # also wrriten in stdout
print(1 / 0)  # also wrriten in stderr
```

## mngs.plt.configure_mpl
``` python
configure_mpl(
    plt,
    dpi=100,
    figsize=(16.2, 10),
    figscale=1.0,
    fontsize=16,
    labelsize="same",
    legendfontsize="xx-small",
    tick_size="auto",
    tick_width="auto",
    hide_spines=False,
)
```

## mngs.plt.ax_*
- mngs.plt.ax_extend
- mngs.plt.ax_scientific_notation
- mngs.plt.ax_set_position

