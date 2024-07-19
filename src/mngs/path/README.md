# [`mngs.path`](https://github.com/ywatanabe1989/mngs/tree/main/src/mngs/path/)

## Installation
```bash
$ pip install mngs
```

## Quick Start
``` python
import mngs

fpath = mngs.path.this_path() # "/tmp/fake.py"
spath = mngs.path.spath() # '/tmp/fake-ywatanabe/.'
dir, fname, ext = mngs.path.split(fpath) # ("/tmp/", "fake", "py")

# Find
mngs.path.find_dir(".", "path") # [./src/mngs/path]
mngs.path.find_file(".", "*wavelet.py") # ['./src/mngs/dsp/_wavelet.py']

# Versioning
mngs.path.find_git_root() # '/home/ywatanabe/proj/mngs'
# mngs.find_latest()
# mngs.path.increment_version()
```

## Contact
Yusuke Watanabe (ywata1989@gmail.com).
