
import pytest
import os
import tempfile
import numpy as np
import pandas as pd
import json
import yaml
from mngs.io import load, glob

def test_load_csv():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        tmp.write('a,b,c\n1,2,3\n4,5,6')
        tmp.flush()
        df = load(tmp.name)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 3)
        assert list(df.columns) == ['a', 'b', 'c']

def test_load_npy():
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
        np.save(tmp.name, np.array([1, 2, 3]))
        arr = load(tmp.name)
        assert isinstance(arr, np.ndarray)
        assert np.array_equal(arr, np.array([1, 2, 3]))

def test_load_json():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        json.dump({"key": "value"}, tmp)
        tmp.flush()
        data = load(tmp.name)
        assert isinstance(data, dict)
        assert data == {"key": "value"}

def test_load_yaml():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
        yaml.dump({"key": "value"}, tmp)
        tmp.flush()
        data = load(tmp.name)
        assert isinstance(data, dict)
        assert data == {"key": "value"}

def test_load_txt():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
        tmp.write("line1\nline2\nline3")
        tmp.flush()
        lines = load(tmp.name)
        assert isinstance(lines, list)
        assert lines == ["line1", "line2", "line3"]

def test_glob():
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            open(os.path.join(tmpdir, f'file{i}.txt'), 'w').close()
        files = glob(os.path.join(tmpdir, '*.txt'))
        assert len(files) == 3
        assert all(f.endswith('.txt') for f in files)

def test_load_unsupported():
    with tempfile.NamedTemporaryFile(suffix='.unsupported', delete=False) as tmp:
        tmp.write(b"some data")
        tmp.flush()
        with pytest.raises(ValueError):
            load(tmp.name)

# Add more tests for other load functions as needed
