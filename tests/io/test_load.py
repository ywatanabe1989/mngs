import pytest
import os
import numpy as np
import pandas as pd
from mngs.io._load import load

@pytest.fixture
def sample_data(tmp_path):
    # Create sample files for testing
    csv_file = tmp_path / 'sample.csv'
    pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}).to_csv(csv_file, index=False)
    
    npy_file = tmp_path / 'sample.npy'
    np.save(npy_file, np.array([1, 2, 3]))
    
    return {'csv': csv_file, 'npy': npy_file}

def test_load_csv(sample_data):
    result = load(str(sample_data['csv']))
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (3, 2)

def test_load_npy(sample_data):
    result = load(str(sample_data['npy']))
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([1, 2, 3]))

def test_load_unsupported_format(tmp_path):
    unsupported_file = tmp_path / 'unsupported.xyz'
    unsupported_file.touch()
    with pytest.raises(ValueError):
        load(str(unsupported_file))
