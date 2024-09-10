import os

import numpy as np
import pandas as pd
import pytest
from mngs.io._save import save


@pytest.fixture
def sample_data():
    return {
        "df": pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}),
        "array": np.array([1, 2, 3]),
        "dict": {"key": "value"},
    }


def test_save_csv(tmp_path, sample_data):
    file_path = tmp_path / "test.csv"
    save(sample_data["df"], str(file_path), index=False)
    assert os.path.exists(file_path)
    loaded_df = pd.read_csv(file_path)
    pd.testing.assert_frame_equal(loaded_df, sample_data["df"])


def test_save_npy(tmp_path, sample_data):
    file_path = tmp_path / "test.npy"
    save(sample_data["array"], str(file_path))
    assert os.path.exists(file_path)
    loaded_array = np.load(file_path)
    np.testing.assert_array_equal(loaded_array, sample_data["array"])


def test_save_pkl(tmp_path, sample_data):
    file_path = tmp_path / "test.pkl"
    save(sample_data["dict"], str(file_path))
    assert os.path.exists(file_path)
    loaded_dict = pd.read_pickle(file_path)
    assert loaded_dict == sample_data["dict"]


def test_save_unsupported_format(tmp_path, sample_data):
    file_path = tmp_path / "test.unsupported"
    with pytest.raises(
        ValueError,
        match=f"Unsupported file format. {file_path} was not saved.",
    ):
        save(sample_data["df"], str(file_path))
