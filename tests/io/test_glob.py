import pytest
from mngs.io._glob import glob
import os

def test_glob(tmp_path):
    # Create test files
    (tmp_path / 'file1.txt').touch()
    (tmp_path / 'file2.txt').touch()
    (tmp_path / 'file3.csv').touch()
    
    # Test basic glob
    result = glob(str(tmp_path / '*.txt'))
    assert len(result) == 2
    assert all(f.endswith('.txt') for f in result)
    
    # Test glob with multiple patterns
    result = glob(str(tmp_path / '*.{txt,csv}'))
    assert len(result) == 3
    
    # Test glob with non-existing pattern
    result = glob(str(tmp_path / '*.json'))
    assert len(result) == 0
