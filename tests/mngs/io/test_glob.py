import pytest
from mngs.io._glob import glob
import os
import tempfile

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname

def test_glob_natural_sorting(temp_dir):
    # Create test files
    for i in range(1, 12):
        open(os.path.join(temp_dir, f'file{i}.txt'), 'w').close()
    
    result = glob(os.path.join(temp_dir, '*.txt'))
    expected = [os.path.join(temp_dir, f'file{i}.txt') for i in range(1, 12)]
    assert result == expected

def test_glob_curly_brace_expansion(temp_dir):
    # Create test directories and files
    os.makedirs(os.path.join(temp_dir, 'a'))
    os.makedirs(os.path.join(temp_dir, 'b'))
    for dir in ['a', 'b']:
        for i in range(1, 3):
            open(os.path.join(temp_dir, dir, f'file{i}.txt'), 'w').close()
    
    result = glob(os.path.join(temp_dir, '{a,b}/*.txt'))
    expected = [
        os.path.join(temp_dir, 'a', 'file1.txt'),
        os.path.join(temp_dir, 'a', 'file2.txt'),
        os.path.join(temp_dir, 'b', 'file1.txt'),
        os.path.join(temp_dir, 'b', 'file2.txt')
    ]
    assert result == expected
