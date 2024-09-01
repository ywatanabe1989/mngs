import os
import pytest
from mngs.io import _path

@pytest.fixture
def temp_dir(tmpdir):
    return str(tmpdir)


def test_get_this_fpath():
    # Test with a regular file
    fpath = _path.get_this_fpath()
    assert os.path.exists(fpath)
    
    # Test with IPython-like environment
    fpath = _path.get_this_fpath(when_ipython="/tmp/test.py")
    assert fpath == "/tmp/test.py"

def test_mk_spath(temp_dir):
    sfname = "test_file.txt"
    spath = _path.mk_spath(sfname, makedirs=True)
    assert os.path.dirname(spath).endswith("test_path")
    assert os.path.basename(spath) == sfname

def test_split_fpath():
    fpath = "/path/to/file/example.txt"
    dirname, fname, ext = _path.split_fpath(fpath)
    assert dirname == "/path/to/file/"
    assert fname == "example"
    assert ext == ".txt"

def test_touch(temp_dir):
    test_file = os.path.join(temp_dir, "test_touch.txt")
    _path.touch(test_file)
    assert os.path.exists(test_file)

def test_find(temp_dir):
    # Create test files
    os.makedirs(os.path.join(temp_dir, "subdir"))
    open(os.path.join(temp_dir, "file1.txt"), "w").close()
    open(os.path.join(temp_dir, "file2.txt"), "w").close()
    open(os.path.join(temp_dir, "subdir", "file3.txt"), "w").close()

    # Test finding all files
    files = _path.find(temp_dir, type="f", exp="*.txt")
    assert len(files) == 3
    assert all(f.endswith(".txt") for f in files)

    # Test finding only in the root directory
    files = _path.find(temp_dir, type="f", exp="file*.txt")
    assert len(files) == 2

def test_find_latest(temp_dir):
    # Create test files
    open(os.path.join(temp_dir, "test_v001.txt"), "w").close()
    open(os.path.join(temp_dir, "test_v002.txt"), "w").close()
    open(os.path.join(temp_dir, "test_v003.txt"), "w").close()

    latest = _path.find_latest(temp_dir, "test", ".txt")
    assert os.path.basename(latest) == "test_v003.txt"

def test_increment_version(temp_dir):
    # Create test files
    open(os.path.join(temp_dir, "test_v001.txt"), "w").close()
    open(os.path.join(temp_dir, "test_v002.txt"), "w").close()

    next_version = _path.increment_version(temp_dir, "test", ".txt")
    assert os.path.basename(next_version) == "test_v003.txt"

    # Test when no files exist
    next_version = _path.increment_version(temp_dir, "new", ".txt")
    assert os.path.basename(next_version) == "new_v001.txt"
