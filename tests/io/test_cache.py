import pytest
import os
import shutil
from pathlib import Path
from mngs.io._cache import cache

@pytest.fixture
def temp_cache_dir(tmp_path):
    cache_dir = tmp_path / '.cache' / 'your_app_name'
    cache_dir.mkdir(parents=True, exist_ok=True)
    yield cache_dir
    shutil.rmtree(cache_dir)

def test_cache_save_and_load(temp_cache_dir, monkeypatch):
    monkeypatch.setattr(Path, 'home', lambda: temp_cache_dir.parent.parent)
    
    # Test saving
    var1, var2, var3 = 'x', 1, [1, 2, 3]
    result = cache('test_id', 'var1', 'var2', 'var3')
    assert result == (var1, var2, var3)
    
    # Test loading
    del var1, var2, var3
    loaded_var1, loaded_var2, loaded_var3 = cache('test_id', 'var1', 'var2', 'var3')
    assert loaded_var1 == 'x'
    assert loaded_var2 == 1
    assert loaded_var3 == [1, 2, 3]

def test_cache_file_not_found(temp_cache_dir, monkeypatch):
    monkeypatch.setattr(Path, 'home', lambda: temp_cache_dir.parent.parent)
    
    with pytest.raises(ValueError, match='Cache file not found and not all variables are defined.'):
        cache('non_existent_id', 'var1', 'var2')
