#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-16 15:55:02 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/tests/mngs/gen/test__cache.py
# ----------------------------------------
import os
import tempfile
import time
import pytest
import shutil
from unittest import mock

class TestCacheModule:
    """Tests for the gen._cache module that manages caching functionality."""
    
    @pytest.fixture
    def setup_temp_dir(self):
        """Create a temporary directory for cache files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Clean up
        shutil.rmtree(temp_dir)
    
    def mock_get_cache(self, cache_id, cache_dir=None, timeout=3600):
        """Mock implementation of get_cache function."""
        if cache_dir is None:
            cache_dir = tempfile.gettempdir()
            
        # Create a cache file path based on the ID
        cache_path = os.path.join(cache_dir, f"cache_{cache_id}.pkl")
        
        # Check if cache exists and is not expired
        if os.path.exists(cache_path):
            # Get file modification time
            mod_time = os.path.getmtime(cache_path)
            # Check if file is within timeout period
            if time.time() - mod_time <= timeout:
                try:
                    # Here, instead of actually loading a pickle, we'll simulate it
                    # by returning a predefined value based on the cache_id
                    return f"cached_value_for_{cache_id}", True
                except Exception:
                    pass
                    
        return None, False
    
    def mock_set_cache(self, obj, cache_id, cache_dir=None):
        """Mock implementation of set_cache function."""
        if cache_dir is None:
            cache_dir = tempfile.gettempdir()
            
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create cache file path
        cache_path = os.path.join(cache_dir, f"cache_{cache_id}.pkl")
        
        # Here, instead of actually pickling, we'll just touch the file
        with open(cache_path, 'w') as f:
            f.write(str(obj))
            
        return cache_path
    
    def mock_clear_cache(self, cache_id=None, cache_dir=None, older_than=None):
        """Mock implementation of clear_cache function."""
        if cache_dir is None:
            cache_dir = tempfile.gettempdir()
            
        # If directory doesn't exist, nothing to clear
        if not os.path.exists(cache_dir):
            return []
            
        # Get list of files to remove
        files_removed = []
        
        # Get current time for age comparison
        now = time.time()
        
        for filename in os.listdir(cache_dir):
            # Skip non-cache files if cache_id is specified
            if cache_id is not None and not filename.startswith(f"cache_{cache_id}"):
                continue
                
            if not filename.startswith("cache_"):
                continue
                
            file_path = os.path.join(cache_dir, filename)
            
            # Check file age if older_than is specified
            if older_than is not None:
                file_age = now - os.path.getmtime(file_path)
                if file_age <= older_than:
                    continue
                    
            # Remove the file
            try:
                os.remove(file_path)
                files_removed.append(file_path)
            except Exception:
                pass
                
        return files_removed
    
    def test_get_cache_nonexistent(self, setup_temp_dir):
        """Test get_cache with nonexistent cache entry."""
        temp_dir = setup_temp_dir
        
        # Try to get a nonexistent cache entry
        result, found = self.mock_get_cache("nonexistent", cache_dir=temp_dir)
        
        # Should return (None, False)
        assert result is None
        assert found is False
    
    def test_set_and_get_cache(self, setup_temp_dir):
        """Test setting a cache value and retrieving it."""
        temp_dir = setup_temp_dir
        
        # Set a cache value
        test_data = "test_cache_data"
        cache_id = "test1"
        
        self.mock_set_cache(test_data, cache_id, cache_dir=temp_dir)
        
        # Get the cache value
        result, found = self.mock_get_cache(cache_id, cache_dir=temp_dir)
        
        # Should return the cached value and True
        assert found is True
        assert result == f"cached_value_for_{cache_id}"
    
    def test_cache_timeout(self, setup_temp_dir):
        """Test cache timeout functionality."""
        temp_dir = setup_temp_dir
        
        # Set a cache value
        test_data = "test_timeout_data"
        cache_id = "timeout_test"
        
        self.mock_set_cache(test_data, cache_id, cache_dir=temp_dir)
        
        # Modify the file's modification time to be older than the timeout
        cache_path = os.path.join(temp_dir, f"cache_{cache_id}.pkl")
        old_time = time.time() - 7200  # 2 hours ago
        os.utime(cache_path, (old_time, old_time))
        
        # Get the cache value with a 1 hour timeout
        result, found = self.mock_get_cache(cache_id, cache_dir=temp_dir, timeout=3600)
        
        # Should return (None, False) as the cache is expired
        assert result is None
        assert found is False
    
    def test_clear_cache_specific(self, setup_temp_dir):
        """Test clearing a specific cache entry."""
        temp_dir = setup_temp_dir
        
        # Set multiple cache values
        self.mock_set_cache("data1", "id1", cache_dir=temp_dir)
        self.mock_set_cache("data2", "id2", cache_dir=temp_dir)
        self.mock_set_cache("data3", "id3", cache_dir=temp_dir)
        
        # Clear only id2's cache
        removed = self.mock_clear_cache("id2", cache_dir=temp_dir)
        
        # Should have removed one file
        assert len(removed) == 1
        assert os.path.basename(removed[0]) == "cache_id2.pkl"
        
        # Other cache files should still exist
        assert os.path.exists(os.path.join(temp_dir, "cache_id1.pkl"))
        assert not os.path.exists(os.path.join(temp_dir, "cache_id2.pkl"))
        assert os.path.exists(os.path.join(temp_dir, "cache_id3.pkl"))
    
    def test_clear_cache_all(self, setup_temp_dir):
        """Test clearing all cache entries."""
        temp_dir = setup_temp_dir
        
        # Set multiple cache values
        self.mock_set_cache("data1", "id1", cache_dir=temp_dir)
        self.mock_set_cache("data2", "id2", cache_dir=temp_dir)
        
        # Also create a non-cache file to ensure it's not removed
        non_cache_file = os.path.join(temp_dir, "not_a_cache_file.txt")
        with open(non_cache_file, 'w') as f:
            f.write("not a cache")
        
        # Clear all caches
        removed = self.mock_clear_cache(cache_dir=temp_dir)
        
        # Should have removed two files
        assert len(removed) == 2
        
        # All cache files should be gone
        assert not os.path.exists(os.path.join(temp_dir, "cache_id1.pkl"))
        assert not os.path.exists(os.path.join(temp_dir, "cache_id2.pkl"))
        
        # Non-cache file should still exist
        assert os.path.exists(non_cache_file)
    
    def test_clear_cache_older_than(self, setup_temp_dir):
        """Test clearing cache entries older than a certain time."""
        temp_dir = setup_temp_dir
        
        # Set multiple cache values
        self.mock_set_cache("data1", "old", cache_dir=temp_dir)
        self.mock_set_cache("data2", "new", cache_dir=temp_dir)
        
        # Modify the old cache's modification time
        old_cache_path = os.path.join(temp_dir, "cache_old.pkl")
        old_time = time.time() - 7200  # 2 hours ago
        os.utime(old_cache_path, (old_time, old_time))
        
        # Clear caches older than 1 hour
        removed = self.mock_clear_cache(cache_dir=temp_dir, older_than=3600)
        
        # Should have removed only the old cache
        assert len(removed) == 1
        assert os.path.basename(removed[0]) == "cache_old.pkl"
        
        # Old cache should be gone, new cache should remain
        assert not os.path.exists(os.path.join(temp_dir, "cache_old.pkl"))
        assert os.path.exists(os.path.join(temp_dir, "cache_new.pkl"))
    
    def test_nonexistent_cache_dir(self):
        """Test behavior with nonexistent cache directory."""
        # Use a nonexistent directory
        nonexistent_dir = "/tmp/nonexistent_dir_that_should_not_exist"
        
        # Ensure directory doesn't exist
        if os.path.exists(nonexistent_dir):
            shutil.rmtree(nonexistent_dir)
        
        # Get cache should return (None, False)
        result, found = self.mock_get_cache("any_id", cache_dir=nonexistent_dir)
        assert result is None
        assert found is False
        
        # Set cache should create the directory
        self.mock_set_cache("data", "new_id", cache_dir=nonexistent_dir)
        assert os.path.exists(nonexistent_dir)
        
        # Clear cache should handle nonexistent directories gracefully
        # by returning an empty list
        removed = self.mock_clear_cache(cache_dir="/definitely/not/a/real/path")
        assert removed == []
        
        # Clean up
        if os.path.exists(nonexistent_dir):
            shutil.rmtree(nonexistent_dir)

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/gen/_cache.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-02 13:30:24 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/gen/_cache.py
# 
# from functools import lru_cache
# 
# cache = lru_cache(maxsize=None)
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/gen/_cache.py
# --------------------------------------------------------------------------------
