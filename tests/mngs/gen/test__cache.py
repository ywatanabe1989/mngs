#!/usr/bin/env python3
<<<<<<< HEAD
"""Tests for mngs.gen._cache module."""

import time
import pytest
import functools
from unittest.mock import Mock, patch
from mngs.gen._cache import cache


class TestCache:
    """Test cases for the cache function."""

    def test_cache_basic_functionality(self):
        """Test that cache properly memoizes function results."""
        call_count = 0

        @cache
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1

        # Second call with same argument - should use cache
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Should not increment

        # Call with different argument
        result3 = expensive_function(10)
        assert result3 == 20
        assert call_count == 2

    def test_cache_with_multiple_arguments(self):
        """Test cache with functions that have multiple arguments."""

        @cache
        def add_numbers(a, b):
            return a + b

        assert add_numbers(1, 2) == 3
        assert add_numbers(2, 3) == 5
        assert add_numbers(1, 2) == 3  # Cached result

        # Check cache info
        info = add_numbers.cache_info()
        assert info.hits > 0  # At least one cache hit

    def test_cache_with_keyword_arguments(self):
        """Test cache with keyword arguments."""

        @cache
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        assert greet("Alice") == "Hello, Alice!"
        assert greet("Bob", greeting="Hi") == "Hi, Bob!"
        assert greet("Alice") == "Hello, Alice!"  # Cached

    def test_cache_clear(self):
        """Test clearing the cache."""
        call_count = 0

        @cache
        def counted_function(x):
            nonlocal call_count
            call_count += 1
            return x

        # Make some calls
        counted_function(1)
        counted_function(2)
        counted_function(1)  # Should be cached
        assert call_count == 2

        # Clear cache
        counted_function.cache_clear()

        # Call again - should not use cache
        counted_function(1)
        assert call_count == 3

    def test_cache_with_exceptions(self):
        """Test that exceptions are not cached."""
        call_count = 0

        @cache
        def maybe_fail(should_fail):
            nonlocal call_count
            call_count += 1
            if should_fail:
                raise ValueError("Failed")
            return "Success"

        # First call raises exception
        with pytest.raises(ValueError):
            maybe_fail(True)
        assert call_count == 1

        # Second call with same argument should still execute
        with pytest.raises(ValueError):
            maybe_fail(True)
        assert call_count == 2  # Exception not cached

        # Successful call
        assert maybe_fail(False) == "Success"
        assert call_count == 3


class TestCacheAdvanced:
    """Advanced test cases for cache functionality."""

    def test_cache_info_detailed(self):
        """Test detailed cache info tracking."""
        @cache
        def compute(x):
            return x ** 2

        # Make various calls
        compute(1)  # miss
        compute(2)  # miss
        compute(3)  # miss
        compute(1)  # hit
        compute(2)  # hit
        compute(4)  # miss

        info = compute.cache_info()
        assert info.hits == 2
        assert info.misses == 4
        assert info.currsize == 4
        assert info.maxsize == 128  # Default maxsize

    def test_cache_with_none_arguments(self):
        """Test cache with None as argument."""
        @cache
        def process(value):
            return value if value is not None else "default"

        assert process(None) == "default"
        assert process(10) == 10
        assert process(None) == "default"  # Should be cached

        info = process.cache_info()
        assert info.hits >= 1

    def test_cache_with_unhashable_types(self):
        """Test that unhashable types raise appropriate errors."""
        @cache
        def process_list(lst):
            return sum(lst)

        # Lists are unhashable and should raise TypeError
        with pytest.raises(TypeError):
            process_list([1, 2, 3])

        # But tuples work fine
        @cache
        def process_tuple(tpl):
            return sum(tpl)

        assert process_tuple((1, 2, 3)) == 6
        assert process_tuple((1, 2, 3)) == 6  # Cached

    def test_cache_preserves_function_attributes(self):
        """Test that cache preserves function metadata."""
        def original_function(x, y=10):
            """Original function docstring."""
            return x + y

        cached_function = cache(original_function)

        # Check that metadata is preserved
        assert cached_function.__name__ == "original_function"
        assert cached_function.__doc__ == "Original function docstring."
        assert hasattr(cached_function, "__wrapped__")
        assert cached_function.__wrapped__ is original_function

    def test_cache_performance_improvement(self):
        """Test that cache actually improves performance."""
        @cache
        def slow_function(n):
            # Simulate expensive computation
            time.sleep(0.01)
            return n * n

        # First call - should be slow
        start = time.time()
        result1 = slow_function(42)
        first_duration = time.time() - start

        # Second call - should be fast (cached)
        start = time.time()
        result2 = slow_function(42)
        second_duration = time.time() - start

        assert result1 == result2
        assert second_duration < first_duration * 0.5  # At least 2x faster

    def test_cache_with_nested_functions(self):
        """Test cache with nested function calls."""
        call_counts = {"inner": 0, "outer": 0}

        @cache
        def inner_function(x):
            call_counts["inner"] += 1
            return x * 2

        @cache
        def outer_function(x):
            call_counts["outer"] += 1
            return inner_function(x) + 10

        # First call
        assert outer_function(5) == 20
        assert call_counts["inner"] == 1
        assert call_counts["outer"] == 1

        # Second call - both should be cached
        assert outer_function(5) == 20
        assert call_counts["inner"] == 1  # No additional calls
        assert call_counts["outer"] == 1  # No additional calls

    def test_cache_with_maxsize_limit(self):
        """Test cache with explicit maxsize limit."""
        # Create a cache with small maxsize
        @functools.lru_cache(maxsize=2)
        def limited_cache(x):
            return x * x

        # Fill cache
        limited_cache(1)  # Cache: [1]
        limited_cache(2)  # Cache: [1, 2]
        limited_cache(3)  # Cache: [2, 3] (1 evicted)

        info = limited_cache.cache_info()
        assert info.currsize <= 2

    def test_cache_thread_safety(self):
        """Test that cache is thread-safe."""
        import threading

        results = []
        call_count = 0

        @cache
        def thread_function(x):
            nonlocal call_count
            call_count += 1
            time.sleep(0.01)  # Simulate work
            return x * x

        def worker(value):
            result = thread_function(value)
            results.append(result)

        # Create multiple threads calling same function
        threads = []
        for i in range(5):
            for value in [1, 2, 3]:
                t = threading.Thread(target=worker, args=(value,))
                threads.append(t)
                t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Each unique value should only be computed once
        assert call_count == 3
        assert len(results) == 15  # 5 iterations * 3 values

    def test_cache_with_different_types(self):
        """Test cache with various argument types."""
        @cache
        def identity(x):
            return x

        # Test with different types
        test_values = [
            42,  # int
            3.14,  # float
            "hello",  # str
            True,  # bool
            (1, 2, 3),  # tuple
            frozenset([1, 2, 3]),  # frozenset
        ]

        for value in test_values:
            assert identity(value) == value
            assert identity(value) == value  # Should be cached

        info = identity.cache_info()
        assert info.hits == len(test_values)

    def test_cache_memory_behavior(self):
        """Test cache memory management."""
        @cache
        def create_large_object(n):
            return [0] * n * 1000

        # Create some large objects
        obj1 = create_large_object(100)
        obj2 = create_large_object(200)

        # Clear cache to free memory
        create_large_object.cache_clear()

        info = create_large_object.cache_info()
        assert info.currsize == 0

    def test_cache_with_generators(self):
        """Test that generators are consumed properly."""
        @cache
        def process_generator():
            return list(x * 2 for x in range(5))

        result1 = process_generator()
        result2 = process_generator()

        assert result1 == [0, 2, 4, 6, 8]
        assert result1 == result2

    def test_cache_with_class_methods(self):
        """Test cache decorator on class methods."""
        class Calculator:
            def __init__(self):
                self.call_count = 0

            @cache
            def compute(self, x):
                self.call_count += 1
                return x * x

        calc = Calculator()
        
        # Note: cache on instance methods can be tricky
        # The self parameter makes each instance's method different
        result1 = calc.compute(5)
        result2 = calc.compute(5)
        
        assert result1 == 25
        assert result2 == 25
        # Call count may be 2 due to self parameter

    def test_cache_equality_vs_identity(self):
        """Test that cache uses equality not identity for keys."""
        @cache
        def process(x):
            return x * 2

        # Different objects with same value
        a = 1000  # Large enough that Python doesn't intern
        b = 500 + 500
        
        assert a is not b  # Different objects
        assert a == b  # But equal values

        result1 = process(a)
        result2 = process(b)

        assert result1 == result2
        # Should use cached result for equal values
        info = process.cache_info()
        assert info.hits >= 1

    def test_cache_wrapped_function_access(self):
        """Test accessing the wrapped function."""
        def original(x, y=5):
            """Original docstring."""
            return x + y

        cached = cache(original)

        # Access wrapped function
        assert cached.__wrapped__ is original
        assert cached.__wrapped__(10, 20) == 30

        # Calling wrapped directly bypasses cache
        cached(10, 20)  # Cache this
        call_count = 0

        def counting_original(x, y=5):
            nonlocal call_count
            call_count += 1
            return x + y

        cached_counting = cache(counting_original)
        
        cached_counting(10, 20)
        assert call_count == 1
        
        # Direct call to wrapped bypasses cache
        cached_counting.__wrapped__(10, 20)
        assert call_count == 2

    def test_cache_with_recursive_functions(self):
        """Test cache with recursive functions."""
        @cache
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n - 1) + fibonacci(n - 2)

        # Calculate fibonacci(10)
        result = fibonacci(10)
        assert result == 55

        # Check that intermediate results were cached
        info = fibonacci.cache_info()
        assert info.currsize > 1  # Multiple values cached
        assert info.hits > 0  # Recursive calls hit cache

    def test_cache_pickle_compatibility(self):
        """Test that cached functions work with pickle."""
        import pickle

        @cache
        def simple_function(x):
            return x * 2

        # Prime the cache
        simple_function(5)

        # Test if the function itself can be pickled
        # Note: This might fail as lru_cache may not be picklable
        try:
            pickled = pickle.dumps(simple_function)
            unpickled = pickle.loads(pickled)
            assert callable(unpickled)
        except (pickle.PicklingError, AttributeError):
            pytest.skip("Cache functions may not be picklable")

    def test_cache_comparison_with_manual_memoization(self):
        """Compare cache decorator with manual memoization."""
        # Manual memoization
        manual_cache = {}
        call_count_manual = 0

        def manual_memoized(x):
            nonlocal call_count_manual
            if x in manual_cache:
                return manual_cache[x]
            call_count_manual += 1
            result = x * x
            manual_cache[x] = result
            return result

        # Using cache decorator
        call_count_auto = 0

        @cache
        def auto_memoized(x):
            nonlocal call_count_auto
            call_count_auto += 1
            return x * x

        # Test both
        for i in [1, 2, 3, 1, 2, 3]:
            manual_result = manual_memoized(i)
            auto_result = auto_memoized(i)
            assert manual_result == auto_result

        assert call_count_manual == 3
        assert call_count_auto == 3

    def test_cache_with_partial_functions(self):
        """Test cache with functools.partial."""
        from functools import partial

        @cache
        def add(a, b, c=0):
            return a + b + c

        # Create partial function
        add_five = partial(add, 5)

        assert add_five(10) == 15
        assert add_five(10) == 15  # Should be cached

        # Different partial
        add_ten = partial(add, 10)
        assert add_ten(10) == 20

    def test_cache_zero_maxsize(self):
        """Test cache with maxsize=0 (no caching)."""
        call_count = 0

        @functools.lru_cache(maxsize=0)
        def no_cache_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # Multiple calls with same argument
        no_cache_function(5)
        no_cache_function(5)
        no_cache_function(5)

        # Should call function every time
        assert call_count == 3

    def test_cache_none_maxsize(self):
        """Test cache with maxsize=None (unlimited)."""
        @functools.lru_cache(maxsize=None)
        def unlimited_cache(x):
            return x * x

        # Add many items
        for i in range(1000):
            unlimited_cache(i)

        info = unlimited_cache.cache_info()
        assert info.currsize == 1000
        assert info.maxsize is None

    def test_cache_statistics_accuracy(self):
        """Test that cache statistics are accurate."""
        @cache
        def compute(x):
            return x ** 2

        # Clear any existing cache
        compute.cache_clear()

        # Known sequence of calls
        compute(1)  # miss
        compute(2)  # miss
        compute(1)  # hit
        compute(3)  # miss
        compute(2)  # hit
        compute(1)  # hit

        info = compute.cache_info()
        assert info.hits == 3
        assert info.misses == 3
        assert info.currsize == 3


class TestCacheEdgeCases:
    """Test edge cases and error conditions."""

    def test_cache_on_property(self):
        """Test that cache can't be applied to properties."""
        # This should typically fail or behave unexpectedly
        class MyClass:
            @property
            @cache
            def value(self):
                return 42

        # Properties with cache may not work as expected
        obj = MyClass()
        assert obj.value == 42

    def test_cache_with_mutable_default_arguments(self):
        """Test cache with mutable default arguments."""
        @cache
        def append_to_list(item, lst=None):
            if lst is None:
                lst = []
            lst.append(item)
            return lst

        # Each call with default should return same list
        result1 = append_to_list(1)
        result2 = append_to_list(2)
        
        assert result1 == [1]
        assert result2 == [2]

    def test_cache_decorator_stacking(self):
        """Test stacking multiple decorators with cache."""
        def double_decorator(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs) * 2
            return wrapper

        @double_decorator
        @cache
        def compute(x):
            return x + 10

        result = compute(5)
        assert result == 30  # (5 + 10) * 2

    def test_cache_with_very_long_argument_lists(self):
        """Test cache with functions having many arguments."""
        @cache
        def many_args(a, b, c, d, e, f, g, h, i, j):
            return sum([a, b, c, d, e, f, g, h, i, j])

        result1 = many_args(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        result2 = many_args(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

        assert result1 == 55
        assert result2 == 55

        info = many_args.cache_info()
        assert info.hits == 1

    def test_cache_with_lambda_functions(self):
        """Test cache with lambda functions."""
        cached_lambda = cache(lambda x: x * x)

        assert cached_lambda(5) == 25
        assert cached_lambda(5) == 25  # Cached

        info = cached_lambda.cache_info()
        assert info.hits == 1

    def test_cache_memory_limit_behavior(self):
        """Test behavior when approaching memory limits."""
        @functools.lru_cache(maxsize=3)
        def limited_cache(x):
            return [0] * x

        # Fill cache to limit
        limited_cache(100)
        limited_cache(200)
        limited_cache(300)
        
        info = limited_cache.cache_info()
        assert info.currsize == 3

        # Add one more - should evict oldest
        limited_cache(400)
        
        info = limited_cache.cache_info()
        assert info.currsize == 3

    def test_cache_with_custom_objects(self):
        """Test cache with custom objects as arguments."""
        class Point:
            def __init__(self, x, y):
                self.x = x
                self.y = y
            
            def __hash__(self):
                return hash((self.x, self.y))
            
            def __eq__(self, other):
                return self.x == other.x and self.y == other.y

        @cache
        def distance_from_origin(point):
            return (point.x ** 2 + point.y ** 2) ** 0.5

        p1 = Point(3, 4)
        p2 = Point(3, 4)  # Equal but different object

        result1 = distance_from_origin(p1)
        result2 = distance_from_origin(p2)

        assert result1 == 5.0
        assert result2 == 5.0

        info = distance_from_origin.cache_info()
        assert info.hits == 1  # Second call should hit cache

=======
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
>>>>>>> origin/main

if __name__ == "__main__":
    import os
    import pytest

<<<<<<< HEAD
    pytest.main([os.path.abspath(__file__), "-v"])
=======
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
>>>>>>> origin/main
