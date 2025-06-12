#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-06-02 15:40:00 (ywatanabe)"
# File: ./mngs_repo/tests/mngs/reproduce/test__gen_timestamp.py

"""Tests for gen_timestamp function."""

import re
import time
from datetime import datetime
from unittest.mock import patch

import pytest

from mngs.reproduce._gen_timestamp import gen_timestamp, timestamp


def test_gen_timestamp_basic():
    """Test basic timestamp generation."""
    ts = gen_timestamp()
    
    # Check format: YYYY-MMDD-HHMM
    assert isinstance(ts, str)
    assert len(ts) == 14  # 4+1+4+1+4 = 14 characters
    
    # Check format pattern
    assert re.match(r'^\d{4}-\d{4}-\d{4}$', ts)


def test_gen_timestamp_format_details():
    """Test timestamp format matches expected pattern."""
    ts = gen_timestamp()
    
    # Split into parts
    parts = ts.split('-')
    assert len(parts) == 3
    
    # Year part
    year = parts[0]
    assert len(year) == 4
    assert year.isdigit()
    assert 2000 <= int(year) <= 2100  # Reasonable year range
    
    # Month-day part
    monthday = parts[1]
    assert len(monthday) == 4
    assert monthday.isdigit()
    month = int(monthday[:2])
    day = int(monthday[2:])
    assert 1 <= month <= 12
    assert 1 <= day <= 31
    
    # Hour-minute part
    hourmin = parts[2]
    assert len(hourmin) == 4
    assert hourmin.isdigit()
    hour = int(hourmin[:2])
    minute = int(hourmin[2:])
    assert 0 <= hour <= 23
    assert 0 <= minute <= 59


def test_gen_timestamp_current_time():
    """Test that timestamp reflects current time."""
    # Get timestamp
    ts = gen_timestamp()
    
    # Get current time
    now = datetime.now()
    
    # Parse timestamp
    parts = ts.split('-')
    ts_year = int(parts[0])
    ts_month = int(parts[1][:2])
    ts_day = int(parts[1][2:])
    ts_hour = int(parts[2][:2])
    ts_minute = int(parts[2][2:])
    
    # Should match current time (allowing 1 minute difference)
    assert ts_year == now.year
    assert ts_month == now.month
    assert ts_day == now.day
    assert ts_hour == now.hour
    assert abs(ts_minute - now.minute) <= 1  # Allow 1 minute difference


def test_gen_timestamp_consistency():
    """Test timestamp consistency within same minute."""
    ts1 = gen_timestamp()
    time.sleep(0.1)  # Small delay
    ts2 = gen_timestamp()
    
    # If called within same minute, should be identical
    # (This test might fail if run exactly at minute boundary)
    # So we'll check they're very close
    if ts1 != ts2:
        # Must have crossed a minute boundary
        parts1 = ts1.split('-')
        parts2 = ts2.split('-')
        
        # Year, month, day should be same
        assert parts1[0] == parts2[0]  # Year
        assert parts1[1] == parts2[1]  # Month-day
        
        # Hour should be same or differ by 1
        hour1 = int(parts1[2][:2])
        hour2 = int(parts2[2][:2])
        assert abs(hour2 - hour1) <= 1


def test_gen_timestamp_mocked():
    """Test timestamp with mocked datetime."""
    # Mock specific datetime
    mock_dt = datetime(2025, 5, 31, 14, 30, 45)
    
    with patch('mngs.reproduce._gen_timestamp._datetime') as mock_datetime:
        mock_datetime.now.return_value = mock_dt
        
        ts = gen_timestamp()
        assert ts == "2025-0531-1430"


def test_gen_timestamp_edge_cases():
    """Test timestamp generation at edge cases."""
    # Test various edge case times
    edge_cases = [
        (datetime(2025, 1, 1, 0, 0, 0), "2025-0101-0000"),      # New year
        (datetime(2025, 12, 31, 23, 59, 59), "2025-1231-2359"), # End of year
        (datetime(2025, 2, 28, 12, 30, 0), "2025-0228-1230"),   # End of Feb
        (datetime(2024, 2, 29, 12, 30, 0), "2024-0229-1230"),   # Leap day
        (datetime(2025, 10, 5, 9, 5, 0), "2025-1005-0905"),     # Single digit formatting
    ]
    
    for mock_dt, expected in edge_cases:
        with patch('mngs.reproduce._gen_timestamp._datetime') as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            
            ts = gen_timestamp()
            assert ts == expected


def test_gen_timestamp_alias():
    """Test that timestamp alias works correctly."""
    # timestamp should be an alias for gen_timestamp
    assert timestamp is gen_timestamp
    
    # Should work the same way
    ts1 = gen_timestamp()
    ts2 = timestamp()
    
    # Format should be identical
    assert re.match(r'^\d{4}-\d{4}-\d{4}$', ts2)
    assert len(ts2) == 14


def test_gen_timestamp_uniqueness_over_time():
    """Test timestamp uniqueness when called over time."""
    timestamps = []
    
    # Collect timestamps over 2+ minutes
    # In practice, we'll simulate this
    times = [
        datetime(2025, 5, 31, 14, 30),
        datetime(2025, 5, 31, 14, 31),
        datetime(2025, 5, 31, 14, 32),
        datetime(2025, 5, 31, 15, 0),
        datetime(2025, 6, 1, 0, 0),
    ]
    
    for mock_time in times:
        with patch('mngs.reproduce._gen_timestamp._datetime') as mock_datetime:
            mock_datetime.now.return_value = mock_time
            timestamps.append(gen_timestamp())
    
    # All should be unique
    assert len(timestamps) == len(set(timestamps))
    
    # Should be in chronological order
    assert timestamps == sorted(timestamps)


def test_gen_timestamp_file_naming():
    """Test timestamp usage in file naming."""
    ts = gen_timestamp()
    
    # Should work well in filenames
    filename = f"data_{ts}.csv"
    assert re.match(r'^data_\d{4}-\d{4}-\d{4}\.csv$', filename)
    
    # Should be sortable
    filenames = []
    times = [
        datetime(2025, 5, 31, 14, 30),
        datetime(2025, 5, 31, 14, 31),
        datetime(2025, 5, 31, 14, 32),
    ]
    
    for mock_time in times:
        with patch('mngs.reproduce._gen_timestamp._datetime') as mock_datetime:
            mock_datetime.now.return_value = mock_time
            filenames.append(f"log_{gen_timestamp()}.txt")
    
    # Alphabetical sort should match chronological order
    assert filenames == sorted(filenames)


def test_gen_timestamp_performance():
    """Test timestamp generation performance."""
    start = time.time()
    
    # Generate many timestamps
    timestamps = [gen_timestamp() for _ in range(1000)]
    
    duration = time.time() - start
    
    # Should be very fast (< 0.1 seconds for 1000)
    assert duration < 0.1
    
    # All should have correct format
    for ts in timestamps:
        assert re.match(r'^\d{4}-\d{4}-\d{4}$', ts)


def test_gen_timestamp_thread_safety():
    """Test timestamp generation is thread-safe."""
    import threading
    
    timestamps = []
    lock = threading.Lock()
    
    def generate_timestamp():
        ts = gen_timestamp()
        with lock:
            timestamps.append(ts)
    
    # Create multiple threads
    threads = []
    for _ in range(10):
        t = threading.Thread(target=generate_timestamp)
        threads.append(t)
        t.start()
    
    # Wait for all threads
    for t in threads:
        t.join()
    
    # All should have correct format
    for ts in timestamps:
        assert re.match(r'^\d{4}-\d{4}-\d{4}$', ts)


def test_gen_timestamp_examples():
    """Test examples from docstring."""
    # Basic usage
    ts = gen_timestamp()
    assert isinstance(ts, str)
    assert len(ts) == 14
    
    # Filename example
    filename = f"experiment_{gen_timestamp()}.csv"
    assert filename.startswith("experiment_")
    assert filename.endswith(".csv")
    assert re.match(r'^experiment_\d{4}-\d{4}-\d{4}\.csv$', filename)


def test_gen_timestamp_vs_gen_id():
    """Test relationship between gen_timestamp and gen_id."""
    # Import gen_id to compare
    from mngs.reproduce._gen_ID import gen_id
    
    # gen_timestamp provides fixed format
    ts = gen_timestamp()
    
    # gen_id with similar format should produce similar timestamp part
    # but with added random string
    id_custom = gen_id(time_format="%Y-%m%d-%H%M", N=0)
    
    # The timestamp parts should match (if called close in time)
    # Remove the trailing underscore from gen_id
    id_ts = id_custom.rstrip('_')
    
    # They might differ by a minute if called at minute boundary
    if ts != id_ts:
        # Check they're within 1 minute
        ts_min = int(ts[-2:])
        id_min = int(id_ts[-2:])
        assert abs(ts_min - id_min) <= 1


if __name__ == "__main__":
    import os
    pytest.main([os.path.abspath(__file__), "-v"])


# EOF