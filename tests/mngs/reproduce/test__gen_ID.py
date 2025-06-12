#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-06-02 15:40:00 (ywatanabe)"
# File: ./mngs_repo/tests/mngs/reproduce/test__gen_ID.py

"""Tests for gen_id/gen_ID function."""

import re
import time
from datetime import datetime
from unittest.mock import patch

import pytest

from mngs.reproduce._gen_ID import gen_ID, gen_id


def test_gen_id_basic():
    """Test basic ID generation."""
    id1 = gen_id()
    
    # Check format: timestamp_randomchars
    assert '_' in id1
    parts = id1.split('_')
    assert len(parts) == 2
    
    # Check default random part length (8 chars)
    assert len(parts[1]) == 8
    
    # Check random part contains only alphanumeric
    assert parts[1].isalnum()


def test_gen_id_uniqueness():
    """Test that generated IDs are unique."""
    ids = [gen_id() for _ in range(100)]
    
    # All IDs should be unique
    assert len(ids) == len(set(ids))


def test_gen_id_custom_time_format():
    """Test ID generation with custom time format."""
    # Simple date format
    id1 = gen_id(time_format="%Y%m%d")
    parts = id1.split('_')
    assert len(parts[0]) == 8  # YYYYMMDD
    assert parts[0].isdigit()
    
    # Hour-minute format
    id2 = gen_id(time_format="%H%M")
    parts = id2.split('_')
    assert len(parts[0]) == 4  # HHMM
    assert parts[0].isdigit()
    
    # Custom format with separators
    id3 = gen_id(time_format="%Y-%m-%d_%H-%M-%S")
    parts = id3.split('_')
    # Should have timestamp_randomchars
    assert len(parts) == 3  # Due to extra _ in time format
    
    # Check timestamp pattern
    timestamp = f"{parts[0]}_{parts[1]}"
    assert re.match(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', timestamp)


def test_gen_id_custom_length():
    """Test ID generation with custom random string length."""
    # Short random string
    id1 = gen_id(N=4)
    parts = id1.split('_')
    assert len(parts[1]) == 4
    
    # Long random string
    id2 = gen_id(N=16)
    parts = id2.split('_')
    assert len(parts[1]) == 16
    
    # Zero length random string
    id3 = gen_id(N=0)
    parts = id3.split('_')
    assert parts[1] == ''


def test_gen_id_default_format():
    """Test the default time format."""
    # Mock datetime to control the timestamp
    mock_dt = datetime(2025, 5, 31, 12, 30, 45)
    with patch('mngs.reproduce._gen_ID._datetime') as mock_datetime:
        mock_datetime.now.return_value = mock_dt
        
        id1 = gen_id()
        parts = id1.split('_')
        
        # Default format: "%YY-%mM-%dD-%Hh%Mm%Ss"
        expected = "2025Y-05M-31D-12h30m45s"
        assert parts[0] == expected


def test_gen_id_time_progression():
    """Test that timestamps progress correctly."""
    id1 = gen_id(time_format="%Y%m%d%H%M%S")
    time.sleep(0.01)  # Small delay
    id2 = gen_id(time_format="%Y%m%d%H%M%S")
    
    # Extract timestamps
    ts1 = id1.split('_')[0]
    ts2 = id2.split('_')[0]
    
    # Second timestamp should be >= first
    assert ts2 >= ts1


def test_gen_id_character_set():
    """Test that random part uses correct character set."""
    import string
    
    # Generate many IDs to test character set
    all_chars = set()
    for _ in range(100):
        id1 = gen_id(N=10)
        random_part = id1.split('_')[1]
        all_chars.update(random_part)
    
    # Should only contain letters and digits
    valid_chars = set(string.ascii_letters + string.digits)
    assert all_chars.issubset(valid_chars)
    
    # Should eventually use both letters and digits (probabilistic)
    assert any(c.isalpha() for c in all_chars)
    assert any(c.isdigit() for c in all_chars)


def test_gen_id_edge_cases():
    """Test edge cases for ID generation."""
    # Very long random string
    id1 = gen_id(N=100)
    parts = id1.split('_')
    assert len(parts[1]) == 100
    assert parts[1].isalnum()
    
    # Empty time format (should still work)
    id2 = gen_id(time_format="", N=5)
    assert id2 == f"_{id2.split('_')[1]}"
    assert len(id2.split('_')[1]) == 5
    
    # Complex time format
    id3 = gen_id(time_format="Year%Y_Month%m_Day%d")
    parts = id3.split('_')
    # Will have multiple parts due to underscores in format
    assert parts[0].startswith("Year")
    assert "Month" in parts[1]
    assert "Day" in parts[2]


def test_gen_ID_backward_compatibility():
    """Test that gen_ID (deprecated) still works."""
    # Both should work the same way
    id_new = gen_id(N=5)
    id_old = gen_ID(N=5)
    
    # Same structure
    assert '_' in id_old
    parts = id_old.split('_')
    assert len(parts) == 2
    assert len(parts[1]) == 5
    
    # Test they're actually the same function
    assert gen_ID is gen_id


def test_gen_id_reproducibility_with_seed():
    """Test ID generation with fixed random seed."""
    import random
    
    # Set seed and generate ID
    random.seed(42)
    id1 = gen_id(time_format="%Y%m%d", N=10)
    
    # Reset seed and generate again
    random.seed(42)
    id2 = gen_id(time_format="%Y%m%d", N=10)
    
    # Random parts should be the same
    assert id1.split('_')[1] == id2.split('_')[1]


def test_gen_id_concurrent_generation():
    """Test concurrent ID generation maintains uniqueness."""
    import threading
    
    ids = []
    lock = threading.Lock()
    
    def generate_id():
        id_val = gen_id()
        with lock:
            ids.append(id_val)
    
    # Create multiple threads
    threads = []
    for _ in range(10):
        t = threading.Thread(target=generate_id)
        threads.append(t)
        t.start()
    
    # Wait for all threads
    for t in threads:
        t.join()
    
    # All IDs should be unique
    assert len(ids) == len(set(ids))


def test_gen_id_special_formats():
    """Test with various datetime format strings."""
    # Test various format strings
    formats = [
        "%Y",  # Year only
        "%Y%m%d",  # Date only
        "%H%M%S",  # Time only
        "%Y-%m-%d %H:%M:%S",  # Full datetime
        "%B %d, %Y",  # Month name
        "%A",  # Day name
        "%j",  # Day of year
        "%U",  # Week number
    ]
    
    for fmt in formats:
        id1 = gen_id(time_format=fmt, N=4)
        assert '_' in id1
        parts = id1.split('_')
        # Last part should always be the random string
        assert len(parts[-1]) == 4
        assert parts[-1].isalnum()


def test_gen_id_performance():
    """Test that ID generation is reasonably fast."""
    import time
    
    start = time.time()
    ids = [gen_id() for _ in range(1000)]
    duration = time.time() - start
    
    # Should generate 1000 IDs in less than 1 second
    assert duration < 1.0
    
    # All should be unique
    assert len(ids) == len(set(ids))


def test_gen_id_example_usage():
    """Test example usage patterns from docstring."""
    # Basic usage
    exp_id = gen_id()
    save_path = f"results/experiment_{exp_id}.pkl"
    assert save_path.startswith("results/experiment_")
    assert save_path.endswith(".pkl")
    
    # Custom format for daily logs
    daily_id = gen_id(time_format="%Y%m%d", N=4)
    log_file = f"logs/{daily_id}.log"
    assert re.match(r'logs/\d{8}_\w{4}\.log', log_file)
    
    # Session tracking
    session_id = gen_id(time_format="%H%M%S", N=6)
    assert re.match(r'\d{6}_\w{6}', session_id)


if __name__ == "__main__":
    import os
    pytest.main([os.path.abspath(__file__), "-v"])


<<<<<<< HEAD
# EOF
=======
    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/reproduce/_gen_ID.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-07 17:53:38 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/reproduce/_gen_ID.py
# 
# import random as _random
# import string as _string
# from datetime import datetime as _datetime
# 
# 
# def gen_ID(time_format="%YY-%mM-%dD-%Hh%Mm%Ss", N=8):
#     now_str = _datetime.now().strftime(time_format)
#     rand_str = "".join(
#         [_random.choice(_string.ascii_letters + _string.digits) for i in range(N)]
#     )
#     return now_str + "_" + rand_str
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/reproduce/_gen_ID.py
# --------------------------------------------------------------------------------
>>>>>>> origin/main
