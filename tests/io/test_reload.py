import pytest
from mngs.io._reload import reload
import sys

def test_reload():
    # This is a basic test as reload is a built-in function
    # We'll just check if it doesn't raise any exceptions
    try:
        reload(sys)
    except Exception as e:
        pytest.fail(f'reload raised {e} unexpectedly')

def test_reload_non_existent_module():
    with pytest.raises(ImportError):
        reload('non_existent_module')
