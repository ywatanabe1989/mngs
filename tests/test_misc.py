import pytest
import numpy as np
from mngs.gen import misc

def test_to_even():
    assert misc.to_even(5) == 4
    assert misc.to_even(6) == 6
    assert misc.to_even(3.7) == 2

def test_to_odd():
    assert misc.to_odd(6) == 5
    assert misc.to_odd(7) == 7
    assert misc.to_odd(5.8) == 5

def test_float_linspace():
    result = misc.float_linspace(0, 1, 5)
    expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_array_almost_equal(result, expected)

def test_replace():
    assert misc.replace("Hello, {name}!", {"name": "World"}) == "Hello, World!"
    assert misc.replace("Original string", "New string") == "New string"
    assert misc.replace("Value: {x}", {"x": "42"}) == "Value: 42"
