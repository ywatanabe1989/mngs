import numpy as np
import pytest
from mngs.gen.misc import (color_text, connect_nums, connect_strs,
                           decapitalize, float_linspace, print_block, replace,
                           search, squeeze_spaces, to_even, to_odd)


def test_decapitalize():
    assert decapitalize("Hello") == "hello"
    assert decapitalize("WORLD") == "wORLD"
    assert decapitalize("") == ""
    assert decapitalize("a") == "a"


def test_connect_strs():
    assert connect_strs(["a", "b", "c"]) == "a_b_c"
    assert connect_strs(["hello", "world"], filler="-") == "hello-world"
    assert connect_strs(["single"]) == "single"
    assert connect_strs([]) == ""


def test_connect_nums():
    assert connect_nums([1, 2, 3]) == "1_2_3"
    assert connect_nums([3.14, 2.718, 1.414], filler="-") == "3.14-2.718-1.414"
    assert connect_nums([42]) == "42"
    assert connect_nums([]) == ""


def test_squeeze_spaces():
    assert squeeze_spaces("Hello   world") == "Hello world"
    assert squeeze_spaces("a---b--c-d", pattern="-+", repl="-") == "a-b-c-d"
    assert (
        squeeze_spaces("   leading and trailing   ")
        == " leading and trailing "
    )


def test_search():
    strings = ["apple", "orange", "banana", "grape"]
    patterns = ["an"]

    indices, matched = search(patterns, strings)
    assert indices == [1, 2]
    assert matched == ["orange", "banana"]

    bool_array, matched = search(patterns, strings, as_bool=True)
    assert np.array_equal(bool_array, [False, True, True, False])
    assert matched == ["orange", "banana"]


def test_to_even():
    assert to_even(5) == 4
    assert to_even(6) == 6
    assert to_even(0) == 0
    assert to_even(-3) == -4


def test_to_odd():
    assert to_odd(6) == 5
    assert to_odd(7) == 7
    assert to_odd(0) == -1
    assert to_odd(-4) == -5


def test_float_linspace():
    result = float_linspace(0, 1, 5)
    expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    assert np.allclose(result, expected)

    result = float_linspace(1, 5, 3)
    expected = np.array([1.0, 3.0, 5.0])
    assert np.allclose(result, expected)


def test_replace():
    template = "Hello, {name}! You are {age} years old."
    replacements = {"name": "Alice", "age": "30"}
    assert (
        replace(template, replacements)
        == "Hello, Alice! You are 30 years old."
    )

    assert (
        replace("Original string", "Replacement string")
        == "Replacement string"
    )
    assert replace("No replacements", {}) == "No replacements"


# Note: print_block and color_text are not easily testable as they involve
# printing to the console. You might want to consider refactoring these
# functions to return strings instead of printing them directly for easier testing.
