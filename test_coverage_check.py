#!/usr/bin/env python3
"""Simple test to verify mngs module can be imported and used."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test basic imports work."""
    try:
        import mngs
        print("✓ mngs imported successfully")
        
        # Test gen module
        from mngs.gen import to_even, to_odd
        assert to_even(5) == 4
        assert to_odd(4) == 5
        print("✓ mngs.gen functions work")
        
        # Test str module  
        from mngs.str import squeeze_space
        # Note: function might be named differently
        print("✓ mngs.str imported")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_gen_functions():
    """Test various gen module functions."""
    try:
        from mngs.gen import to_even, to_odd
        
        # Test to_even
        test_cases_even = [
            (5, 4),
            (6, 6), 
            (0, 0),
            (-3, -4),
            (3.7, 2),
        ]
        
        for input_val, expected in test_cases_even:
            result = to_even(input_val)
            assert result == expected, f"to_even({input_val}) = {result}, expected {expected}"
        
        print("✓ to_even tests passed")
        
        # Test to_odd
        test_cases_odd = [
            (4, 5),
            (5, 5),
            (0, 1),
            (-2, -1),
            (2.3, 3),
        ]
        
        for input_val, expected in test_cases_odd:
            result = to_odd(input_val)
            assert result == expected, f"to_odd({input_val}) = {result}, expected {expected}"
            
        print("✓ to_odd tests passed")
        
        return True
    except Exception as e:
        print(f"✗ Error in gen functions: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_type_checking():
    """Test type checking utilities."""
    try:
        from mngs.gen import type as get_type
        
        # Test various types
        assert "int" in str(get_type(5))
        assert "str" in str(get_type("hello"))
        assert "list" in str(get_type([1, 2, 3]))
        
        print("✓ Type checking works")
        return True
    except Exception as e:
        print(f"✗ Error in type checking: {e}")
        return False

def check_module_structure():
    """Check the mngs module structure."""
    try:
        import mngs
        
        # List main modules
        main_modules = ['gen', 'io', 'plt', 'str', 'pd', 'np']
        available = []
        missing = []
        
        for module in main_modules:
            try:
                mod = getattr(mngs, module, None)
                if mod:
                    available.append(module)
                else:
                    missing.append(module)
            except:
                missing.append(module)
        
        print(f"\nAvailable modules: {', '.join(available)}")
        if missing:
            print(f"Missing modules: {', '.join(missing)}")
            
        # Check gen submodules
        if hasattr(mngs, 'gen'):
            gen_funcs = [name for name in dir(mngs.gen) if not name.startswith('_')]
            print(f"\nmngs.gen functions ({len(gen_funcs)}): {', '.join(gen_funcs[:10])}...")
            
        return True
    except Exception as e:
        print(f"✗ Error checking structure: {e}")
        return False

def main():
    """Run all tests."""
    print("=== MNGS Module Test ===\n")
    
    all_passed = True
    
    # Run tests
    all_passed &= test_basic_imports()
    all_passed &= check_module_structure()
    all_passed &= test_gen_functions()
    all_passed &= test_type_checking()
    
    print("\n" + "="*50)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
        
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)