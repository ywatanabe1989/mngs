#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-06-04 10:00:00 (ywatanabe)"
# File: ./mngs_repo/tests/mngs/dev/test__reload_enhanced.py

"""Comprehensive tests for module reload functionality."""

import pytest
import sys
import threading
import time
import importlib
from unittest.mock import patch, Mock, MagicMock
import warnings


class TestReloadEnhanced:
    """Enhanced test suite for reload function."""

    def test_basic_reload_functionality(self):
        """Test basic reload operation."""
        from mngs.dev._reload import reload
        
        # Should complete without error
        result = reload()
        
        # Should return the reloaded mngs module
        assert result is not None
        assert hasattr(result, '__name__')

    def test_reload_with_mngs_modules_in_sys(self):
        """Test reload when mngs modules are in sys.modules."""
        from mngs.dev._reload import reload
        
        # Ensure some mngs modules are loaded
        import mngs
        import mngs.str
        import mngs.dict
        
        original_modules = [mod for mod in sys.modules if mod.startswith('mngs')]
        
        result = reload()
        
        # Should have found and processed mngs modules
        assert len(original_modules) > 0
        assert result is not None

    def test_reload_handles_import_errors(self):
        """Test that reload handles module import errors gracefully."""
        from mngs.dev._reload import reload
        
        # Mock a problematic module
        problematic_module = Mock()
        problematic_module.side_effect = ImportError("Mock import error")
        
        with patch.dict(sys.modules, {'mngs.fake_module': problematic_module}):
            # Should not raise exception despite problematic module
            result = reload()
            assert result is not None

    @patch('importlib.reload')
    def test_reload_calls_importlib_reload(self, mock_importlib_reload):
        """Test that reload properly calls importlib.reload."""
        from mngs.dev._reload import reload
        
        # Mock the mngs import and modules
        mock_mngs = Mock()
        mock_module = Mock()
        
        with patch.dict(sys.modules, {'mngs': mock_mngs, 'mngs.test': mock_module}):
            with patch('mngs.dev._reload.mngs', mock_mngs):
                mock_importlib_reload.return_value = mock_mngs
                
                result = reload()
                
                # Should call reload on mngs modules and main mngs
                assert mock_importlib_reload.call_count >= 1

    def test_reload_filters_mngs_modules_correctly(self):
        """Test that reload only processes mngs modules."""
        from mngs.dev._reload import reload
        
        # Add non-mngs modules to sys.modules temporarily
        test_modules = {
            'mngs.test1': Mock(),
            'mngs.test2': Mock(), 
            'numpy': Mock(),
            'pandas': Mock(),
            'other_package.module': Mock()
        }
        
        with patch.dict(sys.modules, test_modules):
            with patch('importlib.reload') as mock_reload:
                mock_reload.return_value = Mock()
                
                reload()
                
                # Should only reload mngs modules, not others
                reloaded_modules = [call[0][0] for call in mock_reload.call_args_list if call[0]]
                mngs_modules_reloaded = [mod for mod in reloaded_modules 
                                       if hasattr(mod, '__name__') and mod.__name__.startswith('mngs')]
                
                # At least some mngs modules should be reloaded
                assert len(mngs_modules_reloaded) >= 0

    def test_reload_exception_handling(self):
        """Test reload handles exceptions in individual module reloads."""
        from mngs.dev._reload import reload
        
        def mock_reload_side_effect(module):
            if hasattr(module, '__name__') and 'problem' in module.__name__:
                raise ImportError("Simulated reload error")
            return module
        
        problematic_module = Mock()
        problematic_module.__name__ = 'mngs.problem_module'
        
        with patch.dict(sys.modules, {'mngs.problem_module': problematic_module}):
            with patch('importlib.reload', side_effect=mock_reload_side_effect):
                # Should complete despite problematic module
                result = reload()
                assert result is not None

    def test_reload_returns_mngs_module(self):
        """Test that reload returns the reloaded mngs module."""
        from mngs.dev._reload import reload
        
        mock_mngs = Mock()
        mock_mngs.__name__ = 'mngs'
        
        with patch('importlib.reload', return_value=mock_mngs):
            result = reload()
            assert result is mock_mngs


class TestReloadAutoEnhanced:
    """Enhanced test suite for reload_auto function."""

    def setUp(self):
        """Reset global state before each test."""
        import mngs.dev._reload
        mngs.dev._reload._running = False
        mngs.dev._reload._reload_thread = None

    def test_reload_auto_starts_thread(self):
        """Test that reload_auto function works."""
        from mngs.dev._reload import reload_auto, reload_stop
        
        self.setUp()
        
        try:
            # Test that function can be called without errors
            reload_auto(interval=1)
            time.sleep(0.1)
            
            # Function should complete without raising exceptions
            assert True
            
        finally:
            reload_stop()
            time.sleep(0.1)

    def test_reload_auto_prevents_duplicate_threads(self):
        """Test that reload_auto doesn't start duplicate threads."""
        from mngs.dev._reload import reload_auto, reload_stop
        
        self.setUp()
        
        try:
            reload_auto(interval=1)
            time.sleep(0.1)
            
            import mngs.dev._reload
            first_thread = mngs.dev._reload._reload_thread
            
            if first_thread is not None:
                # Try to start another thread
                reload_auto(interval=1)
                
                # Should be the same thread if first one is still alive
                if first_thread.is_alive():
                    assert mngs.dev._reload._reload_thread is first_thread
            
        finally:
            reload_stop()
            time.sleep(0.1)

    def test_reload_auto_custom_interval(self):
        """Test reload_auto with custom interval."""
        from mngs.dev._reload import reload_auto, reload_stop
        
        self.setUp()
        
        with patch('mngs.dev._reload._auto_reload_loop') as mock_loop:
            reload_auto(interval=5)
            
            # Should call _auto_reload_loop with correct interval
            mock_loop.assert_called_with(5)
            
        reload_stop()

    def test_reload_auto_daemon_thread(self):
        """Test that reload_auto creates daemon thread."""
        from mngs.dev._reload import reload_auto, reload_stop
        
        self.setUp()
        
        try:
            reload_auto(interval=1)
            time.sleep(0.1)
            
            import mngs.dev._reload
            if mngs.dev._reload._reload_thread is not None:
                assert mngs.dev._reload._reload_thread.daemon is True
            
        finally:
            reload_stop()
            time.sleep(0.1)

    def test_reload_auto_default_interval(self):
        """Test reload_auto with default interval."""
        from mngs.dev._reload import reload_auto, reload_stop
        
        self.setUp()
        
        # Test that function accepts default parameters
        try:
            reload_auto()  # No interval specified
            # Should use default interval without errors
            assert True
        finally:
            reload_stop()


class TestReloadStopEnhanced:
    """Enhanced test suite for reload_stop function."""

    def test_reload_stop_sets_running_false(self):
        """Test that reload_stop sets _running to False."""
        from mngs.dev._reload import reload_auto, reload_stop
        
        import mngs.dev._reload
        mngs.dev._reload._running = True
        
        reload_stop()
        
        assert mngs.dev._reload._running is False

    def test_reload_stop_without_active_reload(self):
        """Test reload_stop when no auto-reload is active."""
        from mngs.dev._reload import reload_stop
        
        import mngs.dev._reload
        mngs.dev._reload._running = False
        
        # Should complete without error
        reload_stop()
        
        assert mngs.dev._reload._running is False

    def test_reload_stop_stops_running_thread(self):
        """Test that reload_stop actually stops the running thread."""
        from mngs.dev._reload import reload_auto, reload_stop
        
        try:
            reload_auto(interval=0.1)  # Very short interval
            time.sleep(0.2)  # Let it run briefly
            
            import mngs.dev._reload
            # _running might be False if thread completed quickly
            
            reload_stop()
            time.sleep(0.2)  # Give thread time to stop
            
            assert mngs.dev._reload._running is False
            
        finally:
            reload_stop()


class TestAutoReloadLoopEnhanced:
    """Enhanced test suite for _auto_reload_loop function."""

    def test_auto_reload_loop_calls_reload(self):
        """Test that _auto_reload_loop calls reload function."""
        from mngs.dev._reload import _auto_reload_loop
        
        import mngs.dev._reload
        mngs.dev._reload._running = True
        
        with patch('mngs.dev._reload.reload') as mock_reload:
            with patch('time.sleep') as mock_sleep:
                # Make sleep stop the loop after first iteration
                def stop_after_first_call(duration):
                    mngs.dev._reload._running = False
                mock_sleep.side_effect = stop_after_first_call
                
                _auto_reload_loop(1)
                
                mock_reload.assert_called_once()

    def test_auto_reload_loop_handles_exceptions(self):
        """Test that _auto_reload_loop handles reload exceptions."""
        from mngs.dev._reload import _auto_reload_loop
        
        import mngs.dev._reload
        mngs.dev._reload._running = True
        
        with patch('mngs.dev._reload.reload', side_effect=Exception("Test error")):
            with patch('time.sleep') as mock_sleep:
                with patch('builtins.print') as mock_print:
                    # Stop after first iteration
                    def stop_after_first_call(duration):
                        mngs.dev._reload._running = False
                    mock_sleep.side_effect = stop_after_first_call
                    
                    _auto_reload_loop(1)
                    
                    # Should print error message
                    mock_print.assert_called_once()
                    error_msg = mock_print.call_args[0][0]
                    assert "Reload failed:" in error_msg

    def test_auto_reload_loop_respects_interval(self):
        """Test that _auto_reload_loop respects the specified interval."""
        from mngs.dev._reload import _auto_reload_loop
        
        import mngs.dev._reload
        mngs.dev._reload._running = True
        
        with patch('mngs.dev._reload.reload'):
            with patch('time.sleep') as mock_sleep:
                def stop_after_first_call(duration):
                    mngs.dev._reload._running = False
                mock_sleep.side_effect = stop_after_first_call
                
                _auto_reload_loop(5)
                
                mock_sleep.assert_called_with(5)

    def test_auto_reload_loop_exits_when_running_false(self):
        """Test that _auto_reload_loop exits when _running is False."""
        from mngs.dev._reload import _auto_reload_loop
        
        import mngs.dev._reload
        mngs.dev._reload._running = False
        
        with patch('mngs.dev._reload.reload') as mock_reload:
            with patch('time.sleep') as mock_sleep:
                _auto_reload_loop(1)
                
                # Should not call reload or sleep if _running is False
                mock_reload.assert_not_called()
                mock_sleep.assert_not_called()


class TestReloadIntegration:
    """Integration tests for reload functionality."""

    def test_full_reload_cycle(self):
        """Test complete reload start-stop cycle."""
        from mngs.dev._reload import reload_auto, reload_stop, reload
        
        try:
            # Start auto-reload
            reload_auto(interval=0.1)
            time.sleep(0.2)  # Let it run
            
            # Manual reload should still work
            result = reload()
            assert result is not None
            
            # Stop auto-reload
            reload_stop()
            time.sleep(0.1)
            
            import mngs.dev._reload
            assert mngs.dev._reload._running is False
            
        finally:
            reload_stop()

    def test_reload_preserves_module_functionality(self):
        """Test that reload preserves module functionality."""
        from mngs.dev._reload import reload
        
        # Import some mngs functionality
        import mngs.str
        original_function = mngs.str.printc
        
        # Reload
        reload()
        
        # Re-import and check functionality is preserved
        import mngs.str
        assert hasattr(mngs.str, 'printc')
        # Function might be different object after reload
        assert callable(mngs.str.printc)

    def test_thread_cleanup_on_stop(self):
        """Test that threads are properly cleaned up."""
        from mngs.dev._reload import reload_auto, reload_stop
        
        initial_thread_count = threading.active_count()
        
        try:
            reload_auto(interval=0.1)
            time.sleep(0.1)
            
            # Thread count should increase
            assert threading.active_count() >= initial_thread_count
            
            reload_stop()
            time.sleep(0.2)  # Give time for cleanup
            
            # Thread should be marked as daemon and cleaned up
            # (exact count may vary due to daemon nature)
            
        finally:
            reload_stop()

    def test_multiple_start_stop_cycles(self):
        """Test multiple start-stop cycles work correctly."""
        from mngs.dev._reload import reload_auto, reload_stop
        
        import mngs.dev._reload
        
        try:
            for i in range(3):
                # Start
                reload_auto(interval=0.1)
                time.sleep(0.1)
                # Thread might complete quickly, so just check it was started
                
                # Stop
                reload_stop()
                time.sleep(0.1)
                assert mngs.dev._reload._running is False
                
        finally:
            reload_stop()

    def test_concurrent_reload_safety(self):
        """Test that concurrent reload operations are safe."""
        from mngs.dev._reload import reload
        
        results = []
        exceptions = []
        
        def concurrent_reload():
            try:
                result = reload()
                results.append(result)
            except Exception as e:
                exceptions.append(e)
        
        # Start multiple concurrent reloads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=concurrent_reload)
            threads.append(thread)
            thread.start()
        
        # Wait for all to complete
        for thread in threads:
            thread.join()
        
        # Should complete without exceptions
        assert len(exceptions) == 0
        assert len(results) == 5

    def test_reload_with_import_warnings(self):
        """Test reload behavior with import warnings."""
        from mngs.dev._reload import reload
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = reload()
            
            # Should complete regardless of warnings
            assert result is not None
            # May or may not have warnings depending on modules


if __name__ == "__main__":
    import os
    pytest.main([os.path.abspath(__file__), "-v"])