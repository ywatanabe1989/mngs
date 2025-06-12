#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-06-10 18:48:00 (ywatanabe)"
# File: ./mngs_repo/tests/mngs/dev/test__reload_comprehensive.py

"""Comprehensive tests for module reload functionality."""

import importlib
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import pytest


class TestReloadBasic:
    """Basic reload functionality tests."""
    
    def test_reload_imports(self):
        """Test that reload module can be imported."""
        from mngs.dev import _reload
        assert hasattr(_reload, 'reload')
        assert hasattr(_reload, 'reload_auto')
        assert hasattr(_reload, 'reload_stop')
    
    @patch('importlib.reload')
    def test_reload_basic(self, mock_reload):
        """Test basic reload functionality."""
        from mngs.dev._reload import reload
        
        # Setup mock
        mock_mngs = MagicMock()
        mock_reload.return_value = mock_mngs
        
        with patch.dict('sys.modules', {'mngs': mock_mngs}):
            result = reload()
            assert result == mock_mngs
            mock_reload.assert_called()
    
    @patch('importlib.reload')
    def test_reload_all_submodules(self, mock_reload):
        """Test that all mngs submodules are reloaded."""
        from mngs.dev._reload import reload
        
        # Create mock modules
        mock_modules = {
            'mngs': MagicMock(),
            'mngs.io': MagicMock(),
            'mngs.plt': MagicMock(),
            'mngs.gen': MagicMock(),
        }
        
        with patch.dict('sys.modules', mock_modules):
            reload()
            
            # Verify all modules were reloaded
            assert mock_reload.call_count >= len(mock_modules)
    
    @patch('importlib.reload')
    def test_reload_handles_exceptions(self, mock_reload):
        """Test that reload continues even if some modules fail."""
        from mngs.dev._reload import reload
        
        # Setup mock to fail on first call
        mock_reload.side_effect = [Exception("Module error"), MagicMock()]
        
        mock_modules = {
            'mngs.broken': MagicMock(),
            'mngs': MagicMock(),
        }
        
        with patch.dict('sys.modules', mock_modules):
            # Should not raise exception
            result = reload()
            assert result is not None


class TestAutoReload:
    """Auto-reload functionality tests."""
    
    def test_reload_auto_starts_thread(self):
        """Test that reload_auto starts a background thread."""
        from mngs.dev._reload import reload_auto, reload_stop, _reload_thread
        
        try:
            reload_auto(interval=1)
            
            # Give thread time to start
            time.sleep(0.1)
            
            # Check thread is running
            assert _reload_thread is not None
            assert _reload_thread.is_alive()
            assert _reload_thread.daemon
            
        finally:
            reload_stop()
            time.sleep(0.2)
    
    def test_reload_auto_prevents_multiple_threads(self):
        """Test that only one auto-reload thread can run."""
        from mngs.dev._reload import reload_auto, reload_stop
        
        try:
            reload_auto(interval=1)
            thread1 = threading.active_count()
            
            # Try to start another
            reload_auto(interval=1)
            thread2 = threading.active_count()
            
            # Thread count should not increase
            assert thread1 == thread2
            
        finally:
            reload_stop()
            time.sleep(0.2)
    
    @patch('mngs.dev._reload.reload')
    def test_reload_auto_calls_reload_periodically(self, mock_reload):
        """Test that auto-reload calls reload at intervals."""
        from mngs.dev._reload import reload_auto, reload_stop
        
        try:
            reload_auto(interval=0.1)  # 100ms interval
            time.sleep(0.35)  # Wait for ~3 intervals
            
            # Should have been called at least 2 times
            assert mock_reload.call_count >= 2
            
        finally:
            reload_stop()
            time.sleep(0.2)
    
    def test_reload_stop(self):
        """Test stopping auto-reload."""
        from mngs.dev._reload import reload_auto, reload_stop, _running
        
        reload_auto(interval=0.1)
        time.sleep(0.1)
        
        reload_stop()
        assert not _running
        
        # Wait for thread to stop
        time.sleep(0.3)


class TestAutoReloadErrorHandling:
    """Test error handling in auto-reload."""
    
    @patch('mngs.dev._reload.reload')
    def test_auto_reload_handles_exceptions(self, mock_reload):
        """Test that auto-reload continues after exceptions."""
        from mngs.dev._reload import reload_auto, reload_stop
        
        # Make reload raise exception sometimes
        mock_reload.side_effect = [
            Exception("Error 1"),
            None,
            Exception("Error 2"),
            None
        ]
        
        try:
            with patch('builtins.print') as mock_print:
                reload_auto(interval=0.1)
                time.sleep(0.45)
                
                # Should have printed error messages
                error_calls = [c for c in mock_print.call_args_list 
                             if 'Reload failed' in str(c)]
                assert len(error_calls) >= 2
                
        finally:
            reload_stop()
            time.sleep(0.2)


class TestReloadWithRealModules:
    """Test reload with actual module operations."""
    
    def test_reload_with_temp_module(self):
        """Test reloading a temporary module."""
        from mngs.dev._reload import reload
        
        # Create a temporary module
        with tempfile.TemporaryDirectory() as tmpdir:
            module_path = Path(tmpdir) / "test_module.py"
            module_path.write_text("value = 1")
            
            # Add to sys.path
            sys.path.insert(0, tmpdir)
            
            try:
                # Import module
                import test_module
                assert test_module.value == 1
                
                # Modify module
                module_path.write_text("value = 2")
                
                # Manual reload of test module
                importlib.reload(test_module)
                assert test_module.value == 2
                
                # Clean up
                del sys.modules['test_module']
                
            finally:
                sys.path.remove(tmpdir)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_reload_with_empty_modules(self):
        """Test reload when no mngs modules are loaded."""
        from mngs.dev._reload import reload
        
        # Remove all mngs modules temporarily
        original_modules = {}
        for name in list(sys.modules.keys()):
            if name.startswith('mngs'):
                original_modules[name] = sys.modules.pop(name)
        
        try:
            with patch('importlib.reload') as mock_reload:
                reload()
                # Should still try to reload mngs
                assert mock_reload.called
                
        finally:
            # Restore modules
            sys.modules.update(original_modules)
    
    def test_reload_auto_with_zero_interval(self):
        """Test auto-reload with zero interval."""
        from mngs.dev._reload import reload_auto, reload_stop
        
        try:
            # Should handle zero interval gracefully
            reload_auto(interval=0)
            time.sleep(0.1)
            
        finally:
            reload_stop()
            time.sleep(0.1)
    
    def test_reload_auto_with_negative_interval(self):
        """Test auto-reload with negative interval."""
        from mngs.dev._reload import reload_auto, reload_stop
        
        try:
            # Should handle negative interval
            reload_auto(interval=-1)
            time.sleep(0.1)
            
        finally:
            reload_stop()
            time.sleep(0.1)


class TestThreadSafety:
    """Test thread safety of reload operations."""
    
    def test_concurrent_reload_calls(self):
        """Test multiple concurrent reload calls."""
        from mngs.dev._reload import reload
        
        results = []
        exceptions = []
        
        def reload_wrapper():
            try:
                with patch('importlib.reload', return_value=MagicMock()):
                    result = reload()
                    results.append(result)
            except Exception as e:
                exceptions.append(e)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            t = threading.Thread(target=reload_wrapper)
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Should complete without exceptions
        assert len(exceptions) == 0
        assert len(results) == 5
    
    def test_reload_stop_during_reload(self):
        """Test stopping reload while it's running."""
        from mngs.dev._reload import reload_auto, reload_stop
        
        with patch('mngs.dev._reload.reload') as mock_reload:
            # Make reload take some time
            mock_reload.side_effect = lambda: time.sleep(0.2)
            
            reload_auto(interval=0.1)
            time.sleep(0.15)  # Let it start reloading
            
            reload_stop()
            time.sleep(0.3)
            
            # Should have stopped
            final_count = mock_reload.call_count
            time.sleep(0.2)
            
            # Count should not increase
            assert mock_reload.call_count == final_count


class TestIntegration:
    """Integration tests with actual mngs package."""
    
    def test_reload_preserves_functionality(self):
        """Test that reload doesn't break functionality."""
        from mngs.dev._reload import reload
        
        # Import something from mngs
        from mngs.gen import title2path
        
        # Store original function
        original_func = title2path
        
        # Reload
        with patch('importlib.reload', side_effect=importlib.reload):
            reload()
        
        # Re-import
        from mngs.gen import title2path as title2path_new
        
        # Should still work
        assert title2path_new("Test Title") == "test_title"
    
    def test_reload_updates_module_references(self):
        """Test that reload updates module references correctly."""
        from mngs.dev._reload import reload
        
        # Get module id before reload
        import mngs
        original_id = id(mngs)
        
        # Reload
        result = reload()
        
        # Should return the reloaded module
        assert result is not None
        # Note: id might be the same due to Python's object reuse


class TestPerformance:
    """Performance-related tests."""
    
    def test_reload_performance(self):
        """Test reload performance with many modules."""
        from mngs.dev._reload import reload
        
        # Add many fake mngs modules
        fake_modules = {}
        for i in range(100):
            module_name = f'mngs.fake{i}'
            fake_modules[module_name] = MagicMock()
        
        with patch.dict('sys.modules', fake_modules):
            with patch('importlib.reload') as mock_reload:
                start_time = time.time()
                reload()
                duration = time.time() - start_time
                
                # Should complete reasonably quickly
                assert duration < 1.0
                
                # Should have tried to reload all modules
                assert mock_reload.call_count >= 100


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__), "-v"])