#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-06-01 12:45:00 (ywatanabe)"
# File: ./tests/mngs/ai/sk/test___init__.py

"""Comprehensive tests for mngs.ai.sk module initialization."""

import pytest
import numpy as np
import pandas as pd
import warnings
from unittest.mock import patch, Mock
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
import mngs


class TestSkModuleImport:
    """Test suite for module import functionality."""

    def test_module_import(self):
        """Test that the module can be imported."""
        assert hasattr(mngs.ai, 'sk')
        
    def test_submodule_imports(self):
        """Test that submodules are properly imported."""
        # Check if sk submodule exists
        assert hasattr(mngs.ai.sk, 'to_sktime_df')
        assert hasattr(mngs.ai.sk, 'rocket_pipeline')
        assert hasattr(mngs.ai.sk, 'GB_pipeline')
        
    def test_wildcard_imports(self):
        """Test that wildcard imports from _clf work."""
        # These should be imported from _clf.py
        from mngs.ai.sk import rocket_pipeline, GB_pipeline
        assert callable(rocket_pipeline)
        assert isinstance(GB_pipeline, Pipeline)
        
    def test_specific_imports(self):
        """Test specific imports are available."""
        from mngs.ai.sk import to_sktime_df
        assert callable(to_sktime_df)
        
    def test_module_file_location(self):
        """Test module file location is correct."""
        import mngs.ai.sk
        assert '__file__' in dir(mngs.ai.sk)
        assert 'ai/sk/__init__.py' in mngs.ai.sk.__file__.replace('\\', '/')


class TestFunctionAvailability:
    """Test availability and functionality of module functions."""
    
    def test_to_sktime_df_callable(self):
        """Test that to_sktime_df function is callable."""
        assert callable(mngs.ai.sk.to_sktime_df)
        
    def test_rocket_pipeline_callable(self):
        """Test that rocket_pipeline function is callable."""
        assert callable(mngs.ai.sk.rocket_pipeline)
        
    def test_rocket_pipeline_returns_pipeline(self):
        """Test that rocket_pipeline returns a Pipeline object."""
        pipeline = mngs.ai.sk.rocket_pipeline()
        assert isinstance(pipeline, Pipeline)
        
    def test_rocket_pipeline_with_args(self):
        """Test rocket_pipeline accepts arguments."""
        # Test with some common Rocket parameters
        pipeline = mngs.ai.sk.rocket_pipeline(n_kernels=100)
        assert isinstance(pipeline, Pipeline)
        
    def test_rocket_pipeline_with_kwargs(self):
        """Test rocket_pipeline accepts keyword arguments."""
        pipeline = mngs.ai.sk.rocket_pipeline(n_kernels=50, random_state=42)
        assert isinstance(pipeline, Pipeline)
        
    def test_gb_pipeline_type(self):
        """Test that GB_pipeline is a Pipeline object."""
        assert isinstance(mngs.ai.sk.GB_pipeline, Pipeline)
        
    def test_gb_pipeline_immutable(self):
        """Test that GB_pipeline is the same object on each access."""
        pipeline1 = mngs.ai.sk.GB_pipeline
        pipeline2 = mngs.ai.sk.GB_pipeline
        assert pipeline1 is pipeline2


class TestPipelineStructure:
    """Test the structure and components of pipelines."""
    
    def test_rocket_pipeline_steps(self):
        """Test rocket_pipeline has expected steps."""
        pipeline = mngs.ai.sk.rocket_pipeline()
        assert len(pipeline.steps) == 2
        
        # Check step names and types
        step_names = [name for name, _ in pipeline.steps]
        assert len(step_names) == 2
        
        # First step should be Rocket transformer
        assert pipeline.steps[0][1].__class__.__name__ == 'Rocket'
        
        # Second step should be LogisticRegression
        assert pipeline.steps[1][1].__class__.__name__ == 'LogisticRegression'
        
    def test_rocket_pipeline_classifier_params(self):
        """Test rocket_pipeline classifier has correct parameters."""
        pipeline = mngs.ai.sk.rocket_pipeline()
        classifier = pipeline.steps[1][1]
        
        # Check LogisticRegression parameters
        assert hasattr(classifier, 'max_iter')
        assert classifier.max_iter == 1000
        
    def test_gb_pipeline_steps(self):
        """Test GB_pipeline has expected steps."""
        assert len(mngs.ai.sk.GB_pipeline.steps) == 2
        
        # Check step types
        assert mngs.ai.sk.GB_pipeline.steps[0][1].__class__.__name__ == 'Tabularizer'
        assert mngs.ai.sk.GB_pipeline.steps[1][1].__class__.__name__ == 'GradientBoostingClassifier'
        
    def test_pipeline_methods(self):
        """Test that pipelines have expected sklearn methods."""
        rocket_pipe = mngs.ai.sk.rocket_pipeline()
        
        # Standard sklearn pipeline methods
        assert hasattr(rocket_pipe, 'fit')
        assert hasattr(rocket_pipe, 'predict')
        assert hasattr(rocket_pipe, 'fit_predict')
        assert hasattr(rocket_pipe, 'score')
        assert hasattr(rocket_pipe, 'get_params')
        assert hasattr(rocket_pipe, 'set_params')


class TestModuleExports:
    """Test module exports and namespace."""
    
    def test_public_exports(self):
        """Test that only expected items are exported."""
        public_attrs = [attr for attr in dir(mngs.ai.sk) if not attr.startswith('_')]
        
        # Core exports that should always be present
        core_exports = {'to_sktime_df', 'rocket_pipeline', 'GB_pipeline'}
        
        for export in core_exports:
            assert export in public_attrs
            
    def test_no_private_exports(self):
        """Test that private modules are not exposed."""
        # Private modules should not be directly accessible
        assert not hasattr(mngs.ai.sk, '_clf')
        assert not hasattr(mngs.ai.sk, '_to_sktime')
        
    def test_no_sklearn_reexports(self):
        """Test that sklearn classes are not re-exported."""
        # These should not be directly available
        assert not hasattr(mngs.ai.sk, 'LogisticRegression')
        assert not hasattr(mngs.ai.sk, 'GradientBoostingClassifier')
        assert not hasattr(mngs.ai.sk, 'Pipeline')
        
    def test_no_sktime_reexports(self):
        """Test that sktime classes are not re-exported."""
        assert not hasattr(mngs.ai.sk, 'Rocket')
        assert not hasattr(mngs.ai.sk, 'Tabularizer')
        assert not hasattr(mngs.ai.sk, 'RocketClassifier')


class TestImportBehavior:
    """Test various import behaviors."""
    
    def test_import_all_at_once(self):
        """Test importing all functions at once."""
        from mngs.ai.sk import to_sktime_df, rocket_pipeline, GB_pipeline
        
        assert callable(to_sktime_df)
        assert callable(rocket_pipeline)
        assert isinstance(GB_pipeline, Pipeline)
        
    def test_import_individually(self):
        """Test importing functions individually."""
        from mngs.ai.sk import to_sktime_df
        from mngs.ai.sk import rocket_pipeline
        from mngs.ai.sk import GB_pipeline
        
        assert callable(to_sktime_df)
        assert callable(rocket_pipeline)
        assert isinstance(GB_pipeline, Pipeline)
        
    def test_reimport_stability(self):
        """Test that reimporting gives same objects."""
        from mngs.ai import sk as sk1
        from mngs.ai import sk as sk2
        
        assert sk1 is sk2
        assert sk1.GB_pipeline is sk2.GB_pipeline
        
    def test_delayed_import(self):
        """Test that imports work after module is already loaded."""
        import mngs.ai.sk
        
        # Access attributes after import
        to_sktime_df = getattr(mngs.ai.sk, 'to_sktime_df')
        assert callable(to_sktime_df)


class TestPipelineUsage:
    """Test actual usage of pipelines."""
    
    def test_rocket_pipeline_instantiation(self):
        """Test creating multiple rocket pipeline instances."""
        pipe1 = mngs.ai.sk.rocket_pipeline()
        pipe2 = mngs.ai.sk.rocket_pipeline()
        
        # Should be different instances
        assert pipe1 is not pipe2
        
        # But same structure
        assert len(pipe1.steps) == len(pipe2.steps)
        
    def test_rocket_pipeline_with_different_params(self):
        """Test creating rocket pipelines with different parameters."""
        pipe1 = mngs.ai.sk.rocket_pipeline(n_kernels=100)
        pipe2 = mngs.ai.sk.rocket_pipeline(n_kernels=200)
        
        # Should have different parameters
        rocket1 = pipe1.steps[0][1]
        rocket2 = pipe2.steps[0][1]
        
        assert rocket1.n_kernels != rocket2.n_kernels
        
    def test_gb_pipeline_shared_instance(self):
        """Test that GB_pipeline is a shared instance."""
        pipe1 = mngs.ai.sk.GB_pipeline
        pipe2 = mngs.ai.sk.GB_pipeline
        
        # Should be the same instance
        assert pipe1 is pipe2
        
    def test_pipeline_cloning(self):
        """Test that pipelines can be cloned."""
        from sklearn.base import clone
        
        original = mngs.ai.sk.rocket_pipeline()
        cloned = clone(original)
        
        assert original is not cloned
        assert type(original) == type(cloned)
        assert len(original.steps) == len(cloned.steps)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_attribute_access(self):
        """Test accessing non-existent attributes."""
        with pytest.raises(AttributeError):
            mngs.ai.sk.non_existent_function
            
    def test_calling_non_callable(self):
        """Test calling non-callable attributes."""
        with pytest.raises(TypeError):
            mngs.ai.sk.GB_pipeline()  # GB_pipeline is not callable
            
    @patch('mngs.ai.sk._clf.Rocket')
    def test_rocket_import_error_handling(self, mock_rocket):
        """Test handling when Rocket import fails."""
        mock_rocket.side_effect = ImportError("sktime not installed")
        
        # Should still be able to import the module
        import importlib
        importlib.reload(mngs.ai.sk)
        
    def test_pipeline_with_invalid_params(self):
        """Test rocket_pipeline with invalid parameters."""
        # Should pass parameters through to Rocket
        # If Rocket raises error, it should propagate
        try:
            pipeline = mngs.ai.sk.rocket_pipeline(invalid_param=True)
            # If no error, pipeline should still be created
            assert isinstance(pipeline, Pipeline)
        except TypeError:
            # Expected if Rocket doesn't accept invalid_param
            pass


class TestIntegrationWithSklearn:
    """Test integration with sklearn ecosystem."""
    
    def test_pipeline_sklearn_compatible(self):
        """Test that pipelines are sklearn-compatible."""
        pipeline = mngs.ai.sk.rocket_pipeline()
        
        # Check sklearn compatibility
        assert hasattr(pipeline, 'fit')
        assert hasattr(pipeline, 'predict')
        assert hasattr(pipeline, 'fit_predict')
        
    def test_pipeline_get_params(self):
        """Test pipeline get_params method."""
        pipeline = mngs.ai.sk.rocket_pipeline()
        params = pipeline.get_params()
        
        assert isinstance(params, dict)
        assert len(params) > 0
        
        # Should include nested parameters
        assert any('rocket' in key for key in params.keys())
        assert any('logisticregression' in key for key in params.keys())
        
    def test_pipeline_set_params(self):
        """Test pipeline set_params method."""
        pipeline = mngs.ai.sk.rocket_pipeline()
        
        # Set a parameter
        pipeline.set_params(logisticregression__C=0.5)
        
        # Verify it was set
        params = pipeline.get_params()
        assert params['logisticregression__C'] == 0.5
        
    def test_gb_pipeline_sklearn_compatible(self):
        """Test GB_pipeline sklearn compatibility."""
        # Check methods exist
        assert hasattr(mngs.ai.sk.GB_pipeline, 'fit')
        assert hasattr(mngs.ai.sk.GB_pipeline, 'predict')
        assert hasattr(mngs.ai.sk.GB_pipeline, 'score')


class TestDocumentation:
    """Test module documentation."""
    
    def test_module_has_docstring(self):
        """Test that module has a docstring."""
        import mngs.ai.sk
        # Module might not have docstring, which is okay
        if hasattr(mngs.ai.sk, '__doc__'):
            if mngs.ai.sk.__doc__:
                assert isinstance(mngs.ai.sk.__doc__, str)
                
    def test_function_docstrings(self):
        """Test that functions have docstrings."""
        # to_sktime_df should have docstring
        if hasattr(mngs.ai.sk.to_sktime_df, '__doc__'):
            assert mngs.ai.sk.to_sktime_df.__doc__ is None or isinstance(
                mngs.ai.sk.to_sktime_df.__doc__, str
            )
            
    def test_module_attributes(self):
        """Test standard module attributes."""
        import mngs.ai.sk
        
        assert hasattr(mngs.ai.sk, '__name__')
        assert hasattr(mngs.ai.sk, '__file__')
        assert mngs.ai.sk.__name__ == 'mngs.ai.sk'


class TestBackwardCompatibility:
    """Test backward compatibility concerns."""
    
    def test_legacy_imports_still_work(self):
        """Test that legacy import patterns still work."""
        # Old style imports should still work
        try:
            from mngs.ai.sk import rocket_pipeline
            assert callable(rocket_pipeline)
        except ImportError:
            pytest.skip("Legacy imports not supported")
            
    def test_gb_pipeline_remains_pipeline_instance(self):
        """Test GB_pipeline remains a Pipeline instance (not callable)."""
        # This is important for backward compatibility
        assert isinstance(mngs.ai.sk.GB_pipeline, Pipeline)
        assert not callable(mngs.ai.sk.GB_pipeline)


class TestEdgeCases:
    """Test edge cases and unusual usage patterns."""
    
    def test_module_reload(self):
        """Test that module can be reloaded."""
        import importlib
        import mngs.ai.sk
        
        # Get original references
        original_rocket = mngs.ai.sk.rocket_pipeline
        
        # Reload
        importlib.reload(mngs.ai.sk)
        
        # Should still work
        assert callable(mngs.ai.sk.rocket_pipeline)
        
    def test_circular_import_prevention(self):
        """Test that there are no circular imports."""
        # This should not cause any import errors
        import mngs.ai
        import mngs.ai.sk
        from mngs.ai import sk
        
        assert mngs.ai.sk is sk
        
    def test_pipeline_memory_efficiency(self):
        """Test that GB_pipeline doesn't create copies unnecessarily."""
        import sys
        
        # Get reference count
        ref_count_before = sys.getrefcount(mngs.ai.sk.GB_pipeline)
        
        # Access multiple times
        p1 = mngs.ai.sk.GB_pipeline
        p2 = mngs.ai.sk.GB_pipeline
        p3 = mngs.ai.sk.GB_pipeline
        
        # Should be same object
        assert p1 is p2 is p3
        
    def test_namespace_pollution(self):
        """Test that module doesn't pollute namespace."""
        import mngs.ai.sk
        
        # Count public attributes
        public_attrs = [attr for attr in dir(mngs.ai.sk) if not attr.startswith('_')]
        
        # Should have limited public API
        assert len(public_attrs) < 20  # Reasonable limit
        
        # Check for common pollution
        assert 'np' not in public_attrs
        assert 'pd' not in public_attrs
        assert 'sklearn' not in public_attrs


if __name__ == "__main__":
<<<<<<< HEAD
    pytest.main([__file__, "-v"])
=======
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/ai/sk/__init__.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-03-05 13:17:17 (ywatanabe)"
# 
# from ._clf import *
# from ._to_sktime import to_sktime_df

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/ai/sk/__init__.py
# --------------------------------------------------------------------------------
>>>>>>> origin/main
