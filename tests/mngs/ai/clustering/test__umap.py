#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-05-25 17:10:00 (ywatanabe)"
# File: ./tests/mngs/ai/clustering/test__umap.py

"""
Test module for mngs.ai.clustering._umap
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch
import matplotlib.pyplot as plt
import mngs


class TestUmap:
    """Test cases for umap function"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing"""
        np.random.seed(42)
        n_samples = 100
        n_features = 50
        data = np.random.randn(n_samples, n_features)
        labels = np.random.randint(0, 3, n_samples)
        return data, labels
    
    def test_umap_basic_functionality(self, sample_data):
        """Test basic UMAP functionality with minimal parameters"""
        data, labels = sample_data
        
        # Import should work
        from mngs.ai.clustering import umap
        
        # Basic call should work
        with patch('matplotlib.pyplot.show'):
            result = umap(data, labels)
        
        # Should return a tuple
        assert isinstance(result, tuple)
    
    def test_umap_with_hues(self, sample_data):
        """Test UMAP with hue coloring"""
        data, labels = sample_data
        hues = np.random.randint(0, 2, len(labels))
        
        from mngs.ai.clustering import umap
        
        with patch('matplotlib.pyplot.show'):
            result = umap(data, labels, hues=hues)
        
        assert isinstance(result, tuple)
    
    def test_umap_supervised_mode(self, sample_data):
        """Test supervised UMAP mode"""
        data, labels = sample_data
        
        from mngs.ai.clustering import umap
        
        with patch('matplotlib.pyplot.show'):
            result = umap(data, labels, supervised=True)
        
        assert isinstance(result, tuple)
    
    def test_umap_with_existing_axes(self, sample_data):
        """Test UMAP plotting on existing axes"""
        data, labels = sample_data
        
        fig, ax = plt.subplots()
        
        from mngs.ai.clustering import umap
        
        with patch('matplotlib.pyplot.show'):
            result = umap(data, labels, axes=ax)
        
        assert isinstance(result, tuple)
        plt.close(fig)
    
    def test_umap_with_pretrained_model(self, sample_data):
        """Test UMAP with pre-fitted model"""
        data, labels = sample_data
        
        # Mock a pre-fitted UMAP model
        mock_model = Mock()
        mock_model.transform.return_value = np.random.randn(len(data), 2)
        
        from mngs.ai.clustering import umap
        
        with patch('matplotlib.pyplot.show'):
            result = umap(data, labels, umap_model=mock_model)
        
        assert isinstance(result, tuple)
        # Model's transform should have been called
        mock_model.transform.assert_called()
    
    def test_umap_visualization_parameters(self, sample_data):
        """Test UMAP with custom visualization parameters"""
        data, labels = sample_data
        
        from mngs.ai.clustering import umap
        
        with patch('matplotlib.pyplot.show'):
            result = umap(
                data, 
                labels,
                title="Custom Title",
                alpha=0.5,
                s=10,
                use_independent_legend=True
            )
        
        assert isinstance(result, tuple)
    
    def test_umap_input_validation(self):
        """Test UMAP input validation"""
        from mngs.ai.clustering import umap
        
        # Test with mismatched data and labels
        data = np.random.randn(100, 50)
        labels = np.random.randint(0, 3, 50)  # Wrong size
        
        with pytest.raises(Exception):
            with patch('matplotlib.pyplot.show'):
                umap(data, labels)
    
    @pytest.mark.parametrize("n_samples,n_features,n_classes", [
        (50, 10, 2),
        (100, 50, 3),
        (200, 100, 5),
    ])
    def test_umap_various_data_sizes(self, n_samples, n_features, n_classes):
        """Test UMAP with various data sizes"""
        np.random.seed(42)
        data = np.random.randn(n_samples, n_features)
        labels = np.random.randint(0, n_classes, n_samples)
        
        from mngs.ai.clustering import umap
        
        with patch('matplotlib.pyplot.show'):
            result = umap(data, labels)
        
        assert isinstance(result, tuple)


# EOF