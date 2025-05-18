#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-16 14:15:20 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/tests/mngs/io/_save_modules/test__catboost.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/io/_save_modules/test__catboost.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import os
import tempfile
import pytest
import numpy as np


def _is_catboost_available():
    """Check if CatBoost is available."""
    try:
        import catboost
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _is_catboost_available(), reason="CatBoost is not installed")
def test_save_catboost_model():
    """Test saving a CatBoost model."""
    from mngs.io._save_modules._catboost import _save_catboost
    import catboost
    
    # Create a simple dataset
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([0, 1, 0])
    
    # Create and train a simple model
    model = catboost.CatBoost(params={'iterations': 2, 'verbose': False})
    model.fit(X, y)
    
    # Create temp file path
    with tempfile.NamedTemporaryFile(suffix=".cbm", delete=False) as tmp:
        temp_path = tmp.name
        
    try:
        # Save the model
        _save_catboost(model, temp_path)
        
        # Verify the file exists
        assert os.path.exists(temp_path)
        
        # Load the model back
        loaded_model = catboost.CatBoost()
        loaded_model.load_model(temp_path)
        
        # Check that the model works for predictions
        # Both models should give the same predictions on the training data
        original_preds = model.predict(X)
        loaded_preds = loaded_model.predict(X)
        
        assert np.array_equal(original_preds, loaded_preds)
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@pytest.mark.skipif(not _is_catboost_available(), reason="CatBoost is not installed")
def test_save_catboost_classifier():
    """Test saving a CatBoostClassifier model."""
    from mngs.io._save_modules._catboost import _save_catboost
    import catboost
    
    # Create a simple dataset
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([0, 1, 0])
    
    # Create and train a classifier
    clf = catboost.CatBoostClassifier(iterations=2, verbose=False)
    clf.fit(X, y)
    
    # Create temp file path
    with tempfile.NamedTemporaryFile(suffix=".cbm", delete=False) as tmp:
        temp_path = tmp.name
        
    try:
        # Save the model
        _save_catboost(clf, temp_path)
        
        # Verify the file exists
        assert os.path.exists(temp_path)
        
        # Load the model back
        loaded_clf = catboost.CatBoostClassifier()
        loaded_clf.load_model(temp_path)
        
        # Check that the model works for predictions
        # Both models should give the same predictions on the training data
        original_preds = clf.predict(X)
        loaded_preds = loaded_clf.predict(X)
        
        assert np.array_equal(original_preds, loaded_preds)
        
        # Also check probabilities
        original_probs = clf.predict_proba(X)
        loaded_probs = loaded_clf.predict_proba(X)
        
        assert np.allclose(original_probs, loaded_probs)
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save_modules/_catboost.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-16 12:29:11 (ywatanabe)"
# # File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save_modules/_catboost.py
# 
# 
# def _save_catboost(obj, spath):
#     """
#     Save a CatBoost model.
#     
#     Parameters
#     ----------
#     obj : catboost.CatBoost
#         The CatBoost model to save.
#     spath : str
#         Path where the CatBoost model file will be saved.
#         
#     Returns
#     -------
#     None
#     """
#     obj.save_model(spath)
# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save_modules/_catboost.py
# --------------------------------------------------------------------------------
