import unittest
import numpy as np
from mngs.ai.EarlyStopping import EarlyStopping

class TestEarlyStopping(unittest.TestCase):
    def setUp(self):
        self.early_stopping = EarlyStopping(patience=3, verbose=False, delta=0.01, direction="minimize")

    def test_initialization(self):
        self.assertEqual(self.early_stopping.patience, 3)
        self.assertFalse(self.early_stopping.verbose)
        self.assertEqual(self.early_stopping.counter, 0)
        self.assertEqual(self.early_stopping.best_score, np.Inf)
        self.assertEqual(self.early_stopping.delta, 0.01)
        self.assertEqual(self.early_stopping.direction, "minimize")

    def test_is_best(self):
        self.assertTrue(self.early_stopping.is_best(0.5))
        self.early_stopping.best_score = 0.5
        self.assertFalse(self.early_stopping.is_best(0.505))  # Within delta
        self.assertTrue(self.early_stopping.is_best(0.48))
        
        # Test maximize direction
        self.early_stopping.direction = "maximize"
        self.early_stopping.best_score = 0.5
        self.assertFalse(self.early_stopping.is_best(0.495))  # Within delta
        self.assertTrue(self.early_stopping.is_best(0.52))

    def test_early_stopping(self):
        models_spaths_dict = {'model1': 'path1', 'model2': 'path2'}
        
        # First call
        self.assertFalse(self.early_stopping(0.5, models_spaths_dict, 0))
        self.assertEqual(self.early_stopping.best_score, 0.5)

        # Improvement
        self.assertFalse(self.early_stopping(0.4, models_spaths_dict, 1))
        self.assertEqual(self.early_stopping.best_score, 0.4)

        # No improvement, but not stopping yet
        self.assertFalse(self.early_stopping(0.41, models_spaths_dict, 2))
        self.assertFalse(self.early_stopping(0.42, models_spaths_dict, 3))
        self.assertFalse(self.early_stopping(0.43, models_spaths_dict, 4))

        # Should stop now
        self.assertTrue(self.early_stopping(0.44, models_spaths_dict, 5))

    def test_save_method(self):
        models_spaths_dict = {'model1': 'path1', 'model2': 'path2'}
        self.early_stopping.save(0.3, models_spaths_dict, 10)
        self.assertEqual(self.early_stopping.best_score, 0.3)
        self.assertEqual(self.early_stopping.best_i_global, 10)
        self.assertEqual(self.early_stopping.models_spaths_dict, models_spaths_dict)

if __name__ == '__main__':
    unittest.main()
