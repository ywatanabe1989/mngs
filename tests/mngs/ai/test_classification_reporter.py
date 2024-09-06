import unittest
import numpy as np
from mngs.ai.ClassificationReporter import ClassificationReporter

class TestClassificationReporter(unittest.TestCase):
    def setUp(self):
        self.reporter = ClassificationReporter()

    def test_initialization(self):
        self.assertIsInstance(self.reporter, ClassificationReporter)

    def test_calc_metrics(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 1, 0, 1, 1])
        metrics = self.reporter.calc_metrics(y_true, y_pred)
        
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)

    def test_plot_confusion_matrix(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 1, 0, 1, 1])
        fig = self.reporter.plot_confusion_matrix(y_true, y_pred)
        
        self.assertIsNotNone(fig)

    def test_plot_roc_curve(self):
        y_true = np.array([0, 1, 1, 0])
        y_scores = np.array([[0.1, 0.9], [0.4, 0.6], [0.35, 0.65], [0.8, 0.2]])
        fig = self.reporter.plot_roc_curve(y_true, y_scores)
        
        self.assertIsNotNone(fig)

if __name__ == '__main__':
    unittest.main()
