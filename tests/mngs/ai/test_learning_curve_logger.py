import unittest
import numpy as np
from mngs.ai.LearningCurveLogger import LearningCurveLogger

class TestLearningCurveLogger(unittest.TestCase):
    def setUp(self):
        self.logger = LearningCurveLogger()

    def test_initialization(self):
        self.assertIsInstance(self.logger, LearningCurveLogger)

    def test_update(self):
        self.logger.update(1, 0.5, 0.6)
        self.assertEqual(len(self.logger.epochs), 1)
        self.assertEqual(len(self.logger.train_losses), 1)
        self.assertEqual(len(self.logger.val_losses), 1)

    def test_plot(self):
        self.logger.update(1, 0.5, 0.6)
        self.logger.update(2, 0.4, 0.5)
        fig = self.logger.plot()
        self.assertIsNotNone(fig)

    def test_save_and_load(self):
        self.logger.update(1, 0.5, 0.6)
        self.logger.update(2, 0.4, 0.5)
        self.logger.save("test_logger.pkl")
        
        new_logger = LearningCurveLogger.load("test_logger.pkl")
        self.assertEqual(new_logger.epochs, self.logger.epochs)
        self.assertEqual(new_logger.train_losses, self.logger.train_losses)
        self.assertEqual(new_logger.val_losses, self.logger.val_losses)

if __name__ == '__main__':
    unittest.main()
