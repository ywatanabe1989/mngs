#!/usr/bin/env python3
# Time-stamp: "2024-03-25 14:52:31 (ywatanabe)"

import os

import mngs
import numpy as np


class EarlyStopping:
    """
    Early stops the training if the validation score doesn't improve after a given patience period.

    This class is used to monitor the validation score during training and stop the process
    if no improvement is seen for a specified number of consecutive checks.

    Attributes:
        patience (int): Number of epochs to wait before early stopping.
        verbose (bool): If True, prints a message for each validation score improvement.
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        direction (str): Direction of improvement, either "minimize" or "maximize".
        counter (int): Counter for patience.
        best_score (float): Best score observed.
        best_i_global (int): Global iteration number when the best score was observed.
        models_spaths_dict (dict): Dictionary of model save paths.
    """

    def __init__(
        self, patience=7, verbose=False, delta=1e-5, direction="minimize"
    ):
        """
        Initialize the EarlyStopping object.

        Args:
            patience (int): How long to wait after last time validation score improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation score improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 1e-5
            direction (str): Direction of improvement, either "minimize" or "maximize".
                            Default: "minimize"
        """
        self.patience = patience
        self.verbose = verbose
        self.direction = direction
        self.delta = delta

        # default
        self.counter = 0
        self.best_score = np.Inf if direction == "minimize" else -np.Inf
        self.best_i_global = None
        self.models_spaths_dict = {}

    def is_best(self, val_score):
        """
        Check if the current validation score is the best so far.

        Args:
            val_score (float): The current validation score.

        Returns:
            bool: True if the current score is the best, False otherwise.
        """
        is_smaller = val_score < self.best_score - abs(self.delta)
        is_larger = self.best_score + abs(self.delta) < val_score
        return is_smaller if self.direction == "minimize" else is_larger

    def __call__(self, current_score, models_spaths_dict, i_global):
        """
        Check if training should be stopped based on the current validation score.

        Args:
            current_score (float): The current validation score.
            models_spaths_dict (dict): Dictionary of model save paths.
            i_global (int): Global iteration number.

        Returns:
            bool: True if training should be stopped, False otherwise.
        """
        # The 1st call
        if self.best_score is None:
            self.save(current_score, models_spaths_dict, i_global)
            return False

        # After the 2nd call
        if self.is_best(current_score):
            self.save(current_score, models_spaths_dict, i_global)
            self.counter = 0
            return False

        else:
            self.counter += 1
            if self.verbose:
                print(
                    f"
EarlyStopping counter: {self.counter} out of {self.patience}
"
                )
            if self.counter >= self.patience:
                if self.verbose:
                    mngs.gen.print_block("Early-stopped.", c="yellow")
                return True

    def save(self, current_score, models_spaths_dict, i_global):
        """
        Save the model when the validation score improves.

        Args:
            current_score (float): The current validation score.
            models_spaths_dict (dict): Dictionary of model save paths.
            i_global (int): Global iteration number.
        """
        if self.verbose:
            print(
                f"
Update the best score: ({self.best_score:.6f} --> {current_score:.6f})"
            )

        self.best_score = current_score
        self.best_i_global = i_global

        for model, spath in models_spaths_dict.items():
            mngs.io.save(model.state_dict(), spath)

        self.models_spaths_dict = models_spaths_dict


if __name__ == "__main__":
    pass
    # Example usage code here (commented out)
