"""
Source: https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
"""
import torch


class LRScheduler:
    """
    Learning rate scheduler. If the value does not decrease/increase for the
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """

    def __init__(self, optimizer, patience: int = 5, min_lr: float = 1e-10, factor=0.1, mode: str = "min", threshold: float = 1e-4):
        self._optimizer = optimizer
        self._patience = patience
        self._min_lr = min_lr
        self._factor = factor
        self._mode = mode

        self._lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer,
            mode=self._mode,
            patience=self._patience,
            factor=self._factor,
            min_lr=self._min_lr,
            verbose=True,
        )

    def __call__(self, val_loss):
        self._lr_scheduler.step(val_loss)
