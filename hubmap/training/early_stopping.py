"""
Source: https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
"""
from tqdm.auto import tqdm


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    @property
    def early_stop(self):
        return self._early_stop

    def __init__(self, patience: int = 5, min_delta: float = 0):
        """
        Parameters
        ----------
        patience : int, optional
            How many epochs to wait before stopping when loss is not improving,
            by default 5
        min_delta : float, optional
            Minimum difference between new loss and old loss for new loss to be
            considered as an improvement, by default 0
        """
        self._patience = patience
        self._min_delta = min_delta
        self._counter = 0
        self._best_loss = None
        self._early_stop = False

    def __call__(self, val_loss):
        if self._best_loss is None:
            self._best_loss = val_loss
        elif self._best_loss - val_loss > self._min_delta:
            self._best_loss = val_loss
            # reset the counter if the validation loss imrpoves
            self._counter = 0
        elif self._best_loss - val_loss < self._min_delta:
            self._counter += 1
            tqdm.write(f"EaryStopping counter: {self._counter} out of {self._patience}")
            if self._counter >= self._patience:
                tqdm.write("Early Stopping Activated")
                self._early_stop = True
