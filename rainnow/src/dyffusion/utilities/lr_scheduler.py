"""Module for learning rate schedulers."""

from pytorch_lightning.callbacks import Callback
from torch.optim.lr_scheduler import ReduceLROnPlateau

from rainnow.src.utilities.utils import get_logger

log = get_logger(log_file_name=None, name=__name__)


class ReduceLROnPlateauCallback(Callback):
    """
    A PyTorch Lightning callback that implements the torch.ReduceLROnPlateau learning rate scheduler.

    This callback adjusts the learning rate when a metric has stopped improving.
    It monitors a specified metric and reduces the learning rate when no improvement
    is seen for a given number of epochs.

    Parameters
    ----------
    monitor : str, optional
        The metric to monitor, by default "val_loss".
    mode : str, optional
        One of "min" or "max". In "min" mode, the learning rate will be reduced when the
        quantity monitored has stopped decreasing; in "max" mode it will be reduced when
        the quantity monitored has stopped increasing, by default "min".
    factor : float, optional
        Factor by which the learning rate will be reduced, by default 0.1.
    patience : int, optional
        Number of epochs with no improvement after which learning rate will be reduced,
        by default 10.
    """

    def __init__(self, monitor="val_loss", mode="min", factor=0.1, patience=10):
        """initialisation."""
        self.monitor = monitor
        self.scheduler_config = {
            "mode": mode,
            "factor": factor,
            "patience": patience,
        }
        self.curr_lr = None
        self.scheduler = None

    def on_train_start(self, trainer, pl_module):
        """
        Initialize the scheduler at the start of training.

        This is an override and required method for a scheduler.

        Parameters
        ----------
        trainer : Trainer
            The PyTorch Lightning trainer instance.
        pl_module : LightningModule
            The PyTorch Lightning module being trained.
        """
        optimizer = trainer.optimizers[0]
        self.scheduler = ReduceLROnPlateau(optimizer, **self.scheduler_config)

    def on_validation_end(self, trainer, pl_module):
        """
        Update the learning rate based on the monitored metric at the end of each validation.

        This is an override and required method for a scheduler.

        Parameters
        ----------
        trainer : Trainer
            The PyTorch Lightning trainer instance.
        pl_module : LightningModule
            The PyTorch Lightning module being trained.
        """
        # ensure that self.scheduler is initialised.
        if self.scheduler is None:
            self.on_train_start(trainer, pl_module)

        tracked_loss = trainer.callback_metrics.get(self.monitor)
        if tracked_loss is not None:
            self.scheduler.step(tracked_loss)
            # update lr attr and log it for tracking purposes.
            if (self.curr_lr is None) or (self.curr_lr != self.scheduler._last_lr[0]):
                self.curr_lr = self.scheduler._last_lr[0]
                log.info(f"lr={self.curr_lr}")
