from pytorch_lightning.callbacks import EarlyStopping

class EarlyStoppingHandler:
    """
    Manages early stopping during training.
    """
    @staticmethod
    def get_early_stopping_callback(monitor="val_loss", patience=5, mode="min"):
        """
        Creates an EarlyStopping callback.
        Args:
            monitor (str): Metric to monitor.
            patience (int): Number of epochs to wait before stopping.
            mode (str): 'min' or 'max', depending on whether to minimize or maximize the monitored metric.
        Returns:
            EarlyStopping: Configured EarlyStopping callback.
        """
        return EarlyStopping(
            monitor=monitor,
            patience=patience,
            mode=mode,
            verbose=True
        )

