import pytorch_lightning as pl

class HistogramLoggerCallback(pl.Callback):
    """
    Logs histograms of weights and gradients for selected layers.
    """
    def __init__(self, log_every_n_epochs=1, layers_to_log=None):
        """
        Args:
            log_every_n_epochs (int): Frequency of histogram logging in epochs.
            layers_to_log (list of str): List of layer name substrings to log (e.g., "transformer", "decoder").
        """
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.layers_to_log = layers_to_log if layers_to_log else []

    def on_train_epoch_end(self, trainer, pl_module):
        # Log histograms every n epochs
        if (trainer.current_epoch + 1) % self.log_every_n_epochs == 0:
            for name, param in pl_module.named_parameters():
                # Log only selected layers if specified
                if self.layers_to_log and not any(layer in name for layer in self.layers_to_log):
                    continue

                # Log weights
                trainer.logger.experiment.add_histogram(f"{name}_weights", param, trainer.current_epoch)

                # Log gradients (if available)
                if param.grad is not None:
                    trainer.logger.experiment.add_histogram(f"{name}_gradients", param.grad, trainer.current_epoch)

