import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

class MetricsLoggingCallback(Callback):
    """
    Custom callback to compute and log metrics at the end of each training and validation epoch.
    """
    def on_train_epoch_end(self, trainer, pl_module):
        """
        Called at the end of the training epoch.
        """
        # Compute metrics
        train_acc = pl_module.train_accuracy.compute()
        train_prec = pl_module.train_precision.compute()
        train_rec = pl_module.train_recall.compute()
        train_f1 = pl_module.train_f1.compute()

        # Log metrics
        pl_module.log("train_accuracy", train_acc, prog_bar=True)
        pl_module.log("train_precision", train_prec, prog_bar=True)
        pl_module.log("train_recall", train_rec, prog_bar=True)
        pl_module.log("train_f1", train_f1, prog_bar=True)

        # Reset metrics
        pl_module.train_accuracy.reset()
        pl_module.train_precision.reset()
        pl_module.train_recall.reset()
        pl_module.train_f1.reset()

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Called at the end of the validation epoch.
        """
        # Compute metrics
        val_acc = pl_module.val_accuracy.compute()
        val_prec = pl_module.val_precision.compute()
        val_rec = pl_module.val_recall.compute()
        val_f1 = pl_module.val_f1.compute()

        # Log metrics
        pl_module.log("val_accuracy", val_acc, prog_bar=True)
        pl_module.log("val_precision", val_prec, prog_bar=True)
        pl_module.log("val_recall", val_rec, prog_bar=True)
        pl_module.log("val_f1", val_f1, prog_bar=True)

        # Reset metrics
        pl_module.val_accuracy.reset()
        pl_module.val_precision.reset()
        pl_module.val_recall.reset()
        pl_module.val_f1.reset()

