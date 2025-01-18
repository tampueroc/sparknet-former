from pytorch_lightning.callbacks import ModelCheckpoint

class CheckpointHandler:
    """
    Manages model checkpointing during training.
    """
    @staticmethod
    def get_checkpoint_callback(dirpath, filename="best_model", monitor="val_loss", mode="min", save_top_k=1):
        """
        Creates a ModelCheckpoint callback.
        Args:
            dirpath (str): Directory to save checkpoints.
            filename (str): Name prefix for checkpoint files.
            monitor (str): Metric to monitor.
            mode (str): 'min' or 'max', depending on whether to minimize or maximize the monitored metric.
            save_top_k (int): Number of top checkpoints to save.
        Returns:
            ModelCheckpoint: Configured ModelCheckpoint callback.
        """
        return ModelCheckpoint(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            save_weights_only=True
        )

