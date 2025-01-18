from pytorch_lightning.loggers import TensorBoardLogger

class Logger:
    """
    Manages logging with TensorBoard.
    """
    @staticmethod
    def get_tensorboard_logger(save_dir, name="default"):
        """
        Creates a TensorBoard logger.
        Args:
            save_dir (str): Directory where logs will be saved.
            name (str): Name of the experiment.
        Returns:
            TensorBoardLogger: Configured TensorBoard logger.
        """
        return TensorBoardLogger(save_dir=save_dir, name=name)

