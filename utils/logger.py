from pytorch_lightning.loggers import TensorBoardLogger

class Logger:
    """
    Manages logging with TensorBoard.
    """
    @staticmethod
    def get_tensorboard_logger(save_dir, name="default", log_graph=True):
        """
        Creates a TensorBoard logger.
        Args:
            save_dir (str): Directory where logs will be saved.
            name (str): Name of the experiment.
            log_graph (bool): Adds the computational graph to tensorboard
        Returns:
            TensorBoardLogger: Configured TensorBoard logger.
        """
        return TensorBoardLogger(save_dir=save_dir, name=name, log_graph=True)

