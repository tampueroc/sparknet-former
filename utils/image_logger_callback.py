import torch
import torchvision.utils as vutils
from pytorch_lightning.callbacks import Callback

class ImagePredictionLogger(Callback):
    """
    Custom callback to log predictions vs. targets as images in TensorBoard during validation.
    """
    def __init__(self, threshold=0.5, log_interval=1, num_images=8):
        """
        Args:
            threshold (float): Threshold to binarize predictions.
            log_interval (int): Log images every 'log_interval' epochs.
            num_images (int): Number of images to log per validation epoch.
        """
        super().__init__()
        self.threshold = threshold
        self.log_interval = log_interval
        self.num_images = num_images

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        Collect predictions for the first N images during validation.
        """
        if batch_idx == 0 and (pl_module.current_epoch % self.log_interval == 0):
            pred, target = outputs["predictions"], outputs["targets"]

            # Normalize predictions to [0, 1] for visualization
            pred_images = torch.sigmoid(pred.detach().cpu())  # Apply sigmoid for visualization
            pred_binary = (pred_images > self.threshold).float()  # Binary mask
            target_images = target.detach().cpu()

            # Concatenate predictions, binary masks, and targets
            combined_images = torch.cat([pred_images[:self.num_images],
                                         pred_binary[:self.num_images],
                                         target_images[:self.num_images]], dim=-1)  # [B, C, H, W * 3]

            # Create a grid for visualization
            comparison_grid = vutils.make_grid(combined_images, nrow=4, normalize=True, value_range=(0, 1))

            # Log the comparison grid to TensorBoard
            trainer.logger.experiment.add_image(
                "Predictions | Binary Mask | Targets",
                comparison_grid,
                global_step=pl_module.current_epoch  # Use `global_step` to track progression
            )

