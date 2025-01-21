from pytorch_lightning.callbacks import Callback

class DynamicAlphaCallback(Callback):
    def __init__(self, loss_fn_attr="loss_fn"):
        """
        Callback to dynamically compute class imbalance (alpha) for focal loss.

        Args:
            loss_fn_attr: The attribute name of the loss function in the model (default: "loss_fn").
            auto_alpha_key: The key in the hyperparameters to check if auto alpha is enabled.
        """
        self.loss_fn_attr = loss_fn_attr

    def on_train_epoch_start(self, trainer, pl_module):
        # Collect the dataloader for the training dataset
        dataloader = trainer.train_dataloader

        # Calculate class imbalance (alpha)
        fire_pixel_count = 0
        non_fire_pixel_count = 0
        for batch in dataloader:
            # Get the target mask (batch[3] should be the isochrone_mask in your case)
            isochrone_mask = batch[3]
            fire_pixel_count += (isochrone_mask == 1).sum().item()
            non_fire_pixel_count += (isochrone_mask == 0).sum().item()

        total_pixels = fire_pixel_count + non_fire_pixel_count
        if total_pixels > 0:
            alpha_1 = float(non_fire_pixel_count) / total_pixels
            alpha_0 = float(fire_pixel_count) / total_pixels

            # Update the alpha value in the focal loss
            if hasattr(pl_module, self.loss_fn_attr):
                loss_fn = getattr(pl_module, self.loss_fn_attr)
                loss_fn.alpha = alpha_1  # Use float values directly
                pl_module.log("dynamic_alpha_1", alpha_1)
                pl_module.log("dynamic_alpha_0", alpha_0)

