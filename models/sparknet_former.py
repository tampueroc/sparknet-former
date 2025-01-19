import pytorch_lightning as pl
import torchvision.utils as vutils
import torch
import torch.optim as optim
import torch.nn.functional as F
from .encoders import FireStateEncoder, StaticLandscapeEncoder
from .transformers import TemporalTransformerEncoder
from .decoders import FeatureFusion, TransposedConvDecoder


def compute_loss(pred, target):
    """
    Computes the binary cross-entropy loss.

    Args:
        pred (Tensor): Predicted output, e.g., [B, T, H, W].
        target (Tensor): Ground truth, e.g., [B, T, H, W].

    Returns:
        loss: Loss value.
    """
    return F.binary_cross_entropy_with_logits(pred, target)

class SparkNetFormer(pl.LightningModule):
    def __init__(self, config, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # 1. Encoders
        fire_cfg = config["fire_state_encoder"]
        self.fire_state_encoder = FireStateEncoder(
            in_channels=fire_cfg["in_channels"],
            base_num_filters=fire_cfg["base_num_filters"],
            depth=fire_cfg["depth"],
            output_dim=fire_cfg["output_dim"]
        )

        static_cfg = config["static_landscape_encoder"]
        self.static_landscape_encoder = StaticLandscapeEncoder(
            in_channels=static_cfg["in_channels"],
            base_num_filters=static_cfg["base_num_filters"],
            depth=static_cfg["depth"],
            output_dim=static_cfg["output_dim"]
        )

        # 2. Fusion
        fusion_cfg = config["feature_fusion"]
        self.feature_fusion = FeatureFusion(
            fire_dim=fusion_cfg["fire_dim"],      # e.g. same as fire_cfg["output_dim"]
            static_dim=fusion_cfg["static_dim"],  # e.g. same as static_cfg["output_dim"]
            wind_dim=fusion_cfg["wind_dim"],      # dimension of scalar wind features
            d_model=fusion_cfg["d_model"],
            fusion_method=fusion_cfg.get("method", "concat"),
            use_mlp=fusion_cfg.get("use_mlp", True)
        )

        # 3. Temporal Transformer
        transformer_cfg = config["temporal_transformer"]
        self.temporal_transformer = TemporalTransformerEncoder(
            d_model=transformer_cfg["d_model"],
            nhead=transformer_cfg["nhead"],
            num_layers=transformer_cfg["num_layers"],
            dim_feedforward=transformer_cfg["dim_feedforward"],
            dropout=transformer_cfg["dropout"],
            activation=transformer_cfg.get("activation", "relu"),
            use_positional_encoding=transformer_cfg.get("use_positional_encoding", True)
        )

        # 4. Decoder
        dec_cfg = config["transposed_conv_decoder"]
        self.decoder = TransposedConvDecoder(
            in_channels=dec_cfg["in_channels"],
            base_num_filters=dec_cfg["base_num_filters"],
            depth=dec_cfg["depth"],
            out_channels=dec_cfg["out_channels"],
            use_batchnorm=dec_cfg.get("use_batchnorm", False),
            final_activation=dec_cfg.get("final_activation", None)
        )

    def forward(self, fire_sequence, static_data, wind_inputs, valid_tokens=None):
        """
        Args:
            fire_sequence (Tensor): [B, T, H, W]
            static_data (Tensor):   [B, C, H, W]
            wind_inputs (Tensor):   [B, T, <wind_features>] (e.g. direction, magnitude)
            valid_tokens (Tensor):  [B, T] - Binary mask indicating valid positions.

        Returns:
            next_fire_state (Tensor): predicted fire mask for the next timestep
        """

        fire_encodings = self.fire_state_encoder(fire_sequence)

        static_encoding = self.static_landscape_encoder(static_data)

        fused_seq = self.feature_fusion(
            fire_encodings=fire_encodings,
            static_encoding=static_encoding,
            wind_inputs=wind_inputs
        )

        temporal_out = self.temporal_transformer(fused_seq, src_key_padding_mask=~valid_tokens.bool())

        # 5. Extract the last timestep
        last_time_step = temporal_out[:, -1, :, :, :]  # [B, d_model, H, W]

        # 6. Decode to generate the predicted fire mask
        pred_fire_mask = self.decoder(last_time_step)  # [B, out_channels, H, W]

        return pred_fire_mask

    def training_step(self, batch, batch_idx):
        fire_seq, static_data, wind_inputs, isochrone_mask, valid_tokens = batch
        pred = self(fire_seq, static_data, wind_inputs, valid_tokens)
        loss = compute_loss(pred, isochrone_mask)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        fire_seq, static_data, wind_inputs, isochrone_mask, valid_tokens = batch
        pred = self(fire_seq, static_data, wind_inputs, valid_tokens)
        loss = compute_loss(pred, isochrone_mask)
        self.log("val_loss", loss, prog_bar=True)
        # Log predicted vs. target images (only for the first batch of the epoch)
        if batch_idx == 0:
            # Normalize predictions and targets to [0, 1] for TensorBoard
            pred_images = torch.sigmoid(pred.detach().cpu())  # Apply sigmoid for visualization
            target_images = isochrone_mask.detach().cpu()

            # Concatenate predictions and targets along width
            side_by_side = torch.cat([pred_images, target_images], dim=-1)  # [B, C, H, W * 2]

            # Create a grid for visualization
            comparison_grid = vutils.make_grid(side_by_side, nrow=4, normalize=True)

            # Log the comparison grid to TensorBoard
            self.logger.experiment.add_image("Predicted vs Target Masks", comparison_grid, self.current_epoch)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

