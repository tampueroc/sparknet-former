import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        """
        Weighted Focal Loss for logits.

        Args:
            alpha (float): Weight for the positive class in the range [0,1].
            gamma (float): Focusing parameter to address class imbalance.
        """
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1 - alpha])
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Compute the Weighted Focal Loss using logits.

        Args:
            inputs (Tensor): Raw logits, shape [B, T, H, W].
            targets (Tensor): Binary ground truth, shape [B, T, H, W].

        Returns:
            Tensor: Scalar loss value.
        """
        # 1) Compute standard BCE loss with logits
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # 2) Convert logits to probabilities for focal factor
        p = torch.sigmoid(inputs)

        # 3) Compute p_t (true class probability)
        pt = p * targets + (1 - p) * (1 - targets)

        # 4) Gather alpha for each pixel
        alpha_t = self.alpha.to(inputs.device).gather(0, targets.long().view(-1)).view_as(targets)

        # 5) Apply focal scaling
        focal_loss = alpha_t * (1 - pt).pow(self.gamma) * bce_loss

        return focal_loss.mean()


class BCEWithWeights(nn.Module):
    def __init__(self, pos_weight=None):
        """
        Binary Cross-Entropy Loss with optional positive weighting using logits.

        Args:
            pos_weight (float): Scalar weight for the positive class.
                               (For multi-class, this can be a 1D Tensor.)
        """
        super(BCEWithWeights, self).__init__()
        # Note: The built-in 'pos_weight' in BCEWithLogitsLoss is broadcasted
        #       only across the channel dimension. For binary classification,
        #       this can be a single scalar in a (1,) tensor.
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        """
        Compute BCE Loss with optional weighting using logits.

        Args:
            inputs (Tensor): Raw logits, shape [B, T, H, W].
            targets (Tensor): Binary ground truth, shape [B, T, H, W].

        Returns:
            Tensor: Scalar loss value.
        """
        if self.pos_weight is not None:
            # Option A: Use the built-in pos_weight argument
            # For a single class, pos_weight must be a tensor of shape [1]
            pos_weight_tensor = torch.tensor([self.pos_weight], device=inputs.device)
            return F.binary_cross_entropy_with_logits(
                inputs, targets, pos_weight=pos_weight_tensor
            )
        else:
            # Option B: No pos_weight
            return F.binary_cross_entropy_with_logits(inputs, targets)

