import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        """
        Weighted Focal Loss for probabilities.

        Args:
            alpha (float): Weight for the positive class.
            gamma (float): Focusing parameter to handle class imbalance.
        """
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1 - alpha])  # Class weights
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Compute the Weighted Focal Loss.

        Args:
            inputs (Tensor): Probabilities (sigmoided outputs), shape [B, T, H, W].
            targets (Tensor): Binary ground truth, shape [B, T, H, W].

        Returns:
            Tensor: Scalar loss value.
        """
        # Ensure inputs are valid probabilities
        inputs = torch.clamp(inputs, min=1e-7, max=1 - 1e-7)

        # Compute binary cross-entropy loss
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

        # Compute p_t (true class probability)
        pt = inputs * targets + (1 - inputs) * (1 - targets)

        # Apply focal scaling
        alpha_t = self.alpha.to(inputs.device).gather(0, targets.long().view(-1))
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        return focal_loss.mean()


class BCEWithWeights(nn.Module):
    def __init__(self, pos_weight=None):
        """
        Binary Cross-Entropy Loss with optional positive weighting.

        Args:
            pos_weight (float): Scalar weight for the positive class.
        """
        super(BCEWithWeights, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        """
        Compute BCE Loss with optional weighting.

        Args:
            inputs (Tensor): Probabilities (sigmoided outputs), shape [B, T, H, W].
            targets (Tensor): Binary ground truth, shape [B, T, H, W].

        Returns:
            Tensor: Scalar loss value.
        """
        if self.pos_weight is not None:
            # Create a weight tensor based on pos_weight
            weight = torch.ones_like(targets)
            weight = torch.where(targets == 1, weight * self.pos_weight, weight)
        else:
            # Uniform weights if no pos_weight is provided
            weight = None

        # Compute weighted BCE loss
        return F.binary_cross_entropy(inputs, targets, weight=weight)

