import torch
import torch.nn as nn
import torch.nn.functional as F


def get_focal_loss(alpha=0.25, gamma=2.0):
    """
    Focal loss for binary classification.
    alpha: weight for positive class (similar to pos_weight)
    gamma: focusing parameter — higher = more focus on hard examples
    """
    def focal_loss(logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t) ** gamma * bce
        return loss.mean()
    return focal_loss
