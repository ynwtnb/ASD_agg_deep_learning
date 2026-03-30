import torch
import torch.nn as nn


def get_loss_fn(pos_weight: torch.Tensor = None) -> nn.BCEWithLogitsLoss:
    """
    BCEWithLogitsLoss for binary aggression classification.
    PatchTSTForClassification defaults to MSE, this overrides it.

    Usage:
        criterion = get_loss_fn(pos_weight=torch.tensor([9.0]))
        loss = criterion(logits.squeeze(), labels.float())
    """
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)