
"""Training-only class weighting and masked binary classification loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_pos_weight(
    labels: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute negative/positive ratio from labelled training targets only.

    Pass the training target mask here. Do not calculate this value from
    validation or test labels.
    """
    if labels.ndim != 1 or target_mask.ndim != 1:
        raise ValueError("labels and target_mask must be 1D tensors.")

    if labels.numel() != target_mask.numel():
        raise ValueError("labels and target_mask must have equal lengths.")

    if target_mask.dtype != torch.bool:
        raise TypeError("target_mask must have boolean dtype.")

    selected = labels[target_mask]

    if selected.numel() == 0:
        raise ValueError("Cannot compute class weight from an empty mask.")

    if not torch.all((selected == 0) | (selected == 1)).item():
        raise ValueError(
            "Selected targets must contain only binary labels 0 and 1."
        )

    positives = (selected == 1).sum().to(dtype=torch.float32)
    negatives = (selected == 0).sum().to(dtype=torch.float32)

    if positives.item() == 0:
        raise ValueError("Training targets contain no positive examples.")

    if negatives.item() == 0:
        raise ValueError("Training targets contain no negative examples.")

    return negatives / positives


def masked_weighted_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    target_mask: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    """Calculate weighted BCE only on the selected labelled targets."""
    if logits.ndim == 2 and logits.size(-1) == 1:
        logits = logits.squeeze(-1)

    if logits.ndim != 1:
        raise ValueError("logits must have shape [num_nodes] or [num_nodes, 1].")

    if labels.ndim != 1 or target_mask.ndim != 1:
        raise ValueError("labels and target_mask must be 1D tensors.")

    if logits.numel() != labels.numel() or labels.numel() != target_mask.numel():
        raise ValueError("logits, labels, and target_mask sizes must match.")

    if target_mask.dtype != torch.bool:
        raise TypeError("target_mask must have boolean dtype.")

    if not target_mask.any().item():
        raise ValueError("Cannot calculate loss with an empty target mask.")

    selected_labels = labels[target_mask]

    if not torch.all(
        (selected_labels == 0) | (selected_labels == 1)
    ).item():
        raise ValueError("Selected targets must contain only labels 0 and 1.")

    if pos_weight.numel() != 1:
        raise ValueError("pos_weight must be a scalar tensor.")

    return F.binary_cross_entropy_with_logits(
        logits[target_mask],
        selected_labels.to(dtype=logits.dtype),
        pos_weight=pos_weight.to(
            device=logits.device,
            dtype=logits.dtype,
        ),
    )
