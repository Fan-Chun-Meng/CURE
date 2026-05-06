import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(1e-6)


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=-1)
    return -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)


def average_entropy(logits: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    ent = entropy_from_logits(logits)
    if mask is None:
        return ent.mean()
    weights = mask.float()
    if weights.sum() <= 0:
        return ent.mean()
    return (ent * weights).sum() / weights.sum()


def unwrap_stream_batch(images, device: torch.device) -> torch.Tensor:
    if isinstance(images, dict):
        images = images.get("pixel_values", next(iter(images.values())))

    if isinstance(images, (list, tuple)):
        if not images:
            raise ValueError("Empty image batch received from the test loader.")
        images = images[0]

    if not isinstance(images, torch.Tensor):
        raise TypeError(f"Unsupported input type for images: {type(images)}")

    images = images.to(device)
    if images.dim() == 5:
        images = images[:, 0]
    elif images.dim() == 3:
        images = images.unsqueeze(0)
    return images.float()


def freeze_all_model_params(model: nn.Module) -> None:
    model.eval()
    model.requires_grad_(False)


def configure_image_encoder_norm_only_adaptation(model: nn.Module) -> List[torch.nn.Parameter]:
    freeze_all_model_params(model)

    norm_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.LayerNorm,
        nn.GroupNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
    )

    visual_encoder = getattr(model, "visual", None)
    if visual_encoder is None:
        raise AttributeError("Expected the CLIP model to expose a `visual` encoder.")

    trainable_params: List[torch.nn.Parameter] = []
    for module in visual_encoder.modules():
        if isinstance(module, norm_types):
            if getattr(module, "weight", None) is not None:
                module.weight.requires_grad_(True)
                trainable_params.append(module.weight)
            if getattr(module, "bias", None) is not None:
                module.bias.requires_grad_(True)
                trainable_params.append(module.bias)

    return trainable_params
