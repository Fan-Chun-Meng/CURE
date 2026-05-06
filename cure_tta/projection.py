from collections import deque
from typing import Dict, Optional, Sequence, Tuple

import torch


class GradientHistoryBuffer:
    def __init__(self, max_size: int) -> None:
        self.buffer = deque(maxlen=max_size)

    def append(self, grad_vec: torch.Tensor) -> None:
        self.buffer.append(grad_vec.detach().cpu())

    def as_matrix(self, device: torch.device) -> Optional[torch.Tensor]:
        if not self.buffer:
            return None
        return torch.stack(list(self.buffer), dim=1).to(device)


class DisentangledNormOptimizer:
    def __init__(
        self,
        params: Sequence[torch.nn.Parameter],
        lr: float,
        weight_decay: float,
        history_size: int,
        low_rank_rank: int,
        svd_update_freq: int,
        plasticity_scale: float = 1.0,
        stability_scale: float = 1.0,
        projection_mode: str = "full",
    ) -> None:
        self.params = [param for param in params if param.requires_grad]
        self.optimizer = torch.optim.AdamW(self.params, lr=lr, weight_decay=weight_decay)
        self.history = GradientHistoryBuffer(history_size)
        self.low_rank_rank = low_rank_rank
        self.svd_update_freq = max(1, svd_update_freq)
        self.plasticity_scale = plasticity_scale
        self.stability_scale = stability_scale
        self.projection_mode = projection_mode
        self.update_step = 0
        self.cached_basis: Optional[torch.Tensor] = None

    def _flatten_grads(self, grads: Sequence[Optional[torch.Tensor]]) -> torch.Tensor:
        chunks = []
        for param, grad in zip(self.params, grads):
            if grad is None:
                chunks.append(torch.zeros_like(param, dtype=torch.float32).reshape(-1))
            else:
                chunks.append(grad.detach().float().reshape(-1))
        return torch.cat(chunks)

    def compute_grad_vector(self, loss: torch.Tensor, retain_graph: bool = False) -> torch.Tensor:
        grads = torch.autograd.grad(loss, self.params, retain_graph=retain_graph, allow_unused=True)
        return self._flatten_grads(grads)

    def _assign_grad_vector(self, grad_vec: torch.Tensor) -> None:
        offset = 0
        for param in self.params:
            numel = param.numel()
            grad_chunk = grad_vec[offset : offset + numel].view_as(param).to(param.dtype)
            param.grad = grad_chunk.clone()
            offset += numel

    def _history_basis(self, device: torch.device) -> Optional[torch.Tensor]:
        should_refresh = (
            self.cached_basis is None
            or self.update_step % self.svd_update_freq == 0
            or self.cached_basis.device != device
        )
        if not should_refresh:
            return self.cached_basis

        matrix = self.history.as_matrix(device)
        if matrix is None or matrix.size(1) < 2:
            self.cached_basis = None
            return None

        matrix = matrix - matrix.mean(dim=1, keepdim=True)
        try:
            basis, _, _ = torch.linalg.svd(matrix, full_matrices=False)
        except RuntimeError:
            self.cached_basis = None
            return None

        rank = min(self.low_rank_rank, basis.size(1))
        self.cached_basis = None if rank <= 0 else basis[:, :rank]
        return self.cached_basis

    def project(
        self,
        plastic_grad: torch.Tensor,
        stability_grad: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if stability_grad is None:
            return plastic_grad, {
                "subspace_rank": 0.0,
                "plastic_norm": float(plastic_grad.norm().item()),
                "stability_norm": 0.0,
            }

        raw_grad = plastic_grad + stability_grad
        if self.projection_mode == "none":
            return raw_grad, {
                "subspace_rank": 0.0,
                "plastic_norm": float(plastic_grad.norm().item()),
                "stability_norm": float(stability_grad.norm().item()),
            }

        basis = self._history_basis(plastic_grad.device)
        if basis is None:
            return raw_grad, {
                "subspace_rank": 0.0,
                "plastic_norm": float(plastic_grad.norm().item()),
                "stability_norm": float(stability_grad.norm().item()),
            }

        if self.projection_mode == "null_only":
            plastic_component = plastic_grad
        else:
            plastic_component = basis @ (basis.t() @ plastic_grad)

        if self.projection_mode == "principal_only":
            stability_component = stability_grad
        else:
            stability_component = stability_grad - basis @ (basis.t() @ stability_grad)

        final_grad = self.plasticity_scale * plastic_component + self.stability_scale * stability_component
        return final_grad, {
            "subspace_rank": float(basis.size(1)),
            "plastic_norm": float(plastic_component.norm().item()),
            "stability_norm": float(stability_component.norm().item()),
        }

    def step(
        self,
        plastic_loss: Optional[torch.Tensor],
        stability_loss: Optional[torch.Tensor] = None,
        precomputed_plastic_grad: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        self.optimizer.zero_grad(set_to_none=True)

        if precomputed_plastic_grad is not None:
            plastic_grad = precomputed_plastic_grad
        elif plastic_loss is not None:
            plastic_grad = self.compute_grad_vector(plastic_loss, retain_graph=stability_loss is not None)
        else:
            raise ValueError("Either plastic_loss or precomputed_plastic_grad must be provided.")

        if stability_loss is None:
            final_grad = plastic_grad
            raw_grad = plastic_grad
            info = {
                "subspace_rank": 0.0,
                "plastic_norm": float(plastic_grad.norm().item()),
                "stability_norm": 0.0,
            }
        else:
            stability_grad = self.compute_grad_vector(stability_loss, retain_graph=False)
            final_grad, info = self.project(plastic_grad, stability_grad)
            raw_grad = plastic_grad + stability_grad

        self._assign_grad_vector(final_grad)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.history.append(raw_grad)
        self.update_step += 1
        return info
