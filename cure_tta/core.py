"""
Main paper-facing method implementation.

The four paper-visible ideas are intentionally kept explicit here:
1. Neighbor-aware semantic voting.
2. Dual memory with short/long cache fusion.
3. Disentangled principal / null-space projection.
4. Prototype alignment between memory and text space.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .memory import DualMemoryBuffer, MemoryEntry
from .projection import DisentangledNormOptimizer
from .prompts import NeighborAwarePromptBank, PromptState, TextPrototypeAdapter
from .utils import (
    average_entropy,
    configure_image_encoder_norm_only_adaptation,
    entropy_from_logits,
    freeze_all_model_params,
    safe_normalize,
    unwrap_stream_batch,
)


class MarkTTAEngine:
    def __init__(self, model: nn.Module, classnames, template, cupl_path: str, args) -> None:
        self.model = model
        self.args = args
        self.device = next(model.parameters()).device
        self.num_classes = len(classnames)
        self.feature_dim = int(model.text_projection.shape[1])
        self.stream_step = 0
        freeze_all_model_params(self.model)

        self.prompt_bank = NeighborAwarePromptBank(
            classnames=classnames,
            template=template,
            cupl_path=cupl_path,
            prompts_per_class=args.prompts_per_class,
            num_bins=args.num_bins,
            prompt_source=args.prompt_source,
        )
        self.base_prompt_state = self.prompt_bank.encode(self.model, self.device)
        self.text_adapter = TextPrototypeAdapter(
            num_classes=self.num_classes,
            feature_dim=self.feature_dim,
            adapt_bins=not args.no_text_adapt_bins,
        ).to(self.device)

        trainable_params: List[torch.nn.Parameter] = []
        if args.adapt_side in {"image_only", "hybrid"}:
            trainable_params.extend(configure_image_encoder_norm_only_adaptation(self.model))
        if args.adapt_side in {"text_only", "hybrid"}:
            trainable_params.extend(list(self.text_adapter.parameters()))
        if not trainable_params:
            raise ValueError(f"No trainable parameters configured for adapt side: {args.adapt_side}")

        self.memory = DualMemoryBuffer(
            num_classes=self.num_classes,
            feature_dim=self.feature_dim,
            short_capacity=args.short_capacity,
            long_capacity=args.long_capacity,
        )
        self.optimizer = DisentangledNormOptimizer(
            params=trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            history_size=args.grad_history_size,
            low_rank_rank=args.low_rank_rank,
            svd_update_freq=args.svd_update_freq,
            plasticity_scale=args.plasticity_scale,
            stability_scale=args.stability_scale,
            projection_mode=args.projection_mode,
        )

    def current_prompt_state(self) -> PromptState:
        if self.args.adapt_side in {"text_only", "hybrid"}:
            return self.text_adapter(self.base_prompt_state)
        return self.base_prompt_state

    def logit_scale(self) -> torch.Tensor:
        return self.model.logit_scale.exp().detach().float()

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        image_features = self.model.encode_image(images)
        return safe_normalize(image_features.float(), dim=-1)

    def neighbor_reliability(
        self,
        image_features: torch.Tensor,
        prompt_state: PromptState,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        standard_scores = self.logit_scale() * (image_features @ prompt_state.standard_prototypes.t())
        standard_pred = standard_scores.argmax(dim=-1)
        if self.args.disable_neighbor_reliability:
            per_bin_pred = standard_pred.unsqueeze(1)
            majority_pred = standard_pred
            vote_count = torch.ones_like(standard_pred)
            stability = torch.ones_like(standard_pred, dtype=image_features.dtype)
            reliability = torch.ones_like(stability)
            return per_bin_pred, majority_pred, vote_count, stability, reliability

        bin_scores = torch.einsum("bd,cmd->bmc", image_features, prompt_state.bin_prototypes)
        per_bin_pred = bin_scores.argmax(dim=-1)
        majority_pred = torch.mode(per_bin_pred, dim=1).values
        vote_count = (per_bin_pred == majority_pred.unsqueeze(1)).sum(dim=1)

        stability = prompt_state.bin_prototypes.size(1) / vote_count.float().clamp_min(1.0)
        reliability = torch.where(
            majority_pred == standard_pred,
            torch.ones_like(stability),
            torch.full_like(stability, self.args.gamma),
        )
        return per_bin_pred, majority_pred, vote_count, stability, reliability

    def alignment_targets(self, prompt_state: PromptState) -> torch.Tensor:
        if self.args.align_prototype_source == "standard":
            return prompt_state.standard_prototypes
        if self.args.align_prototype_source == "final_bin":
            return prompt_state.final_bin_prototypes
        if self.args.align_prototype_source == "hybrid":
            return safe_normalize(
                0.5 * (prompt_state.standard_prototypes + prompt_state.final_bin_prototypes),
                dim=-1,
            )
        raise ValueError(f"Unknown align prototype source: {self.args.align_prototype_source}")

    def memory_cache_logits(
        self,
        image_features: torch.Tensor,
        memory_name: str,
        alpha: float,
        beta: float,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        logits = torch.zeros(image_features.size(0), self.num_classes, device=self.device, dtype=image_features.dtype)
        if alpha <= 0:
            return logits, {"active_classes": 0.0}

        prototypes, valid_mask = self.memory.prototypes(memory_name, self.device)
        if not valid_mask.any():
            return logits, {"active_classes": 0.0}

        valid_indices = valid_mask.nonzero(as_tuple=False).squeeze(1)
        cache_keys = prototypes[valid_mask]
        affinity = image_features @ cache_keys.t()
        cache_scores = torch.exp(beta * (affinity - 1.0))
        logits[:, valid_indices] = alpha * cache_scores
        return logits, {"active_classes": float(valid_indices.numel())}

    def fused_logits(
        self,
        image_features: torch.Tensor,
        prompt_state: PromptState,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        standard_logits = self.logit_scale() * (image_features @ prompt_state.standard_prototypes.t())
        short_cache_logits, short_info = self.memory_cache_logits(
            image_features,
            memory_name="short",
            alpha=self.args.short_cache_alpha,
            beta=self.args.short_cache_beta,
        )
        long_cache_logits, long_info = self.memory_cache_logits(
            image_features,
            memory_name="long",
            alpha=self.args.long_cache_alpha,
            beta=self.args.long_cache_beta,
        )
        fused_logits = standard_logits + short_cache_logits + long_cache_logits
        info = {
            "short_cache_active_classes": short_info["active_classes"],
            "long_cache_active_classes": long_info["active_classes"],
            "short_cache_norm": float(short_cache_logits.norm(dim=-1).mean().item()),
            "long_cache_norm": float(long_cache_logits.norm(dim=-1).mean().item()),
        }
        return standard_logits, fused_logits, info

    def should_update(
        self,
        fused_logits: torch.Tensor,
        majority_pred: torch.Tensor,
        vote_fraction: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        probs = fused_logits.softmax(dim=-1)
        fused_pred = probs.argmax(dim=-1)
        max_prob = probs.max(dim=-1).values

        update_mask = vote_fraction >= self.args.update_min_vote
        update_mask = update_mask & (max_prob >= self.args.update_min_prob)
        if self.args.update_require_agreement:
            update_mask = update_mask & (fused_pred == majority_pred)

        return update_mask, {
            "update_active": float(update_mask.float().mean().item()),
            "mean_max_prob": float(max_prob.mean().item()),
            "mean_update_vote_fraction": float(vote_fraction.mean().item()),
            "mean_fused_entropy": float(entropy_from_logits(fused_logits).mean().item()),
        }

    def compute_consistency_grad(
        self,
        images: Optional[torch.Tensor] = None,
        loss: Optional[torch.Tensor] = None,
        retain_graph: bool = False,
    ) -> torch.Tensor:
        if loss is None:
            if images is None:
                raise ValueError("Need either a precomputed loss or input images.")
            prompt_state = self.current_prompt_state()
            features = self.encode_image(images)
            logits = self.logit_scale() * (features @ prompt_state.standard_prototypes.t())
            loss = entropy_from_logits(logits).mean()
        if not loss.requires_grad:
            return torch.zeros(
                sum(param.numel() for param in self.optimizer.params),
                device=self.device,
                dtype=torch.float32,
            )
        return self.optimizer.compute_grad_vector(loss, retain_graph=retain_graph)

    def promote_to_long_memory(self, current_grad: torch.Tensor) -> Dict[str, float]:
        short_samples = self.memory.get_short_samples()
        if not short_samples:
            return {"promoted": 0.0, "selected_classes": 0.0}

        entry_scores = []
        class_scores: Dict[int, List[float]] = defaultdict(list)
        for sample in short_samples:
            if sample.grad_vector is None:
                if sample.image is None:
                    continue
                image = sample.image.unsqueeze(0).to(self.device)
                grad_j = self.compute_consistency_grad(images=image)
                sample.grad_vector = grad_j.detach().cpu()
            else:
                grad_j = sample.grad_vector.to(self.device)

            score = self.args.lr * torch.dot(current_grad, grad_j).item()
            class_scores[sample.pseudo_label].append(score)
            entry_scores.append((score, sample))

        if not entry_scores:
            return {"promoted": 0.0, "selected_classes": 0.0}

        ranked_classes = sorted(class_scores.keys(), key=lambda cls_idx: float(np.mean(class_scores[cls_idx])))
        keep_class_count = min(max(1, self.args.long_select_classes), len(ranked_classes))
        selected_classes = set(ranked_classes[:keep_class_count])

        promoted = 0
        for cls_idx in selected_classes:
            cls_candidates = [(score, sample) for score, sample in entry_scores if sample.pseudo_label == cls_idx]
            cls_candidates.sort(key=lambda item: item[0])
            for score, sample in cls_candidates[: self.args.long_samples_per_class]:
                self.memory.push_long(
                    MemoryEntry(
                        feature=sample.feature.clone(),
                        pseudo_label=sample.pseudo_label,
                        reweighted_entropy=sample.reweighted_entropy,
                        image=sample.image.clone() if sample.image is not None else None,
                        reliability=sample.reliability,
                        stability=sample.stability,
                        long_score=score,
                        grad_vector=sample.grad_vector.clone() if sample.grad_vector is not None else None,
                    ),
                    step=self.stream_step,
                )
                promoted += 1

        return {"promoted": float(promoted), "selected_classes": float(len(selected_classes))}

    def _alignment_loss_from_memory(
        self,
        memory_name: str,
        memory_source: Dict[int, List[MemoryEntry]],
        final_text_prototypes: torch.Tensor,
        max_classes: int,
    ) -> Optional[torch.Tensor]:
        losses = []
        scale = self.logit_scale()

        selected_classes = self.memory.select_recent_classes(memory_name, max_classes)
        if not selected_classes:
            return None

        for cls_idx in selected_classes:
            items = memory_source.get(cls_idx, [])
            images = [item.image for item in items if item.image is not None]
            if not images:
                continue

            feature_sum = None
            total_count = 0
            for start in range(0, len(images), self.args.align_chunk_size):
                chunk_images = torch.stack(images[start : start + self.args.align_chunk_size], dim=0).to(self.device)
                chunk_features = self.encode_image(chunk_images)
                chunk_sum = chunk_features.sum(dim=0, keepdim=True)
                feature_sum = chunk_sum if feature_sum is None else feature_sum + chunk_sum
                total_count += chunk_features.size(0)

            if feature_sum is None or total_count == 0:
                continue

            memory_prototype = safe_normalize(feature_sum / total_count, dim=-1)
            logits = scale * (memory_prototype @ final_text_prototypes.t())
            target = torch.tensor([cls_idx], device=self.device)
            losses.append(F.cross_entropy(logits, target))

        if not losses:
            return None
        return sum(losses) / len(losses)

    def alignment_loss(
        self,
        final_text_prototypes: torch.Tensor,
        include_short_memory: bool,
        include_long_memory: bool,
    ) -> Optional[torch.Tensor]:
        losses = []
        if include_short_memory:
            short_loss = self._alignment_loss_from_memory(
                "short",
                self.memory.short_memory,
                final_text_prototypes,
                max_classes=self.args.max_short_align_classes,
            )
            if short_loss is not None:
                losses.append(short_loss)

        if include_long_memory:
            long_loss = self._alignment_loss_from_memory(
                "long",
                self.memory.long_memory,
                final_text_prototypes,
                max_classes=self.args.max_long_align_classes,
            )
            if long_loss is not None:
                losses.append(long_loss)

        if not losses:
            return None
        return sum(losses) / len(losses)

    def step(self, images, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        self.stream_step += 1
        input_images = unwrap_stream_batch(images, self.device)
        prompt_state = self.current_prompt_state()

        image_features = self.encode_image(input_images)
        _, fused_logits, cache_info = self.fused_logits(image_features, prompt_state)
        standard_entropy = entropy_from_logits(fused_logits)
        per_bin_pred, majority_pred, vote_count, stability, reliability = self.neighbor_reliability(
            image_features,
            prompt_state,
        )
        vote_fraction = vote_count.float() / max(1, per_bin_pred.size(1))
        update_mask, gate_info = self.should_update(fused_logits, majority_pred, vote_fraction)

        target = target.to(self.device)
        pre_pred = fused_logits.argmax(dim=-1)
        pre_acc = (pre_pred == target).float().mean().item() * 100.0

        reweighted_entropy = (1.0 + torch.log((reliability * stability).clamp_min(1e-6))) * standard_entropy
        if self.args.entropy_loss_mode == "reweighted":
            weights = update_mask.float()
            entropy_loss = reweighted_entropy.mean() if weights.sum() <= 0 else (reweighted_entropy * weights).sum() / weights.sum()
        else:
            entropy_loss = average_entropy(fused_logits, update_mask)

        pseudo_loss = None
        pseudo_active = 0.0
        pseudo_disagree = 0.0
        if self.args.pseudo_label_weight > 0:
            pseudo_mask = update_mask & (vote_fraction >= self.args.pseudo_min_vote)
            if pseudo_mask.any():
                per_sample_pseudo = F.cross_entropy(
                    fused_logits,
                    majority_pred.detach(),
                    reduction="none",
                )
                pseudo_weights = vote_fraction * pseudo_mask.float()
                if pseudo_weights.sum() > 0:
                    pseudo_loss = (per_sample_pseudo * pseudo_weights).sum() / pseudo_weights.sum()
                    pseudo_active = float(pseudo_mask.float().mean().item())
                    pseudo_disagree = float(((majority_pred != pre_pred) & pseudo_mask).float().mean().item())

        plastic_loss = entropy_loss if pseudo_loss is None else entropy_loss + self.args.pseudo_label_weight * pseudo_loss
        perform_update = bool(update_mask.any().item())
        current_grad = self.compute_consistency_grad(loss=plastic_loss, retain_graph=True) if perform_update else None

        for idx in range(input_images.size(0)):
            self.memory.push_short(
                MemoryEntry(
                    feature=image_features[idx].detach().cpu(),
                    pseudo_label=int(majority_pred[idx].item()),
                    reweighted_entropy=float(reweighted_entropy[idx].item()),
                    image=input_images[idx].detach().cpu(),
                    reliability=float(reliability[idx].item()),
                    stability=float(stability[idx].item()),
                    grad_vector=None if current_grad is None else current_grad.detach().cpu(),
                ),
                step=self.stream_step,
            )

        if perform_update and self.stream_step % self.args.long_update_freq == 0 and current_grad is not None:
            long_info = self.promote_to_long_memory(current_grad)
        else:
            long_info = {"promoted": 0.0, "selected_classes": 0.0}

        include_short_memory = self.stream_step % self.args.short_align_freq == 0
        include_long_memory = self.stream_step % self.args.long_update_freq == 0
        if perform_update:
            align_loss = self.alignment_loss(
                self.alignment_targets(prompt_state),
                include_short_memory=include_short_memory,
                include_long_memory=include_long_memory,
            )
            scaled_align_loss = None if align_loss is None else self.args.lambda_align * align_loss
            opt_info = self.optimizer.step(
                plastic_loss=None,
                stability_loss=scaled_align_loss,
                precomputed_plastic_grad=current_grad,
            )
        else:
            align_loss = None
            opt_info = {"subspace_rank": 0.0, "plastic_norm": 0.0, "stability_norm": 0.0}

        with torch.no_grad():
            post_prompt_state = self.current_prompt_state()
            post_features = self.encode_image(input_images)
            _, eval_logits, post_cache_info = self.fused_logits(post_features, post_prompt_state)
            pred = eval_logits.argmax(dim=-1)
            acc = (pred == target).float().mean().item() * 100.0

        metrics = {
            "pre_acc": pre_acc,
            "post_acc": acc,
            "acc_gain": acc - pre_acc,
            "short_align_active": float(include_short_memory),
            "long_align_active": float(include_long_memory),
            "entropy_loss": float(entropy_loss.item()),
            "plastic_loss": float(plastic_loss.item()),
            "pseudo_loss": 0.0 if pseudo_loss is None else float(pseudo_loss.item()),
            "pseudo_active": pseudo_active,
            "pseudo_disagree": pseudo_disagree,
            "perform_update": float(perform_update),
            "align_loss": 0.0 if align_loss is None else float(align_loss.item()),
            "mean_reweighted_entropy": float(reweighted_entropy.mean().item()),
            "mean_reliability": float(reliability.mean().item()),
            "mean_stability": float(stability.mean().item()),
            "mean_vote_fraction": float(vote_fraction.mean().item()),
            "text_residual_norm": float(self.text_adapter.residual.norm().item()),
            "acc": acc,
        }
        metrics.update(cache_info)
        metrics.update(
            {
                "post_short_cache_active_classes": post_cache_info["short_cache_active_classes"],
                "post_long_cache_active_classes": post_cache_info["long_cache_active_classes"],
                "post_short_cache_norm": post_cache_info["short_cache_norm"],
                "post_long_cache_norm": post_cache_info["long_cache_norm"],
            }
        )
        metrics.update(gate_info)
        metrics.update(long_info)
        metrics.update(opt_info)
        return eval_logits, metrics
