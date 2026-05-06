import json
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .compat import ensure_runtime_compat
from .utils import safe_normalize

ensure_runtime_compat()

import clip


@dataclass
class PromptState:
    prompt_embeddings: torch.Tensor
    standard_prototypes: torch.Tensor
    bin_prototypes: torch.Tensor
    final_bin_prototypes: torch.Tensor


class TextPrototypeAdapter(nn.Module):
    def __init__(self, num_classes: int, feature_dim: int, adapt_bins: bool = True) -> None:
        super().__init__()
        self.adapt_bins = adapt_bins
        self.residual = nn.Parameter(torch.zeros(num_classes, feature_dim))

    def forward(self, base_state: PromptState) -> PromptState:
        standard_prototypes = safe_normalize(base_state.standard_prototypes + self.residual, dim=-1)
        if self.adapt_bins:
            bin_prototypes = safe_normalize(base_state.bin_prototypes + self.residual.unsqueeze(1), dim=-1)
        else:
            bin_prototypes = base_state.bin_prototypes

        return PromptState(
            prompt_embeddings=base_state.prompt_embeddings,
            standard_prototypes=standard_prototypes,
            bin_prototypes=bin_prototypes,
            final_bin_prototypes=bin_prototypes[:, -1],
        )


class NeighborAwarePromptBank:
    def __init__(
        self,
        classnames: Sequence[str],
        template: Sequence[str],
        cupl_path: str,
        prompts_per_class: int,
        num_bins: int,
        prompt_source: str,
    ) -> None:
        self.classnames = list(classnames)
        self.template = list(template)
        self.prompts_per_class = prompts_per_class
        self.num_bins = num_bins
        self.num_classes = len(self.classnames)
        self.prompt_source = prompt_source

        with open(cupl_path, "r", encoding="utf-8") as f:
            cupl_prompts = json.load(f)

        token_bank = []
        for classname in self.classnames:
            prompt_strings = self._build_prompt_strings(classname, cupl_prompts)
            token_bank.append(clip.tokenize(prompt_strings, truncate=True))

        self.tokenized_prompts = torch.stack(token_bank, dim=0)
        prefix_sizes = np.ceil(
            np.arange(1, self.num_bins + 1, dtype=np.float32) * self.prompts_per_class / self.num_bins
        ).astype(int)
        self.prefix_sizes = [int(np.clip(v, 1, self.prompts_per_class)) for v in prefix_sizes]

    def _build_prompt_strings(self, classname: str, cupl_prompts: Dict[str, List[str]]) -> List[str]:
        display_name = classname.replace("_", " ")
        template_prompts = [prompt.format(display_name) for prompt in self.template]
        cupl_only_prompts = list(cupl_prompts.get(display_name, cupl_prompts.get(classname, [])))

        if self.prompt_source == "hybrid":
            prompts = template_prompts + cupl_only_prompts
        elif self.prompt_source == "cupl_first":
            prompts = cupl_only_prompts + template_prompts
        elif self.prompt_source == "cupl_only":
            prompts = cupl_only_prompts
        elif self.prompt_source == "template_only":
            prompts = template_prompts
        else:
            raise ValueError(f"Unknown prompt source mode: {self.prompt_source}")

        if not prompts:
            prompts = [f"a photo of a {display_name}."]

        deduped = []
        seen = set()
        for prompt in prompts:
            prompt = prompt.strip()
            if prompt and prompt not in seen:
                deduped.append(prompt)
                seen.add(prompt)

        if len(deduped) >= self.prompts_per_class:
            return deduped[: self.prompts_per_class]

        expanded = []
        idx = 0
        while len(expanded) < self.prompts_per_class:
            expanded.append(deduped[idx % len(deduped)])
            idx += 1
        return expanded

    def encode(self, model: nn.Module, device: torch.device) -> PromptState:
        tokens = self.tokenized_prompts.to(device)
        flat_tokens = tokens.view(-1, tokens.size(-1))

        with torch.no_grad():
            text_embeddings = model.encode_text(flat_tokens)
            text_embeddings = safe_normalize(text_embeddings.float(), dim=-1)
            text_embeddings = text_embeddings.view(self.num_classes, self.prompts_per_class, -1)

        intra_sim = torch.einsum("ckd,cjd->ckj", text_embeddings, text_embeddings)
        reliability_scores = intra_sim.sum(dim=-1) - 1.0
        sorted_idx = reliability_scores.argsort(dim=-1, descending=False)
        sorted_embeddings = torch.gather(
            text_embeddings,
            1,
            sorted_idx.unsqueeze(-1).expand(-1, -1, text_embeddings.size(-1)),
        )

        bin_prototypes = []
        for prefix in self.prefix_sizes:
            prototype = sorted_embeddings[:, :prefix].mean(dim=1)
            bin_prototypes.append(safe_normalize(prototype, dim=-1))
        bin_prototypes = torch.stack(bin_prototypes, dim=1)

        standard_prototypes = safe_normalize(text_embeddings.mean(dim=1), dim=-1)
        return PromptState(
            prompt_embeddings=text_embeddings,
            standard_prototypes=standard_prototypes,
            bin_prototypes=bin_prototypes,
            final_bin_prototypes=bin_prototypes[:, -1],
        )
