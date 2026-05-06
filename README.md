# MARK-TTA

Official paper-facing code organization for a continuous test-time adaptation method built on CLIP-style vision-language models.

This directory is designed for paper submission and reviewer inspection. The implementation is reorganized so that the core methodological contributions are easy to locate, while generic engineering utilities are moved into supporting modules.

## Overview

`paper_submission_code/` provides a compact and method-centric view of the project. Compared with the full research workspace, this package emphasizes:

- clear separation between method logic and auxiliary utilities
- explicit implementation of the main algorithmic components
- a lightweight entry point for evaluation and reproduction
- a reviewer-friendly code layout that maps directly to the paper

## Key Components

The submission package highlights four method-facing components:

1. Neighbor-aware semantic voting for reliable pseudo-label estimation.
2. Dual-memory adaptation with short-term and long-term cache fusion.
3. Disentangled principal / null-space projection for stable online updates.
4. Prototype alignment between memory-derived visual prototypes and text prototypes.

## Repository Layout

```text
paper_submission_code/
├── README.md
├── run_mark_tta.py
└── mark_tta/
    ├── __init__.py
    ├── compat.py
    ├── core.py
    ├── memory.py
    ├── projection.py
    ├── prompts.py
    └── utils.py
```

## File Guide

| Path | Description |
| --- | --- |
| `run_mark_tta.py` | Main evaluation entry point for the paper-facing code package. |
| `mark_tta/core.py` | Main algorithm loop and integration of all core method components. |
| `mark_tta/prompts.py` | Prompt bank construction, neighbor-aware prompt bins, and text prototype adaptation. |
| `mark_tta/memory.py` | Short-term / long-term memory buffers and cache prototype construction. |
| `mark_tta/projection.py` | Gradient history tracking and disentangled low-rank projection updates. |
| `mark_tta/utils.py` | Shared utilities such as seeding, normalization, and input batch handling. |
| `mark_tta/compat.py` | Runtime compatibility shims for optional dependencies. |

## Innovation-to-Code Mapping

| Paper Component | Primary Location |
| --- | --- |
| Neighbor-aware semantic voting | `mark_tta/prompts.py`, `mark_tta/core.py::neighbor_reliability` |
| Dual-memory cache fusion | `mark_tta/memory.py`, `mark_tta/core.py::memory_cache_logits` |
| Long-memory promotion | `mark_tta/core.py::promote_to_long_memory` |
| Principal / null-space projection | `mark_tta/projection.py` |
| Memory-to-text prototype alignment | `mark_tta/core.py::alignment_loss`, `mark_tta/prompts.py::TextPrototypeAdapter` |

## Requirements

This package is intended to be used inside the current repository and therefore reuses root-level assets such as:

- `clip/`
- `datasets/`
- `gpt3_prompts/`
- `utils.py`

The runtime environment should include the standard dependencies already used by the main project, including:

- Python 3.10+
- PyTorch
- torchvision
- numpy
- tqdm
- PyYAML

Additional optional dependencies are handled conservatively through compatibility shims when possible.

## Quick Start

Run a single dataset evaluation with:

```bash
python paper_submission_code/run_mark_tta.py --datasets oxford_flowers
```

Run multiple datasets with:

```bash
python paper_submission_code/run_mark_tta.py \
  --datasets oxford_flowers/ucf101/stanford_cars
```

Inspect all available command-line options with:

```bash
python paper_submission_code/run_mark_tta.py --help
```

## Main Arguments

Some commonly used arguments are listed below:

| Argument | Description |
| --- | --- |
| `--datasets` | Dataset name or slash-separated list of datasets. |
| `--data-root` | Root directory containing dataset assets. |
| `--backbone` | CLIP backbone to evaluate, such as `ViT-B/16`. |
| `--adapt-side` | Adaptation target: `text_only`, `image_only`, or `hybrid`. |
| `--short-capacity` | Capacity of the short-term memory bank. |
| `--long-capacity` | Capacity of the long-term memory bank. |
| `--projection-mode` | Projection mode: `none`, `principal_only`, `null_only`, or `full`. |
| `--lambda-align` | Weight for memory-to-text prototype alignment. |
| `--disable-neighbor-reliability` | Disable neighbor-aware voting and fall back to standard predictions. |

## Method Notes

- The code path in `mark_tta/core.py` is intentionally kept explicit so that the algorithmic flow matches the paper as closely as possible.
- Utility functions that do not represent novel method content are placed in `mark_tta/utils.py` or `mark_tta/compat.py`.
- The current submission package still depends on the parent repository for datasets, CLIP utilities, and prompt resources.

## Reproducibility

The evaluation script exposes the main hyperparameters through the command line. For reproducible runs:

1. keep the dataset layout consistent with the parent repository
2. use the same CLIP backbone specified in the experiment configuration
3. set `--seed` explicitly
4. record the exact command used for each benchmark

## Status

This directory is a paper-submission-oriented reorganization of the broader codebase. It is intended to make the contribution structure easier to inspect, rather than to replace the full experimental workspace.

If a stricter camera-ready release is needed later, this package can be further isolated from root-level dependencies in a follow-up cleanup pass.

## Citation

Citation information can be added here once the paper metadata is finalized.

```bibtex
@misc{mark_tta,
  title  = {MARK-TTA},
  note   = {Paper submission code package}
}
```
