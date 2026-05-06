# CURE-TTA

## Repository Layout

```text
CURE/
├── README.md
├── run_cure_tta.py
└── cure_tta/
    ├── __init__.py
    ├── compat.py
    ├── core.py
    ├── memory.py
    ├── projection.py
    ├── prompts.py
    └── utils.py
```

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


