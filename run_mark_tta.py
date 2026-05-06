import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cure_tta import MarkTTAEngine, set_seed
from cure_tta.compat import ensure_runtime_compat

ensure_runtime_compat()

import clip

from utils import build_test_data_loader


def get_arguments():
    parser = argparse.ArgumentParser(description="Paper submission entry for MARK-style CTTA.")
    parser.add_argument("--datasets", type=str, default="oxford_flowers")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--backbone", type=str, default="ViT-B/16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adapt-side", type=str, choices=["text_only", "hybrid", "image_only"], default="hybrid")
    parser.set_defaults(no_text_adapt_bins=False)
    parser.add_argument("--no-text-adapt-bins", dest="no_text_adapt_bins", action="store_true")
    parser.add_argument("--prompts-per-class", type=int, default=16)
    parser.add_argument("--prompt-source", type=str, choices=["hybrid", "cupl_first", "cupl_only", "template_only"], default="hybrid")
    parser.add_argument("--num-bins", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--disable-neighbor-reliability", action="store_true")
    parser.add_argument("--entropy-loss-mode", type=str, choices=["standard", "reweighted"], default="reweighted")
    parser.add_argument("--short-capacity", type=int, default=8)
    parser.add_argument("--short-align-freq", type=int, default=2)
    parser.add_argument("--max-short-align-classes", type=int, default=16)
    parser.add_argument("--long-capacity", type=int, default=32)
    parser.add_argument("--long-update-freq", type=int, default=16)
    parser.add_argument("--max-long-align-classes", type=int, default=4)
    parser.add_argument("--long-select-classes", type=int, default=1)
    parser.add_argument("--long-samples-per-class", type=int, default=1)
    parser.add_argument("--align-chunk-size", type=int, default=8)
    parser.add_argument("--lambda-align", type=float, default=0.5)
    parser.add_argument("--align-prototype-source", type=str, choices=["standard", "final_bin", "hybrid"], default="standard")
    parser.add_argument("--short-cache-alpha", type=float, default=2.5)
    parser.add_argument("--short-cache-beta", type=float, default=5.0)
    parser.add_argument("--long-cache-alpha", type=float, default=1.25)
    parser.add_argument("--long-cache-beta", type=float, default=5.0)
    parser.add_argument("--pseudo-label-weight", type=float, default=1.0)
    parser.add_argument("--pseudo-min-vote", type=float, default=0.75)
    parser.add_argument("--update-min-vote", type=float, default=0.75)
    parser.add_argument("--update-min-prob", type=float, default=0.45)
    parser.set_defaults(update_require_agreement=True)
    parser.add_argument("--no-update-require-agreement", dest="update_require_agreement", action="store_false")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-history-size", type=int, default=64)
    parser.add_argument("--low-rank-rank", type=int, default=8)
    parser.add_argument("--svd-update-freq", type=int, default=16)
    parser.add_argument("--plasticity-scale", type=float, default=1.0)
    parser.add_argument("--stability-scale", type=float, default=1.0)
    parser.add_argument("--projection-mode", type=str, choices=["none", "principal_only", "null_only", "full"], default="full")
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--log-interval", type=int, default=100)
    return parser.parse_args()


def run_dataset(dataset_name: str, args) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load(args.backbone, device=device)
    model = model.eval()
    if device.type == "cpu":
        model = model.float()

    test_loader, classnames, template, cupl_path = build_test_data_loader(dataset_name, args.data_root, preprocess)
    engine = MarkTTAEngine(model, classnames, template, cupl_path, args)

    correct = 0
    total = 0
    pre_correct = 0
    metrics_window = []

    progress = tqdm(test_loader, desc=f"[MARK-TTA] {dataset_name}")
    for step_idx, (images, target) in enumerate(progress, start=1):
        if args.max_samples > 0 and step_idx > args.max_samples:
            break

        logits, metrics = engine.step(images, target)
        pred = logits.argmax(dim=-1)
        target = target.to(device)

        correct += int((pred == target).sum().item())
        pre_correct += int(round(metrics["pre_acc"] * target.numel() / 100.0))
        total += int(target.numel())
        metrics_window.append(metrics)

        if step_idx % args.log_interval == 0:
            recent = metrics_window[-args.log_interval :]
            progress.set_postfix(
                acc=f"{100.0 * correct / max(total, 1):.2f}",
                pre=f"{np.mean([m['pre_acc'] for m in recent]):.2f}",
                gain=f"{np.mean([m['acc_gain'] for m in recent]):.2f}",
                upd=f"{np.mean([m['perform_update'] for m in recent]):.2f}",
                vote=f"{np.mean([m['mean_vote_fraction'] for m in recent]):.2f}",
                s_cache=f"{np.mean([m['post_short_cache_norm'] for m in recent]):.2f}",
                l_cache=f"{np.mean([m['post_long_cache_norm'] for m in recent]):.2f}",
                proj=f"{np.mean([m['subspace_rank'] for m in recent]):.2f}",
            )

    final_acc = 100.0 * correct / max(total, 1)
    final_pre_acc = 100.0 * pre_correct / max(total, 1)
    print(f"[{dataset_name}] Final Accuracy: {final_acc:.2f}%")
    print(f"[{dataset_name}] Pre-Update Accuracy: {final_pre_acc:.2f}%")
    if metrics_window:
        print(f"[{dataset_name}] Mean Acc Gain per Step: {np.mean([m['acc_gain'] for m in metrics_window]):.2f}")
        print(
            f"[{dataset_name}] Short/Long Cache Norm: "
            f"{np.mean([m['post_short_cache_norm'] for m in metrics_window]):.2f}/"
            f"{np.mean([m['post_long_cache_norm'] for m in metrics_window]):.2f}"
        )
        print(f"[{dataset_name}] Mean Vote Fraction: {np.mean([m['mean_vote_fraction'] for m in metrics_window]):.2f}")
        print(f"[{dataset_name}] Mean Projection Rank: {np.mean([m['subspace_rank'] for m in metrics_window]):.2f}")
    return final_acc


def main():
    args = get_arguments()
    set_seed(args.seed)

    results = {}
    for dataset_name in args.datasets.split("/"):
        results[dataset_name] = run_dataset(dataset_name, args)

    print("\nSummary")
    for dataset_name, acc in results.items():
        print(f"{dataset_name}: {acc:.2f}%")


if __name__ == "__main__":
    main()
