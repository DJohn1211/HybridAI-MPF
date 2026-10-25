"""Command-line entry point for the paper's FNO-ResUNet training run."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the FNO-ResUNet pressure-correction model with the "
            "hyperparameters reported in the paper and supplementary material."
        )
    )
    parser.add_argument(
        "--patch-root",
        type=Path,
        required=True,
        help="Dataset root containing Case1 ... Case13 directories with patch_t*.npz files.",
    )
    parser.add_argument(
        "--u-patch-root",
        type=Path,
        required=True,
        help="Aligned velocity-patch root required by the continuity loss.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for checkpoints, normalization statistics, and training logs.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count (default: 0; safest for NPZ datasets).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start a fresh run instead of resuming out-dir/ckpt_last.pt.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision (deviates from the reported training setup).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from trainer_timewindow_dp_rgh import TrainConfig, run_training

    cfg = TrainConfig(
        patch_root=args.patch_root,
        u_patch_root=args.u_patch_root,
        out_dir=args.out_dir,
        seed=args.seed,
        num_workers=args.num_workers,
        resume=not args.no_resume,
        amp=not args.no_amp,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()
