from __future__ import annotations

import argparse
import logging
from pathlib import Path

from vpc2.data.processing import DEFAULT_RATIOS, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit, clean, and split the Roboflow garbage-classification-3 YOLO dataset. "
            "Writes manifests to data/interim/ and the training layout to data/processed/."
        )
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("data/raw"),
        help="Raw dataset root (auto-discovers the Roboflow export).",
    )
    parser.add_argument("--interim", type=Path, default=Path("data/interim"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_RATIOS[0])
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_RATIOS[1])
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_RATIOS[2])
    parser.add_argument(
        "--drop-oob",
        action="store_true",
        help="Drop out-of-bounds boxes instead of clipping them to [0, 1].",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_pipeline(
        raw_dir=args.raw,
        interim_dir=args.interim,
        processed_dir=args.processed,
        seed=args.seed,
        ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
        clip_out_of_bounds=not args.drop_oob,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
