"""Build balanced multilingual training data for the joint experiment.

TODO:
- Read Javanese-English train JSONL.
- Read Indonesian-English train JSONL.
- Balance the two training sources for `joint_balanced`.
- Write data/processed/joint/train_balanced.jsonl.
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Parse placeholder command-line arguments."""
    parser = argparse.ArgumentParser(description="Build balanced joint training data.")
    parser.add_argument("--jav-train", default="data/processed/jav_eng/train.jsonl")
    parser.add_argument("--ind-train", default="data/processed/ind_eng/train.jsonl")
    parser.add_argument("--output", default="data/processed/joint/train_balanced.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Placeholder entry point."""
    args = parse_args()
    print("Joint data construction is not implemented yet.")
    print(f"Parsed arguments: {args}")


if __name__ == "__main__":
    main()

