"""Evaluate machine translation outputs.

TODO:
- Read generated predictions and Javanese-English test references.
- Compute BLEU and chrF++ with sacrebleu.
- Optionally add COMET later without making it a hard dependency.
- Write evaluation reports under outputs/ and reports/.
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Parse placeholder command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate MT outputs. Not implemented yet.")
    parser.add_argument("--predictions")
    parser.add_argument("--references", default="data/processed/jav_eng/test.jsonl")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    """Placeholder entry point."""
    args = parse_args()
    print("MT evaluation is not implemented yet.")
    print(f"Parsed arguments: {args}")


if __name__ == "__main__":
    main()

