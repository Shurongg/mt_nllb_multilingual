"""Generate translations with an NLLB model or checkpoint.

TODO:
- Load a config and model/checkpoint.
- Read source examples from the Javanese-English test JSONL file.
- Generate English translations with the configured beam size.
- Write predictions for later evaluation.
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Parse placeholder command-line arguments."""
    parser = argparse.ArgumentParser(description="Translate with NLLB. Not implemented yet.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    """Placeholder entry point."""
    args = parse_args()
    print("NLLB translation is not implemented yet.")
    print(f"Parsed arguments: {args}")


if __name__ == "__main__":
    main()

