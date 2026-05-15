"""Fine-tune the NLLB model.

TODO:
- Load an experiment config.
- Tokenize JSONL translation examples with NLLB language codes.
- Fine-tune `facebook/nllb-200-distilled-600M`.
- Save checkpoints and training metadata under the configured output directory.
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Parse placeholder command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune NLLB. Not implemented yet.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    """Placeholder entry point."""
    args = parse_args()
    print("NLLB training is not implemented yet.")
    print(f"Parsed arguments: {args}")


if __name__ == "__main__":
    main()

