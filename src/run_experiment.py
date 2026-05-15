"""CLI entry point for planned experiment stages.

This file intentionally contains only a scaffold. Training, translation, and
evaluation implementations will be added in later project steps.
"""

from __future__ import annotations

import argparse


SUPPORTED_MODES = ("zero_shot", "java_only", "joint_balanced")
SUPPORTED_STAGES = ("train", "translate", "evaluate", "all")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run a planned NLLB experiment stage.")
    parser.add_argument("--mode", choices=SUPPORTED_MODES, required=True)
    parser.add_argument("--stage", choices=SUPPORTED_STAGES, required=True)
    parser.add_argument("--config", dest="config_path", required=True)
    return parser.parse_args()


def main() -> None:
    """Print the parsed arguments and planned action."""
    args = parse_args()
    print("Parsed arguments:")
    print(f"  mode: {args.mode}")
    print(f"  stage: {args.stage}")
    print(f"  config: {args.config_path}")
    print()
    print("Planned action only. Full experiment execution is not implemented yet.")


if __name__ == "__main__":
    main()

