"""Create a consolidated results table for the experiment report.

TODO:
- Read metric files for zero_shot, java_only, and joint_balanced.
- Produce reports/results_table.md.
- Highlight the main java_only vs joint_balanced comparison.
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Parse placeholder command-line arguments."""
    parser = argparse.ArgumentParser(description="Make results table. Not implemented yet.")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--output", default="reports/results_table.md")
    return parser.parse_args()


def main() -> None:
    """Placeholder entry point."""
    args = parse_args()
    print("Results table creation is not implemented yet.")
    print(f"Parsed arguments: {args}")


if __name__ == "__main__":
    main()

