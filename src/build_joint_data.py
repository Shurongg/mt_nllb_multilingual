"""Build balanced multilingual training data for the joint experiment.

The joint training file uses all Javanese-English train records and a seeded
random sample of Indonesian-English train records. It does not use Javanese dev
or test data, and it does not create Indonesian dev/test files.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

try:
    from .io_utils import read_jsonl, write_jsonl
except ImportError:  # Allows `python src/build_joint_data.py` during local debugging.
    from io_utils import read_jsonl, write_jsonl


REQUIRED_FIELDS = {
    "id",
    "src_lang",
    "tgt_lang",
    "src_text",
    "tgt_text",
    "dataset",
    "split",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build balanced joint training data.")
    parser.add_argument("--jav_train", default="data/processed/jav_eng/train.jsonl")
    parser.add_argument("--ind_train", default="data/processed/ind_eng/train.jsonl")
    parser.add_argument("--output_file", default="data/processed/joint/train_balanced.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratio", type=float, default=1.0)
    return parser.parse_args()


def _load_existing_jsonl(path: str | Path) -> list[dict]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Required input file does not exist: {input_path}")
    return read_jsonl(input_path)


def _validate_records(
    records: list[dict],
    *,
    path: str | Path,
    expected_dataset: str,
    expected_src_lang: str,
) -> None:
    for index, record in enumerate(records, start=1):
        missing = REQUIRED_FIELDS - set(record)
        if missing:
            raise ValueError(
                f"{path} record {index} is missing required fields: {sorted(missing)}"
            )

        if record["dataset"] != expected_dataset:
            raise ValueError(
                f"{path} record {index} has dataset={record['dataset']!r}; "
                f"expected {expected_dataset!r}."
            )
        if record["src_lang"] != expected_src_lang:
            raise ValueError(
                f"{path} record {index} has src_lang={record['src_lang']!r}; "
                f"expected {expected_src_lang!r}."
            )
        if record["split"] != "train":
            raise ValueError(
                f"{path} record {index} has split={record['split']!r}; "
                "expected 'train'."
            )
        if record["tgt_lang"] != "eng_Latn":
            raise ValueError(
                f"{path} record {index} has tgt_lang={record['tgt_lang']!r}; "
                "expected 'eng_Latn'."
            )


def build_joint_records(
    jav_records: list[dict],
    ind_records: list[dict],
    *,
    seed: int,
    ratio: float,
) -> tuple[list[dict], int]:
    """Sample Indonesian records and return the shuffled joint records."""
    if ratio < 0:
        raise ValueError(f"--ratio must be non-negative, got {ratio}.")

    sample_size = round(len(jav_records) * ratio)
    if sample_size > len(ind_records):
        raise ValueError(
            "Requested Indonesian sample size is larger than available records: "
            f"requested {sample_size}, available {len(ind_records)}."
        )

    rng = random.Random(seed)
    sampled_ind_records = rng.sample(ind_records, sample_size)
    joint_records = list(jav_records) + sampled_ind_records
    rng.shuffle(joint_records)
    return joint_records, sample_size


def main() -> None:
    """Build and write the balanced joint training file."""
    args = parse_args()

    jav_records = _load_existing_jsonl(args.jav_train)
    ind_records = _load_existing_jsonl(args.ind_train)

    _validate_records(
        jav_records,
        path=args.jav_train,
        expected_dataset="jav_eng",
        expected_src_lang="jav_Latn",
    )
    _validate_records(
        ind_records,
        path=args.ind_train,
        expected_dataset="ind_eng",
        expected_src_lang="ind_Latn",
    )

    joint_records, ind_sample_size = build_joint_records(
        jav_records,
        ind_records,
        seed=args.seed,
        ratio=args.ratio,
    )
    write_jsonl(joint_records, args.output_file)

    print("Built balanced joint training data:")
    print(f"  Javanese examples used: {len(jav_records)}")
    print(f"  Indonesian examples sampled: {ind_sample_size}")
    print(f"  Total output records: {len(joint_records)}")
    print(f"  Output path: {args.output_file}")
    print(f"  Random seed: {args.seed}")
    print(f"  Ratio: {args.ratio}")


if __name__ == "__main__":
    main()
