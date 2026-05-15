"""Prepare raw parallel data into the project JSONL schema.

This phase only converts the fixed raw train/dev/test files into processed
JSONL and writes data statistics. It does not build joint data or run models.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .io_utils import read_text_lines, write_json, write_jsonl
except ImportError:  # Allows `python src/prepare_data.py` during local debugging.
    from io_utils import read_text_lines, write_json, write_jsonl


ENG_LANG = "eng_Latn"


DATASETS = (
    {
        "dataset": "jav_eng",
        "split": "train",
        "id_prefix": "jav_train",
        "src_lang": "jav_Latn",
        "src_file": "train.jv",
        "tgt_file": "train.en",
        "output_relpath": Path("jav_eng") / "train.jsonl",
    },
    {
        "dataset": "jav_eng",
        "split": "dev",
        "id_prefix": "jav_dev",
        "src_lang": "jav_Latn",
        "src_file": "valid.jv",
        "tgt_file": "valid.en",
        "output_relpath": Path("jav_eng") / "dev.jsonl",
    },
    {
        "dataset": "jav_eng",
        "split": "test",
        "id_prefix": "jav_test",
        "src_lang": "jav_Latn",
        "src_file": "test.jv",
        "tgt_file": "test.en",
        "output_relpath": Path("jav_eng") / "test.jsonl",
    },
    {
        "dataset": "ind_eng",
        "split": "train",
        "id_prefix": "ind_train",
        "src_lang": "ind_Latn",
        "src_file": "train.ind",
        "tgt_file": "train.eng",
        "output_relpath": Path("ind_eng") / "train.jsonl",
    },
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Prepare processed JSONL data.")
    parser.add_argument("--jav_raw_dir", default="data/raw/jav_eng")
    parser.add_argument("--ind_raw_dir", default="data/raw/ind_eng")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--report_dir", default="data/reports")
    return parser.parse_args()


def _raw_dir_for(dataset: str, jav_raw_dir: Path, ind_raw_dir: Path) -> Path:
    if dataset == "jav_eng":
        return jav_raw_dir
    if dataset == "ind_eng":
        return ind_raw_dir
    raise ValueError(f"Unsupported dataset: {dataset}")


def _build_records(
    src_lines: list[str],
    tgt_lines: list[str],
    *,
    dataset: str,
    split: str,
    id_prefix: str,
    src_lang: str,
) -> tuple[list[dict], int, int]:
    records: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()
    removed_empty = 0
    removed_duplicate = 0

    for src_raw, tgt_raw in zip(src_lines, tgt_lines):
        src_text = src_raw.strip()
        tgt_text = tgt_raw.strip()

        if not src_text or not tgt_text:
            removed_empty += 1
            continue

        pair = (src_text, tgt_text)
        if pair in seen_pairs:
            removed_duplicate += 1
            continue
        seen_pairs.add(pair)

        record_id = f"{id_prefix}_{len(records) + 1:06d}"
        records.append(
            {
                "id": record_id,
                "src_lang": src_lang,
                "tgt_lang": ENG_LANG,
                "src_text": src_text,
                "tgt_text": tgt_text,
                "dataset": dataset,
                "split": split,
            }
        )

    return records, removed_empty, removed_duplicate


def _process_split(spec: dict, raw_dir: Path, output_dir: Path) -> dict:
    src_path = raw_dir / spec["src_file"]
    tgt_path = raw_dir / spec["tgt_file"]

    src_lines = read_text_lines(src_path)
    tgt_lines = read_text_lines(tgt_path)

    if len(src_lines) != len(tgt_lines):
        raise ValueError(
            "Mismatched line counts for "
            f"{spec['dataset']} {spec['split']}: "
            f"{src_path} has {len(src_lines)} lines, "
            f"{tgt_path} has {len(tgt_lines)} lines."
        )

    records, removed_empty, removed_duplicate = _build_records(
        src_lines,
        tgt_lines,
        dataset=spec["dataset"],
        split=spec["split"],
        id_prefix=spec["id_prefix"],
        src_lang=spec["src_lang"],
    )

    output_path = output_dir / spec["output_relpath"]
    write_jsonl(records, output_path)

    return {
        "dataset": spec["dataset"],
        "split": spec["split"],
        "source_file": str(src_path),
        "target_file": str(tgt_path),
        "output_file": str(output_path),
        "input_source_lines": len(src_lines),
        "input_target_lines": len(tgt_lines),
        "output_records": len(records),
        "removed_empty_count": removed_empty,
        "removed_duplicate_count": removed_duplicate,
        "src_lang": spec["src_lang"],
        "tgt_lang": ENG_LANG,
    }


def _write_markdown_report(stats: dict, path: Path) -> None:
    lines = [
        "# Data Preparation Stats",
        "",
        "| Dataset | Split | Source Lang | Target Lang | Input Src Lines | Input Tgt Lines | Output Records | Removed Empty | Removed Duplicates | Output File |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for item in stats["splits"]:
        lines.append(
            "| {dataset} | {split} | {src_lang} | {tgt_lang} | "
            "{input_source_lines} | {input_target_lines} | {output_records} | "
            "{removed_empty_count} | {removed_duplicate_count} | `{output_file}` |".format(
                **item
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Prepare all Phase 1 processed data files and reports."""
    args = parse_args()
    jav_raw_dir = Path(args.jav_raw_dir)
    ind_raw_dir = Path(args.ind_raw_dir)
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)

    split_stats = []
    for spec in DATASETS:
        raw_dir = _raw_dir_for(spec["dataset"], jav_raw_dir, ind_raw_dir)
        split_stats.append(_process_split(spec, raw_dir, output_dir))

    stats = {"splits": split_stats}
    write_json(stats, report_dir / "data_stats.json")
    _write_markdown_report(stats, report_dir / "data_stats.md")

    print("Prepared Phase 1 data files:")
    for item in split_stats:
        print(f"  {item['output_file']}: {item['output_records']} records")
    print(f"Wrote reports under {report_dir}")


if __name__ == "__main__":
    main()
