"""Generate translations with an NLLB model or checkpoint."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

try:
    from .io_utils import ensure_dir, read_jsonl, write_jsonl, write_text_lines
    from .nllb_utils import ENGLISH, get_forced_bos_token_id
except ImportError:  # Allows `python src/translate_nllb.py` during local debugging.
    from io_utils import ensure_dir, read_jsonl, write_jsonl, write_text_lines
    from nllb_utils import ENGLISH, get_forced_bos_token_id


REQUIRED_INPUT_FIELDS = {
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
    parser = argparse.ArgumentParser(description="Translate JSONL records with NLLB.")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_lang", default=ENGLISH)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_source_length", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    return parser.parse_args()


def _validate_input_records(records: list[dict], input_file: str | Path) -> None:
    if not records:
        raise ValueError(f"Input file has no records: {input_file}")

    for index, record in enumerate(records, start=1):
        missing = REQUIRED_INPUT_FIELDS - set(record)
        if missing:
            raise ValueError(
                f"{input_file} record {index} is missing required fields: "
                f"{sorted(missing)}"
            )


def _load_model_and_tokenizer(model_name: str):
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Zero-shot translation requires torch and transformers. "
            "Install dependencies from requirements.txt first."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, tokenizer, device, torch


def _batched(items: list[int], batch_size: int) -> list[list[int]]:
    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]


def translate_records(
    records: list[dict],
    *,
    model_name: str,
    target_lang: str,
    num_beams: int,
    batch_size: int,
    max_source_length: int = 128,
    max_new_tokens: int = 128,
) -> list[str]:
    """Translate records with NLLB, batching separately by source language."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    model, tokenizer, device, torch = _load_model_and_tokenizer(model_name)
    forced_bos_token_id = get_forced_bos_token_id(tokenizer, target_lang)

    indices_by_src_lang: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        indices_by_src_lang[record["src_lang"]].append(index)

    hypotheses = [""] * len(records)
    with torch.no_grad():
        for src_lang, indices in indices_by_src_lang.items():
            tokenizer.src_lang = src_lang
            for batch_indices in _batched(indices, batch_size):
                batch_sources = [records[index]["src_text"] for index in batch_indices]
                encoded = tokenizer(
                    batch_sources,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_source_length,
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                generated = model.generate(
                    **encoded,
                    forced_bos_token_id=forced_bos_token_id,
                    num_beams=num_beams,
                    max_new_tokens=max_new_tokens,
                )
                decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
                for record_index, hypothesis in zip(batch_indices, decoded):
                    hypotheses[record_index] = hypothesis

    return hypotheses


def write_translation_outputs(
    records: list[dict],
    hypotheses: list[str],
    *,
    output_dir: str | Path,
    target_lang: str,
) -> dict[str, Path | int]:
    """Write sources, references, hypotheses, and prediction JSONL files."""
    if len(hypotheses) != len(records):
        raise ValueError(
            f"Hypothesis count {len(hypotheses)} does not match input count {len(records)}."
        )

    output_path = ensure_dir(output_dir)
    sources = [record["src_text"] for record in records]
    references = [record["tgt_text"] for record in records]
    predictions = [
        {
            "id": record["id"],
            "src_lang": record["src_lang"],
            "tgt_lang": target_lang,
            "source": record["src_text"],
            "reference": record["tgt_text"],
            "hypothesis": hypothesis,
        }
        for record, hypothesis in zip(records, hypotheses)
    ]

    sources_path = output_path / "sources.txt"
    references_path = output_path / "references.txt"
    hypotheses_path = output_path / "hypotheses.txt"
    predictions_path = output_path / "predictions.jsonl"

    write_text_lines(sources, sources_path)
    write_text_lines(references, references_path)
    write_text_lines(hypotheses, hypotheses_path)
    write_jsonl(predictions, predictions_path)

    expected_count = len(records)
    for path in (sources_path, references_path, hypotheses_path, predictions_path):
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        if line_count != expected_count:
            raise ValueError(
                f"{path} has {line_count} lines; expected {expected_count}."
            )

    empty_hypothesis_count = sum(1 for hypothesis in hypotheses if not hypothesis.strip())
    return {
        "sources_path": sources_path,
        "references_path": references_path,
        "hypotheses_path": hypotheses_path,
        "predictions_path": predictions_path,
        "record_count": expected_count,
        "empty_hypothesis_count": empty_hypothesis_count,
    }


def translate_file(
    *,
    model_name: str,
    input_file: str | Path,
    output_dir: str | Path,
    target_lang: str,
    num_beams: int,
    batch_size: int,
    max_source_length: int = 128,
    max_new_tokens: int = 128,
) -> dict[str, Path | int]:
    """Translate an input JSONL file and write standard output artifacts."""
    records = read_jsonl(input_file)
    _validate_input_records(records, input_file)
    hypotheses = translate_records(
        records,
        model_name=model_name,
        target_lang=target_lang,
        num_beams=num_beams,
        batch_size=batch_size,
        max_source_length=max_source_length,
        max_new_tokens=max_new_tokens,
    )
    return write_translation_outputs(
        records,
        hypotheses,
        output_dir=output_dir,
        target_lang=target_lang,
    )


def main() -> None:
    """Run NLLB translation from the command line."""
    args = parse_args()
    result = translate_file(
        model_name=args.model_name,
        input_file=args.input_file,
        output_dir=args.output_dir,
        target_lang=args.target_lang,
        num_beams=args.num_beams,
        batch_size=args.batch_size,
        max_source_length=args.max_source_length,
        max_new_tokens=args.max_new_tokens,
    )

    print("Wrote translation outputs:")
    print(f"  sources: {result['sources_path']}")
    print(f"  references: {result['references_path']}")
    print(f"  hypotheses: {result['hypotheses_path']}")
    print(f"  predictions: {result['predictions_path']}")
    print(f"  records: {result['record_count']}")
    print(f"  empty hypotheses: {result['empty_hypothesis_count']}")


if __name__ == "__main__":
    main()
