"""Evaluate machine translation outputs with local sacrebleu metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .io_utils import read_text_lines, write_json
except ImportError:  # Allows `python src/evaluate_mt.py` during local debugging.
    from io_utils import read_text_lines, write_json


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate MT outputs with BLEU and chrF++.")
    parser.add_argument("--hypotheses", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--sources")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--mode", required=True)
    return parser.parse_args()


def _metric_signature(metric) -> str:
    """Return a sacrebleu metric signature as a string."""
    try:
        return str(metric.get_signature())
    except Exception:
        return ""


def evaluate_files(
    *,
    hypotheses: str | Path,
    references: str | Path,
    output_json: str | Path,
    mode: str,
    sources: str | Path | None = None,
) -> dict:
    """Compute BLEU and chrF++ for plain-text hypothesis/reference files."""
    try:
        from sacrebleu.metrics import BLEU, CHRF
    except ImportError as exc:
        raise ImportError(
            "Evaluation requires sacrebleu. Install dependencies from requirements.txt."
        ) from exc

    hypothesis_lines = read_text_lines(hypotheses)
    reference_lines = read_text_lines(references)
    source_lines = read_text_lines(sources) if sources else None

    if len(hypothesis_lines) != len(reference_lines):
        raise ValueError(
            "Hypothesis/reference line count mismatch: "
            f"{hypotheses} has {len(hypothesis_lines)} lines, "
            f"{references} has {len(reference_lines)} lines."
        )

    if source_lines is not None and len(source_lines) != len(hypothesis_lines):
        raise ValueError(
            "Source/hypothesis line count mismatch: "
            f"{sources} has {len(source_lines)} lines, "
            f"{hypotheses} has {len(hypothesis_lines)} lines."
        )

    bleu_metric = BLEU()
    chrf_metric = CHRF(word_order=2)
    bleu_score = bleu_metric.corpus_score(hypothesis_lines, [reference_lines])
    chrf_score = chrf_metric.corpus_score(hypothesis_lines, [reference_lines])

    result = {
        "mode": mode,
        "num_samples": len(hypothesis_lines),
        "bleu": bleu_score.score,
        "chrf": chrf_score.score,
        "comet": None,
        "metrics": {
            "bleu": {
                "score": bleu_score.score,
                "signature": _metric_signature(bleu_metric),
            },
            "chrf": {
                "score": chrf_score.score,
                "signature": _metric_signature(chrf_metric),
            },
        },
        "files": {
            "hypotheses": str(hypotheses),
            "references": str(references),
            "sources": str(sources) if sources else None,
        },
    }
    write_json(result, output_json)
    return result


def main() -> None:
    """Run local MT evaluation."""
    args = parse_args()
    result = evaluate_files(
        hypotheses=args.hypotheses,
        references=args.references,
        sources=args.sources,
        output_json=args.output_json,
        mode=args.mode,
    )
    print("Wrote evaluation metrics:")
    print(f"  output: {args.output_json}")
    print(f"  samples: {result['num_samples']}")
    print(f"  BLEU: {result['bleu']:.4f}")
    print(f"  chrF++: {result['chrf']:.4f}")
    print("  COMET: not computed")


if __name__ == "__main__":
    main()
