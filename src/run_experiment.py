"""CLI entry point for experiment stages."""

from __future__ import annotations

import argparse

try:
    from .translate_nllb import translate_file
except ImportError:  # Allows `python src/run_experiment.py` during local debugging.
    from translate_nllb import translate_file


SUPPORTED_MODES = ("zero_shot", "java_only", "joint_balanced")
SUPPORTED_STAGES = ("train", "translate", "evaluate", "all")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run a planned NLLB experiment stage.")
    parser.add_argument("--mode", choices=SUPPORTED_MODES, required=True)
    parser.add_argument("--stage", choices=SUPPORTED_STAGES, required=True)
    parser.add_argument("--config", dest="config_path", required=True)
    return parser.parse_args()


def _load_config(config_path: str) -> dict:
    import yaml

    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _run_zero_shot_translate(config: dict) -> None:
    result = translate_file(
        model_name=config["model_name"],
        input_file=config["test_file"],
        output_dir=config["output_dir"],
        target_lang=config["target_lang"],
        num_beams=int(config.get("num_beams", 5)),
        batch_size=int(config.get("batch_size", config.get("batch_size_eval", 8))),
    )
    print("Zero-shot translation complete:")
    print(f"  predictions: {result['predictions_path']}")
    print(f"  records: {result['record_count']}")
    print(f"  empty hypotheses: {result['empty_hypothesis_count']}")


def main() -> None:
    """Run supported experiment stages."""
    args = parse_args()
    config = _load_config(args.config_path)

    if config.get("mode") != args.mode:
        raise ValueError(
            f"Config mode {config.get('mode')!r} does not match CLI mode {args.mode!r}."
        )

    if args.mode == "zero_shot" and args.stage in {"translate", "all"}:
        _run_zero_shot_translate(config)
        if args.stage == "all":
            print("Evaluation metrics are not implemented yet; ran translation only.")
        return

    if args.stage == "train":
        print("Training is not implemented yet.")
        return
    if args.stage == "evaluate":
        print("Evaluation metrics are not implemented yet.")
        return

    print(f"Stage {args.stage!r} for mode {args.mode!r} is not implemented yet.")


if __name__ == "__main__":
    main()
