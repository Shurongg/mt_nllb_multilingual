"""Fine-tune NLLB for the supported Javanese-English experiments."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any

try:
    from .io_utils import ensure_dir, read_jsonl, write_json
    from .nllb_utils import (
        ENGLISH,
        SUPPORTED_MODES,
        get_device_summary,
        get_forced_bos_token_id,
        resolve_fp16,
    )
except ImportError:  # Allows `python src/train_nllb.py` during local debugging.
    from io_utils import ensure_dir, read_jsonl, write_json
    from nllb_utils import (
        ENGLISH,
        SUPPORTED_MODES,
        get_device_summary,
        get_forced_bos_token_id,
        resolve_fp16,
    )


TRAINABLE_MODES = ("java_only", "joint_balanced")
REQUIRED_FIELDS = {
    "id",
    "src_lang",
    "tgt_lang",
    "src_text",
    "tgt_text",
    "dataset",
    "split",
}
DATASET_ALLOWED_KEYS = {"input_ids", "attention_mask", "labels"}
FINAL_MODEL_INPUT_KEYS = {"input_ids", "attention_mask", "labels"}
FINAL_MODEL_INPUT_KEY_ORDER = ["input_ids", "attention_mask", "labels"]
FORBIDDEN_MODEL_INPUT_KEYS = {
    "decoder_input_ids",
    "decoder_inputs_embeds",
    "inputs_embeds",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune NLLB with Seq2SeqTrainer.")
    parser.add_argument("--mode", choices=TRAINABLE_MODES, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_eval_samples", type=int)
    parser.add_argument("--num_train_epochs_override", type=float)
    parser.add_argument("--output_dir_override")
    return parser.parse_args()


def _load_config(config_path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _validate_records(records: list[dict], *, path: str | Path, expected_split: str) -> None:
    if not records:
        raise ValueError(f"No records found in {path}.")

    for index, record in enumerate(records, start=1):
        missing = REQUIRED_FIELDS - set(record)
        if missing:
            raise ValueError(f"{path} record {index} missing fields: {sorted(missing)}")
        if record["split"] != expected_split:
            raise ValueError(
                f"{path} record {index} has split={record['split']!r}; "
                f"expected {expected_split!r}."
            )
        if record["tgt_lang"] != ENGLISH:
            raise ValueError(
                f"{path} record {index} has tgt_lang={record['tgt_lang']!r}; "
                f"expected {ENGLISH!r}."
            )


def _slice_records(records: list[dict], max_samples: int | None) -> list[dict]:
    if max_samples is None:
        return records
    if max_samples <= 0:
        raise ValueError(f"Sample limit must be positive, got {max_samples}.")
    return records[:max_samples]


def _filter_tokenized_example(example: dict, *, context: str) -> dict:
    """Keep only the fields Seq2SeqTrainer should receive from the dataset."""
    filtered = {key: example[key] for key in DATASET_ALLOWED_KEYS if key in example}
    missing = DATASET_ALLOWED_KEYS - set(filtered)
    if missing:
        raise ValueError(f"{context} missing tokenized fields: {sorted(missing)}")

    extra = set(example) - DATASET_ALLOWED_KEYS
    if extra:
        # Tokenizers may return version-specific fields. Do not pass them to the model.
        return filtered
    return filtered


def _filter_model_inputs(inputs: dict) -> dict:
    """Remove every key except the final model input keys."""
    for key in list(inputs.keys()):
        if key not in FINAL_MODEL_INPUT_KEYS:
            inputs.pop(key, None)
    still_forbidden = FORBIDDEN_MODEL_INPUT_KEYS & set(inputs)
    if still_forbidden:
        raise ValueError(f"Forbidden model input keys remain: {sorted(still_forbidden)}")
    return inputs


class TranslationJsonlDataset:
    """Minimal dataset that tokenizes raw JSONL translation records on access."""

    def __init__(
        self,
        records: list[dict],
        tokenizer,
        *,
        max_source_length: int,
        max_target_length: int,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        self.tokenizer.src_lang = record["src_lang"]
        self.tokenizer.tgt_lang = record["tgt_lang"]
        source_encoding = self.tokenizer(
            record["src_text"],
            truncation=True,
            max_length=self.max_source_length,
        )
        target_encoding = self.tokenizer(
            text_target=record["tgt_text"],
            truncation=True,
            max_length=self.max_target_length,
        )
        example = {
            "input_ids": source_encoding["input_ids"],
            "attention_mask": source_encoding["attention_mask"],
            "labels": target_encoding["input_ids"],
        }
        return _filter_tokenized_example(example, context=f"record {index}")


class SafeSeq2SeqCollator:
    """Wrap DataCollatorForSeq2Seq and return only final allowed model inputs."""

    def __init__(self, base_collator) -> None:
        self.base_collator = base_collator

    def __call__(self, features):
        batch = self.base_collator(features)
        return _filter_model_inputs(batch)


def make_safe_seq2seq_trainer(base_trainer_class):
    """Create a trainer subclass that strips unexpected keys before loss."""

    class SafeSeq2SeqTrainer(base_trainer_class):
        def compute_loss(self, model, inputs, *args, **kwargs):
            inputs = _filter_model_inputs(inputs)
            return super().compute_loss(model, inputs, *args, **kwargs)

    return SafeSeq2SeqTrainer


def _assert_training_batch_is_safe(dataset, data_collator) -> None:
    """Validate dataset and collated batch keys before trainer.train()."""
    raw_features = [dataset[0]]
    dataset_keys = set(raw_features[0])
    if dataset_keys != DATASET_ALLOWED_KEYS:
        raise ValueError(
            "Training dataset items must contain only "
            f"{sorted(DATASET_ALLOWED_KEYS)}, got {sorted(dataset_keys)}."
        )
    dataset_forbidden = dataset_keys & FORBIDDEN_MODEL_INPUT_KEYS
    if dataset_forbidden:
        raise ValueError(
            "Training dataset item contains forbidden decoder/input fields: "
            f"{sorted(dataset_forbidden)}"
        )

    collated = data_collator(raw_features)
    collated_keys = set(collated)
    missing = FINAL_MODEL_INPUT_KEYS - collated_keys
    if missing:
        raise ValueError(
            "Collated training batch is missing required fields: "
            f"{sorted(missing)}."
        )
    unexpected = collated_keys - FINAL_MODEL_INPUT_KEYS
    if unexpected:
        raise ValueError(
            "Collated training batch contains unexpected fields: "
            f"{sorted(unexpected)}. Expected only {sorted(FINAL_MODEL_INPUT_KEYS)}."
        )
    still_forbidden = FORBIDDEN_MODEL_INPUT_KEYS & collated_keys
    if still_forbidden:
        raise ValueError(
            "Forbidden model input keys remain after collation: "
            f"{sorted(still_forbidden)}"
        )
    print(f"Final training batch keys: {FINAL_MODEL_INPUT_KEY_ORDER}")


def _make_compute_metrics(tokenizer):
    def compute_metrics(eval_pred) -> dict[str, float]:
        import numpy as np
        from sacrebleu.metrics import BLEU, CHRF

        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_predictions = tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        bleu = BLEU().corpus_score(decoded_predictions, [decoded_labels]).score
        chrf = CHRF(word_order=2).corpus_score(decoded_predictions, [decoded_labels]).score
        return {"bleu": bleu, "chrf": chrf}

    return compute_metrics


def _training_args_kwargs(
    *,
    output_dir: Path,
    training_config: dict,
    seed: int,
    fp16_used: bool,
    num_train_epochs: float,
    num_beams: int,
) -> dict[str, Any]:
    return {
        "output_dir": str(output_dir / "checkpoints"),
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 10,
        "learning_rate": float(training_config.get("learning_rate", 2e-5)),
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": int(
            training_config.get("per_device_train_batch_size", 2)
        ),
        "per_device_eval_batch_size": int(
            training_config.get("per_device_eval_batch_size", 4)
        ),
        "gradient_accumulation_steps": int(
            training_config.get("gradient_accumulation_steps", 1)
        ),
        "warmup_ratio": float(training_config.get("warmup_ratio", 0.0)),
        "weight_decay": float(training_config.get("weight_decay", 0.0)),
        "label_smoothing_factor": float(
            training_config.get("label_smoothing_factor", 0.0)
        ),
        "fp16": fp16_used,
        "predict_with_generate": True,
        "generation_num_beams": num_beams,
        "generation_max_length": int(training_config.get("max_target_length", 128)),
        "load_best_model_at_end": True,
        "metric_for_best_model": "chrf",
        "greater_is_better": True,
        "save_total_limit": 2,
        "seed": seed,
        "data_seed": seed,
        "report_to": [],
        "remove_unused_columns": False,
    }


def train_from_config(
    *,
    mode: str,
    config_path: str | Path,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    num_train_epochs_override: float | None = None,
    output_dir_override: str | Path | None = None,
) -> dict:
    """Fine-tune NLLB from a project YAML config."""
    if mode not in TRAINABLE_MODES:
        raise ValueError(f"Mode {mode!r} is not trainable. Supported: {TRAINABLE_MODES}")

    try:
        import torch
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise ImportError(
            "Training requires torch and transformers. Install requirements.txt first."
        ) from exc

    config = _load_config(config_path)
    if config.get("mode") != mode:
        raise ValueError(
            f"Config mode {config.get('mode')!r} does not match CLI mode {mode!r}."
        )
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported experiment mode: {mode}")

    training_config = dict(config.get("training", {}))
    output_dir = Path(output_dir_override or config["output_dir"])
    ensure_dir(output_dir)
    seed = int(config.get("seed", 42))
    set_seed(seed)

    train_records = _slice_records(
        read_jsonl(config["train_file"]), max_train_samples
    )
    dev_records = _slice_records(read_jsonl(config["dev_file"]), max_eval_samples)
    _validate_records(train_records, path=config["train_file"], expected_split="train")
    _validate_records(dev_records, path=config["dev_file"], expected_split="dev")

    fp16_requested = bool(training_config.get("fp16", False))
    fp16_used = resolve_fp16(fp16_requested, torch)
    device_summary = get_device_summary(torch)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name"])
    forced_bos_token_id = get_forced_bos_token_id(tokenizer, config.get("target_lang", ENGLISH))
    model.config.forced_bos_token_id = forced_bos_token_id

    max_source_length = int(training_config.get("max_source_length", 128))
    max_target_length = int(training_config.get("max_target_length", 128))
    train_dataset = TranslationJsonlDataset(
        train_records,
        tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
    )
    eval_dataset = TranslationJsonlDataset(
        dev_records,
        tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
    )

    base_data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )
    data_collator = SafeSeq2SeqCollator(base_data_collator)
    _assert_training_batch_is_safe(train_dataset, data_collator)

    num_train_epochs = float(
        num_train_epochs_override
        if num_train_epochs_override is not None
        else training_config.get("num_train_epochs", 3)
    )
    args_kwargs = _training_args_kwargs(
        output_dir=output_dir,
        training_config=training_config,
        seed=seed,
        fp16_used=fp16_used,
        num_train_epochs=num_train_epochs,
        num_beams=int(config.get("num_beams", 5)),
    )

    try:
        training_args = Seq2SeqTrainingArguments(**args_kwargs)
        metric_name = "chrf"
        best_metric_name = "eval_chrf"
    except Exception as exc:
        print(
            "Warning: best-by-chrF Trainer setup failed; falling back to eval_loss "
            f"checkpoint selection. Reason: {exc}"
        )
        args_kwargs["metric_for_best_model"] = "eval_loss"
        args_kwargs["greater_is_better"] = False
        training_args = Seq2SeqTrainingArguments(**args_kwargs)
        metric_name = "loss"
        best_metric_name = "eval_loss"

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "compute_metrics": _make_compute_metrics(tokenizer)
        if metric_name == "chrf"
        else None,
    }
    trainer_params = inspect.signature(Seq2SeqTrainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    SafeTrainer = make_safe_seq2seq_trainer(Seq2SeqTrainer)
    trainer = SafeTrainer(**trainer_kwargs)

    train_result = trainer.train()
    trainer.save_state()

    saved_model_path = output_dir / "best_model"
    trainer.save_model(str(saved_model_path))
    tokenizer.save_pretrained(str(saved_model_path))

    train_metrics = dict(train_result.metrics)
    train_metrics["train_samples"] = len(train_records)
    write_json(train_metrics, output_dir / "train_metrics.json")

    resolved_config = {
        "mode": mode,
        "model_name": config["model_name"],
        "train_file": config["train_file"],
        "dev_file": config["dev_file"],
        "test_file": config.get("test_file"),
        "output_dir": str(output_dir),
        "target_lang": config.get("target_lang", ENGLISH),
        "num_beams": int(config.get("num_beams", 5)),
        "seed": seed,
        "training": {
            **training_config,
            "num_train_epochs": num_train_epochs,
            "fp16_requested": fp16_requested,
            "fp16_used": fp16_used,
            "max_source_length": max_source_length,
            "max_target_length": max_target_length,
        },
        "overrides": {
            "max_train_samples": max_train_samples,
            "max_eval_samples": max_eval_samples,
            "num_train_epochs_override": num_train_epochs_override,
            "output_dir_override": str(output_dir_override)
            if output_dir_override is not None
            else None,
        },
    }
    write_json(resolved_config, output_dir / "train_config_resolved.json")

    best_metric_value = None
    if trainer.state.best_metric is not None:
        best_metric_value = float(trainer.state.best_metric)

    summary = {
        "mode": mode,
        "model_name": config["model_name"],
        "train_file": config["train_file"],
        "dev_file": config["dev_file"],
        "output_dir": str(output_dir),
        "seed": seed,
        "num_train_records": len(train_records),
        "num_dev_records": len(dev_records),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "saved_model_path": str(saved_model_path),
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
        "fp16_actually_used": fp16_used,
        "device_summary": device_summary,
    }
    write_json(summary, output_dir / "training_summary.json")
    return summary


def main() -> None:
    """Run NLLB fine-tuning from the command line."""
    args = parse_args()
    summary = train_from_config(
        mode=args.mode,
        config_path=args.config,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        num_train_epochs_override=args.num_train_epochs_override,
        output_dir_override=args.output_dir_override,
    )
    print("Training complete:")
    print(f"  mode: {summary['mode']}")
    print(f"  saved model: {summary['saved_model_path']}")
    print(f"  train records: {summary['num_train_records']}")
    print(f"  dev records: {summary['num_dev_records']}")
    print(f"  best metric: {summary['best_metric_name']}={summary['best_metric_value']}")
    print(f"  fp16 used: {summary['fp16_actually_used']}")


if __name__ == "__main__":
    main()
