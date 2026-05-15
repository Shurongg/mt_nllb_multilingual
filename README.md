# NLLB Multilingual Fine-Tuning for Javanese-English

This project is a course-project scaffold for studying multilingual transfer in machine translation using `facebook/nllb-200-distilled-600M`.

The main translation direction is Javanese to English. Indonesian-English data is used only as auxiliary training data for the balanced multilingual experiment.

## Supported Experiment Modes

- `zero_shot`: evaluate the base NLLB model on the Javanese-English test set without fine-tuning.
- `java_only`: fine-tune on Javanese-English train data and evaluate on the untouched Javanese-English test set.
- `joint_balanced`: fine-tune on balanced Javanese-English and Indonesian-English train data, validate on Javanese-English dev, and evaluate on the untouched Javanese-English test set.

The main comparison is:

```text
java_only vs joint_balanced
```

This comparison tests whether Indonesian-English auxiliary data improves Javanese-English translation.

## Excluded Modes

This scaffold intentionally does not include:

- `indo_only`
- `joint_rawmix`
- pivot translation
- fairseq-based training
- any other experimental mode

## Evaluation

Final evaluation is only on the original Javanese-English test set:

```text
data/processed/jav_eng/test.jsonl
```

The original test split must remain untouched and must not be used for training or development.

## Planned Commands

Phase 1 data preparation, Phase 2 balanced joint-data construction, Phase 3 zero-shot translation, and Phase 4 local BLEU/chrF++ evaluation are implemented. Training is still a planned skeleton at this stage.

```bash
python -m src.prepare_data \
  --jav_raw_dir data/raw/jav_eng \
  --ind_raw_dir data/raw/ind_eng \
  --output_dir data/processed \
  --report_dir data/reports

python -m src.build_joint_data \
  --jav_train data/processed/jav_eng/train.jsonl \
  --ind_train data/processed/ind_eng/train.jsonl \
  --output_file data/processed/joint/train_balanced.jsonl \
  --seed 42 \
  --ratio 1.0

python -m src.translate_nllb \
  --model_name facebook/nllb-200-distilled-600M \
  --input_file data/processed/jav_eng/test.jsonl \
  --output_dir outputs/zero_shot/jav_eng \
  --target_lang eng_Latn \
  --num_beams 5 \
  --batch_size 8

python -m src.run_experiment --mode zero_shot --stage translate --config configs/zero_shot.yaml
python -m src.run_experiment --mode zero_shot --stage evaluate --config configs/zero_shot.yaml
python src/run_experiment.py --mode java_only --stage all --config configs/java_only.yaml
python src/run_experiment.py --mode joint_balanced --stage all --config configs/joint_balanced.yaml

python src/make_results_table.py --help
```
