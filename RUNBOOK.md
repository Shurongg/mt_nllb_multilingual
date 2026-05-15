# Runbook

This runbook describes the planned workflow. Phase 1 data preparation, Phase 2 balanced joint-data construction, and Phase 3 zero-shot translation are implemented; training and evaluation scripts are still skeletons.

## 1. Prepare Data

Create processed JSONL files for Javanese-English and Indonesian-English.

Planned behavior:

- Convert the existing Javanese-English train, valid, and test files.
- Map Javanese-English valid to processed `dev`.
- Keep the original Javanese-English test set untouched.
- Create Indonesian-English train only.
- Do not create Indonesian-English dev or test.

```bash
python -m src.prepare_data \
  --jav_raw_dir data/raw/jav_eng \
  --ind_raw_dir data/raw/ind_eng \
  --output_dir data/processed \
  --report_dir data/reports
```

Expected outputs:

```text
data/processed/jav_eng/train.jsonl
data/processed/jav_eng/dev.jsonl
data/processed/jav_eng/test.jsonl
data/processed/ind_eng/train.jsonl
data/reports/data_stats.json
data/reports/data_stats.md
```

## 2. Build Joint Data

Create the balanced multilingual training file for `joint_balanced`.

```bash
python -m src.build_joint_data \
  --jav_train data/processed/jav_eng/train.jsonl \
  --ind_train data/processed/ind_eng/train.jsonl \
  --output_file data/processed/joint/train_balanced.jsonl \
  --seed 42 \
  --ratio 1.0
```

Expected output:

```text
data/processed/joint/train_balanced.jsonl
```

With the current data, the default ratio creates 500 Javanese records and 500 sampled Indonesian records for 1000 total joint training records.

## 3. Run zero_shot

```bash
python -m src.translate_nllb \
  --model_name facebook/nllb-200-distilled-600M \
  --input_file data/processed/jav_eng/test.jsonl \
  --output_dir outputs/zero_shot/jav_eng \
  --target_lang eng_Latn \
  --num_beams 5 \
  --batch_size 8
```

Equivalent config-driven command:

```bash
python -m src.run_experiment \
  --mode zero_shot \
  --stage translate \
  --config configs/zero_shot.yaml
```

## 4. Run java_only

```bash
python src/run_experiment.py --mode java_only --stage all --config configs/java_only.yaml
```

## 5. Run joint_balanced

```bash
python src/run_experiment.py --mode joint_balanced --stage all --config configs/joint_balanced.yaml
```

## 6. Evaluate

Planned metrics:

- BLEU
- chrF++
- Optional COMET later

```bash
python src/evaluate_mt.py --help
```

## 7. Make Results Table

```bash
python src/make_results_table.py --help
```
