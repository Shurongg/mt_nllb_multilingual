# NLLB Multilingual Fine-Tuning for Javanese-English

This course project studies whether Indonesian-English auxiliary data improves Javanese-English machine translation when fine-tuning `facebook/nllb-200-distilled-600M`.

The main task is Javanese -> English translation. Indonesian-English is used only as auxiliary training data, and final evaluation is always on the Javanese-English test set.

## Project Status

Completed pipeline:

- Data preparation from raw parallel files into JSONL.
- Joint training-data construction for multilingual transfer experiments.
- NLLB fine-tuning with `Seq2SeqTrainer`.
- Translation generation with NLLB checkpoints.
- BLEU and chrF++ evaluation with `sacrebleu`.
- COMET evaluation with `wmt22-comet-da` in a separate COMET environment.

Supported experiment modes remain:

- `zero_shot`
- `java_only`
- `joint_balanced`

The project intentionally does not include `indo_only`, `joint_rawmix`, pivot translation, or fairseq training.

## Dataset Summary

| Dataset | Split | Records | Use |
|---|---|---:|---|
| Javanese-English | train | 500 | Main training data |
| Javanese-English | dev | 100 | Checkpoint selection and tuning |
| Javanese-English | test | 400 | Final evaluation only |
| Indonesian-English | train | 17,850 | Auxiliary training data only |

Language codes:

- Javanese: `jav_Latn`
- Indonesian: `ind_Latn`
- English: `eng_Latn`

## Final Results

Metrics are BLEU, chrF++, and COMET. COMET was computed with `wmt22-comet-da`.

| Run | Train records | Dev chrF++ | Test BLEU | Test chrF++ | Test COMET |
|---|---:|---:|---:|---:|---:|
| java_only_e8 | 500 | 43.5624 | 16.8377 | 42.0921 | 0.72975 |
| joint_1to1_e8 | 1000 | 43.7481 | 17.1178 | 42.1754 | 0.73329 |
| joint_1to2_e8 | 1500 | 43.6208 | 17.3869 | 42.3184 | 0.73448 |
| joint_1to3_e8 | 2000 | 43.5057 | 17.3864 | 42.4393 | 0.73345 |
| joint_2to1_e8 | 750 | 43.7709 | 17.2290 | 42.2404 | 0.73271 |
| joint_javx2_ind1000_e8 | 2000 | 44.2739 | 17.8214 | 42.7937 | 0.73885 |
| joint_javx2_ind1500_e8 | 2500 | 44.7401 | 17.8564 | 42.9802 | 0.73947 |

## Final Selected Model

Selected run: `joint_javx2_ind1500_e8`

Training composition:

- Javanese-English train x2: 1000 effective Javanese examples.
- Indonesian-English auxiliary examples: 1500.
- Total training records: 2500.

Final selected outputs:

- Model directory: `outputs/joint_javx2_ind1500_e8/jav_eng/best_model`
- Training summary: `outputs/joint_javx2_ind1500_e8/jav_eng/training_summary.json`
- Test metrics: `outputs/joint_javx2_ind1500_e8/jav_eng/test_eval/metrics_with_comet.json`

This model was selected because it produced the best dev chrF++ and the best final test chrF++ among the tested optimization runs, while also improving BLEU and COMET over the Java-only baseline. The result supports positive transfer from Indonesian-English auxiliary data, but only when the Javanese signal is preserved through upsampling.

The target chrF++ of 45 was not reached. The optimized model still improved consistently over Java-only:

- BLEU: `16.8377` -> `17.8564`
- chrF++: `42.0921` -> `42.9802`
- COMET: `0.72975` -> `0.73947`

These are empirical project results; no statistical significance claim is made.

## Optimization Summary

Additional optimization experiments were tested after the initial Java-only and balanced-joint runs:

- Decoding sweep: beam size, length penalty, and `max_new_tokens` improved dev slightly, but reduced test chrF++, so it was not adopted.
- Filtered Indonesian data: TF-IDF similarity filtering and source-only similarity filtering were tested. `simfilter_1000` had strong dev chrF++ and BLEU, but did not beat the final model on test chrF++.
- Learning-rate sweep: `1e-5`, `1.5e-5`, `2e-5`, and `3e-5` were tested. The original `2e-5` setting remained best.
- Java x3 upsampling: Java x3 + Indo 1500 and Java x3 + Indo 2000 reduced dev chrF++ compared with Java x2, so they were not adopted.

Main conclusion: Indonesian-English auxiliary data helps Javanese-English translation, but too much auxiliary data or too much upsampling can hurt. The best balance in this project was Java x2 plus 1500 Indonesian-English auxiliary examples.

## Reproduction Notes

Install the training dependencies in a clean training environment, for example `mttrain`:

```bash
pip install -r requirements.txt
```

Prepare processed data:

```bash
python -m src.prepare_data \
  --jav_raw_dir data/raw/jav_eng \
  --ind_raw_dir data/raw/ind_eng \
  --output_dir data/processed \
  --report_dir data/reports
```

Build the initial balanced joint file:

```bash
python -m src.build_joint_data \
  --jav_train data/processed/jav_eng/train.jsonl \
  --ind_train data/processed/ind_eng/train.jsonl \
  --output_file data/processed/joint/train_balanced.jsonl \
  --seed 42 \
  --ratio 1.0
```

Run baseline training from existing configs:

```bash
python -m src.train_nllb --mode java_only --config configs/java_only.yaml
python -m src.train_nllb --mode joint_balanced --config configs/joint_balanced.yaml
```

Run the final selected training configuration:

```bash
python -m src.train_nllb \
  --mode joint_balanced \
  --config configs/joint_javx2_ind1500.yaml \
  --num_train_epochs_override 8
```

Generate translations from a saved checkpoint:

```bash
python -m src.translate_nllb \
  --model_name outputs/joint_javx2_ind1500_e8/jav_eng/best_model \
  --input_file data/processed/jav_eng/test.jsonl \
  --output_dir outputs/joint_javx2_ind1500_e8/jav_eng/test_eval \
  --target_lang eng_Latn \
  --num_beams 5 \
  --batch_size 8
```

Compute local BLEU and chrF++:

```bash
python -m src.evaluate_mt \
  --hypotheses outputs/joint_javx2_ind1500_e8/jav_eng/test_eval/hypotheses.txt \
  --references outputs/joint_javx2_ind1500_e8/jav_eng/test_eval/references.txt \
  --sources outputs/joint_javx2_ind1500_e8/jav_eng/test_eval/sources.txt \
  --output_json outputs/joint_javx2_ind1500_e8/jav_eng/test_eval/metrics.json \
  --mode joint_javx2_ind1500_e8
```

COMET should be run separately in a clean COMET environment, for example `mtcomet`, using the generated `sources.txt`, `references.txt`, and `hypotheses.txt`. Do not install COMET into the training environment, because it previously caused dependency conflicts. Existing final COMET outputs are stored under:

```text
outputs/joint_javx2_ind1500_e8/jav_eng/test_eval/metrics_with_comet.json
outputs/joint_javx2_ind1500_e8/jav_eng/test_eval/comet_scores.json
```

## Key Files

- `DATA_SCHEMA.md`: processed JSONL schema.
- `RUNBOOK.md`: step-by-step workflow notes.
- `configs/`: baseline and optimization training configs.
- `src/`: data preparation, training, translation, and local evaluation scripts.
- `scripts/`: exploratory data filtering and decoding sweep helpers.
- `outputs/final_reports/optimization_final_summary.md`: concise final optimization summary.
