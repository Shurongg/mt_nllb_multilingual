# Experiment Plan

## Research Question

Does adding Indonesian-English auxiliary training data improve Javanese-English machine translation when fine-tuning `facebook/nllb-200-distilled-600M`?

## Model

- Base model: `facebook/nllb-200-distilled-600M`
- Main direction: Javanese to English
- Source language code for Javanese: `jav_Latn`
- Source language code for Indonesian: `ind_Latn`
- Target language code for English: `eng_Latn`

## Data Setting

- Javanese-English has train and test data.
- Javanese-English train data will be split into train and dev.
- The original Javanese-English test set is kept untouched for final evaluation only.
- Indonesian-English has train data only.
- Indonesian-English is used only as auxiliary training data in `joint_balanced`.
- No Indonesian-English dev or test split is created.

## Experiment Designs

### zero_shot

Evaluate the base NLLB model directly on the Javanese-English test set without fine-tuning.

### java_only

Fine-tune NLLB on Javanese-English train data only. Use the Javanese-English dev split for validation and the untouched Javanese-English test set for final evaluation.

### joint_balanced

Fine-tune NLLB on a balanced mixture of Javanese-English train data and Indonesian-English train data. Use the Javanese-English dev split for validation and the untouched Javanese-English test set for final evaluation.

## Evaluation Metrics

- BLEU
- chrF++
- COMET can be added later as an optional metric, but it is not a hard dependency in the initial scaffold.

## Result Interpretation

- `joint_balanced > java_only`: evidence of positive transfer from Indonesian-English auxiliary data.
- `joint_balanced ≈ java_only`: no clear transfer effect.
- `joint_balanced < java_only`: evidence of negative transfer or harmful interference.

