# Data Schema

Processed data is stored as JSONL. Each line is one translation example.

## Required Fields

```json
{
  "id": "string",
  "src_lang": "jav_Latn",
  "tgt_lang": "eng_Latn",
  "src_text": "source sentence",
  "tgt_text": "target sentence",
  "dataset": "jav_eng",
  "split": "train"
}
```

## Field Definitions

- `id`: stable example identifier.
- `src_lang`: source language code.
- `tgt_lang`: target language code.
- `src_text`: source-side text.
- `tgt_text`: target-side reference text.
- `dataset`: dataset identifier, such as `jav_eng` or `ind_eng`.
- `split`: data split, such as `train`, `dev`, or `test`.

## Language Codes

- Javanese: `jav_Latn`
- Indonesian: `ind_Latn`
- English: `eng_Latn`

## Expected Processed Files

```text
data/processed/jav_eng/train.jsonl
data/processed/jav_eng/dev.jsonl
data/processed/jav_eng/test.jsonl
data/processed/ind_eng/train.jsonl
data/processed/joint/train_balanced.jsonl
```

Javanese-English uses the existing raw splits:

- `train.jv` + `train.en` -> `train.jsonl`
- `valid.jv` + `valid.en` -> `dev.jsonl`
- `test.jv` + `test.en` -> `test.jsonl`

The Javanese train set is not split during preparation.

No Indonesian-English dev or test files should be created.

## Cleaning Rules

- Strip leading and trailing whitespace.
- Remove records where source or target is empty after stripping.
- Remove exact duplicate source-target pairs within each output split.
- Do not lowercase, tokenize, apply BPE, or modify punctuation aggressively.
