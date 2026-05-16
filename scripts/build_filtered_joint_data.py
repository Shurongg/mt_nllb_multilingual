import argparse
import json
import random
import re
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def token_len(text):
    return len(text.split())


def alpha_ratio(text):
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    alpha = sum(1 for c in chars if c.isalpha())
    return alpha / len(chars)


def quality_ok(record):
    src = record["src_text"]
    tgt = record["tgt_text"]

    src_len = token_len(src)
    tgt_len = token_len(tgt)

    if src_len < 3 or src_len > 80:
        return False
    if tgt_len < 3 or tgt_len > 100:
        return False

    ratio = tgt_len / max(src_len, 1)
    if ratio < 0.35 or ratio > 3.2:
        return False

    if alpha_ratio(src) < 0.55:
        return False
    if alpha_ratio(tgt) < 0.55:
        return False

    # Remove URL-heavy or very noisy lines.
    if re.search(r"https?://|www\.", src.lower()):
        return False

    return True


def make_java_x2(jav_records):
    out = []
    for pass_id in [1, 2]:
        for r in jav_records:
            rr = dict(r)
            rr["id"] = f"{r['id']}_up{pass_id}"
            rr["upsample_pass"] = pass_id
            out.append(rr)
    return out


def combined_text(record):
    # Use only training-side information. No dev/test is used.
    return f"{record['src_text']} [SEP] {record['tgt_text']}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jav_train", required=True)
    parser.add_argument("--ind_train", required=True)
    parser.add_argument("--out_dir", default="data/processed/joint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_ns", nargs="+", type=int, default=[1000, 1500])
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)

    jav = load_jsonl(Path(args.jav_train))
    ind = load_jsonl(Path(args.ind_train))

    print("Loaded Java train:", len(jav))
    print("Loaded Indo train:", len(ind))

    ind_quality = [r for r in ind if quality_ok(r)]
    print("Quality-filtered Indo:", len(ind_quality))

    # TF-IDF similarity against Java train only.
    java_texts = [combined_text(r) for r in jav]
    indo_texts = [combined_text(r) for r in ind_quality]

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        min_df=1,
        max_features=200000,
    )

    matrix = vectorizer.fit_transform(java_texts + indo_texts)
    java_mat = matrix[:len(java_texts)]
    indo_mat = matrix[len(java_texts):]

    # For each Indo candidate, take max similarity to any Java train example.
    sims = cosine_similarity(indo_mat, java_mat)
    max_scores = np.asarray(sims.max(axis=1)).ravel()

    scored = []
    for r, score in zip(ind_quality, max_scores):
        rr = dict(r)
        rr["selection_method"] = "char_tfidf_maxsim_to_java_train"
        rr["selection_score"] = float(score)
        scored.append(rr)

    scored.sort(key=lambda x: x["selection_score"], reverse=True)

    scored_path = out_dir / "ind_eng_simfilter_candidates.jsonl"
    write_jsonl(scored_path, scored)
    print("Wrote ranked Indo candidates:", scored_path)

    java_x2 = make_java_x2(jav)

    for n in args.top_ns:
        selected_ind = scored[:n]
        records = java_x2 + selected_ind
        rng.shuffle(records)

        out_path = out_dir / f"train_javx2_ind{n}_simfilter.jsonl"
        write_jsonl(out_path, records)

        c = Counter(r["dataset"] for r in records)
        print("\nWrote:", out_path)
        print("Total:", len(records))
        print("Distribution:", dict(c))
        print("Top Indo score:", selected_ind[0]["selection_score"])
        print("Last selected Indo score:", selected_ind[-1]["selection_score"])


if __name__ == "__main__":
    main()
