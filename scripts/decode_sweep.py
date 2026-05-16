import argparse
import json
from pathlib import Path

import sacrebleu
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def read_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def safe_name(model_name, beams, length_penalty, max_new_tokens):
    return f"beams{beams}_lp{str(length_penalty).replace('.', 'p')}_maxnew{max_new_tokens}"


def translate_records(model, tokenizer, records, target_lang, batch_size, beams, length_penalty, max_new_tokens, device):
    hypotheses = [None] * len(records)

    # Keep order, but batch by source language because NLLB uses tokenizer.src_lang.
    by_lang = {}
    for i, r in enumerate(records):
        by_lang.setdefault(r["src_lang"], []).append((i, r))

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)
    if forced_bos_token_id is None or forced_bos_token_id == tokenizer.unk_token_id:
        raise ValueError(f"Could not resolve target language token: {target_lang}")

    for src_lang, indexed_records in by_lang.items():
        tokenizer.src_lang = src_lang

        for start in range(0, len(indexed_records), batch_size):
            batch = indexed_records[start:start + batch_size]
            idxs = [x[0] for x in batch]
            texts = [x[1]["src_text"] for x in batch]

            encoded = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(device)

            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    forced_bos_token_id=forced_bos_token_id,
                    num_beams=beams,
                    length_penalty=length_penalty,
                    max_new_tokens=max_new_tokens,
                )

            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

            for idx, hyp in zip(idxs, decoded):
                hypotheses[idx] = hyp.strip()

    if any(x is None for x in hypotheses):
        missing = sum(x is None for x in hypotheses)
        raise RuntimeError(f"Missing hypotheses: {missing}")

    return hypotheses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--target_lang", default="eng_Latn")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_beams", nargs="+", type=int, default=[4, 5, 8])
    parser.add_argument("--length_penalty", nargs="+", type=float, default=[0.8, 1.0, 1.2])
    parser.add_argument("--max_new_tokens", nargs="+", type=int, default=[64, 96, 128])
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(args.input_file)
    print(f"Loaded records: {len(records)}")

    sources = [r["src_text"] for r in records]
    references = [r["tgt_text"] for r in records]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("model:", args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(device)
    model.eval()

    results = []

    chrf_metric = sacrebleu.CHRF(word_order=2)

    for beams in args.num_beams:
        for lp in args.length_penalty:
            for max_new in args.max_new_tokens:
                run_name = safe_name(args.model_name, beams, lp, max_new)
                out_dir = output_root / run_name
                out_dir.mkdir(parents=True, exist_ok=True)

                print(f"\n== {run_name} ==")

                hyps = translate_records(
                    model=model,
                    tokenizer=tokenizer,
                    records=records,
                    target_lang=args.target_lang,
                    batch_size=args.batch_size,
                    beams=beams,
                    length_penalty=lp,
                    max_new_tokens=max_new,
                    device=device,
                )

                bleu = sacrebleu.corpus_bleu(hyps, [references]).score
                chrf = chrf_metric.corpus_score(hyps, [references]).score
                empty = sum(1 for x in hyps if not x.strip())

                (out_dir / "sources.txt").write_text("\n".join(sources) + "\n", encoding="utf-8")
                (out_dir / "references.txt").write_text("\n".join(references) + "\n", encoding="utf-8")
                (out_dir / "hypotheses.txt").write_text("\n".join(hyps) + "\n", encoding="utf-8")

                preds = []
                for r, hyp in zip(records, hyps):
                    preds.append({
                        "id": r["id"],
                        "src_lang": r["src_lang"],
                        "tgt_lang": args.target_lang,
                        "source": r["src_text"],
                        "reference": r["tgt_text"],
                        "hypothesis": hyp,
                    })

                with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
                    for p in preds:
                        f.write(json.dumps(p, ensure_ascii=False) + "\n")

                metrics = {
                    "num_samples": len(records),
                    "num_beams": beams,
                    "length_penalty": lp,
                    "max_new_tokens": max_new,
                    "bleu": bleu,
                    "chrf": chrf,
                    "empty_hypotheses": empty,
                    "output_dir": str(out_dir),
                }
                (out_dir / "metrics.json").write_text(
                    json.dumps(metrics, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

                results.append(metrics)

                print(f"BLEU={bleu:.4f} chrF++={chrf:.4f} empty={empty}")

    results = sorted(results, key=lambda x: x["chrf"], reverse=True)

    summary_path = output_root / "decode_sweep_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    md_path = output_root / "decode_sweep_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| Rank | beams | length_penalty | max_new_tokens | BLEU | chrF++ | empty | output_dir |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for rank, r in enumerate(results, start=1):
            f.write(
                f"| {rank} | {r['num_beams']} | {r['length_penalty']} | {r['max_new_tokens']} | "
                f"{r['bleu']:.4f} | {r['chrf']:.4f} | {r['empty_hypotheses']} | `{r['output_dir']}` |\n"
            )

    print("\nTop 5 by chrF++:")
    for r in results[:5]:
        print(
            f"beams={r['num_beams']} lp={r['length_penalty']} max_new={r['max_new_tokens']} "
            f"BLEU={r['bleu']:.4f} chrF++={r['chrf']:.4f}"
        )

    print("\nwrote:", summary_path)
    print("wrote:", md_path)


if __name__ == "__main__":
    main()
