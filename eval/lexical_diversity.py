"""Compute lexical diversity metrics: Distinct-N and Self-BLEU."""
import argparse
import json
from collections import Counter
from typing import List
from eval.utils import read_jsonl, ensure_dir
from nltk.tokenize import word_tokenize
import math
import nltk

# ensure punkt is available for tokenization
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


def distinct_n(texts: List[str], n: int) -> float:
    tokens = [word_tokenize(t.lower()) for t in texts]
    ngrams = []
    for t in tokens:
        for i in range(len(t) - n + 1):
            ngrams.append(tuple(t[i : i + n]))
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def self_bleu(texts: List[str]) -> float:
    # compute sentence-level BLEU of each sent vs rest, then average
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    smooth = SmoothingFunction().method1
    toks = [word_tokenize(t.lower()) for t in texts]
    scores = []
    for i, hyp in enumerate(toks):
        refs = [r for j, r in enumerate(toks) if j != i]
        if not refs:
            continue
        try:
            sc = sentence_bleu(refs, hyp, smoothing_function=smooth)
        except Exception:
            sc = 0.0
        scores.append(sc)
    return sum(scores) / len(scores) if scores else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True, help="JSONL with items containing 'text'")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    items = read_jsonl(args.file)
    texts = [it.get("text", "") for it in items]
    d1 = distinct_n(texts, 1)
    d2 = distinct_n(texts, 2)
    sb = self_bleu(texts)
    res = {"distinct_1": d1, "distinct_2": d2, "self_bleu": sb, "n_items": len(texts)}
    ensure_dir(args.out)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"Saved lexical diversity results to {args.out}")


if __name__ == "__main__":
    main()
