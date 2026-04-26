"""Win rate evaluation using an LLM judge.

Expect two JSONL files: `--rag` and `--baseline`. Each line is a JSON object
with at least `id` and `text`. Pairs are matched by `id`.
"""
import os
import json
import argparse
from typing import List
from tqdm import tqdm
import openai
from eval.utils import read_jsonl, pair_by_id, ensure_dir


SYSTEM_PROMPT = (
    "You are an impartial educational judge. Given two quiz questions A and B, "
    "decide which is pedagogically better: clarity, accuracy, appropriateness, and student learning value. "
    "Return a JSON object with keys: {\"winner\": \"A\"|\"B\"|\"Tie\", \"reason\": string}."
)


def call_judge(a_text: str, b_text: str, model: str = "gpt-4o-mini") -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question A:\n{a_text}\n\nQuestion B:\n{b_text}\n\nWhich is better? Reply only with the JSON."},
    ]
    resp = openai.ChatCompletion.create(model=model, messages=messages, temperature=0)
    text = resp.choices[0].message.content.strip()
    # try to parse JSON from text
    try:
        return json.loads(text)
    except Exception:
        # fallback: try to find JSON substring
        import re

        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    # if still failing, return tie with raw text
    return {"winner": "Tie", "reason": f"Unparseable judge output: {text}"}


def evaluate(rag_path: str, baseline_path: str, model: str = "gpt-4o-mini") -> dict:
    a = read_jsonl(rag_path)
    b = read_jsonl(baseline_path)
    pairs = pair_by_id(a, b)
    results = {"A": 0, "B": 0, "Tie": 0}
    details = []
    for a_item, b_item in tqdm(pairs, desc="Judging pairs"):
        j = call_judge(a_item.get("text", ""), b_item.get("text", ""), model=model)
        winner = j.get("winner", "Tie")
        if winner not in results:
            winner = "Tie"
        results[winner] += 1
        details.append({"id": a_item.get("id"), "winner": winner, "reason": j.get("reason")})
    total = sum(results.values())
    summary = {k: (v, v / total if total else 0) for k, v in results.items()}
    return {"summary": summary, "details": details}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rag", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--judge-model", default=os.getenv("WINRATE_JUDGE_MODEL", "gpt-4o-mini"))
    p.add_argument("--out", required=True)
    args = p.parse_args()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    res = evaluate(args.rag, args.baseline, model=args.judge_model)
    ensure_dir(args.out)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()
