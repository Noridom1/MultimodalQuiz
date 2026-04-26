Evaluation scripts for MultimodalQuiz

Place generated outputs as JSONL files with one JSON object per line. Each object should include at least:
- `id`: unique identifier
- `text`: generated question text
- `image_path` (optional): path to the image file associated with the item

Scripts:
- `win_rate.py` — compare two JSONL generators (RAGGen vs baseline) using an LLM judge. Outputs win/lose/tie percentages.
- `clipscore_eval.py` — computes CLIP-based image-text similarity (CLIPScore) for pairs.
- `lexical_diversity.py` — computes Distinct-N and Self-BLEU for a set of texts.
- `run_all.py` — simple orchestration runner to call the above scripts programmatically.

Usage examples (from repo root):

python eval/win_rate.py --rag rag.jsonl --baseline baseline.jsonl --out results/win_rate.json
python eval/clipscore_eval.py --pairs pairs.jsonl --out results/clipscore.json
python eval/lexical_diversity.py --file rag.jsonl --out results/lexdiv.json

Environment:
- Set `OPENAI_API_KEY` for LLM judge calls.
- Install packages in `eval/requirements.txt`.
