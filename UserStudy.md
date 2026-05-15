
Bước 1: Cho làm hand-written exam (không cho biết điểm)

python scripts/visualize_questions.py --no_score --json_path post-test/eco.json --share

sau đó điểm sẽ được upload lên google sheet

Bước 2: Cho đọc pdf (condition cũng có)

Bước 3: Cho làm generated question (có hình ảnh hoặc không)

python scripts/visualize_questions.py --json_path outputs/20260426_000329_eco_12bf04/generation/questions.json --share

Bước 4:
python scripts/visualize_questions.py --json_path post-test/eco.json --share
