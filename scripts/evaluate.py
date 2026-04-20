from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated quiz outputs.")
    parser.add_argument("results_path", type=Path, help="Path to a generated quiz JSON file")
    args = parser.parse_args()

    _results = read_json(args.results_path)
    print("Evaluation entry point is ready. Add scoring logic here.")


if __name__ == "__main__":
    main()
