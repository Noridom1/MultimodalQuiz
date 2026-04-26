"""Orchestrator to run all evaluations programmatically."""
import argparse
import subprocess
import sys
import os


def run_cmd(cmd):
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd, check=False)
    return res.returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rag", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--pairs", required=True)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rc = run_cmd([sys.executable, "eval/win_rate.py", "--rag", args.rag, "--baseline", args.baseline, "--out", os.path.join(args.outdir, "win_rate.json")])
    if rc:
        print("win_rate failed")
    rc = run_cmd([sys.executable, "eval/clipscore_eval.py", "--pairs", args.pairs, "--out", os.path.join(args.outdir, "clipscore.json")])
    if rc:
        print("clipscore failed")
    rc = run_cmd([sys.executable, "eval/lexical_diversity.py", "--file", args.rag, "--out", os.path.join(args.outdir, "lexdiv.json")])
    if rc:
        print("lexical_diversity failed")


if __name__ == "__main__":
    main()
