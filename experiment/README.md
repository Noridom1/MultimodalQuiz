# Experiment Runner

This folder contains a batch runner to execute either baseline or mmquiz generation on a directory of PDF files.

## Usage

Run baseline:

```powershell
python experiment/run_experiment.py --method baseline --input-dir data/raw --num-questions 5
```

Run mmquiz:

```powershell
python experiment/run_experiment.py --method mmquiz --input-dir data/raw --num-questions 5
```

## Key Flags

- `--method`: `baseline` or `mmquiz`
- `--input-dir`: directory containing `.pdf` files
- `--num-questions`: question count per document
- `--easy`, `--medium`, `--hard`: difficulty distribution
- `--recursive`: discover PDFs recursively
- `--limit N`: run at most N files
- `--dry-run`: print planned runs without execution
- `--stop-on-error`: stop batch after first failure
- `--echo-run-logs`: print tail of each run log to console
- `--log-level`: experiment runner log verbosity

MMQuiz-only flags:

- `--generation-mode topic_agentic|legacy`
- `--mock-image`
- `--mock-question`

## Output Layout

All experiment outputs are grouped under:

```text
experiment/
  baseline/
    <document_name>_<num_questions>/
      ...run artifacts...
    experiment.log
    experiment_summary.json
  mmquiz/
    <document_name>_<num_questions>/
      ...run artifacts...
    experiment.log
    experiment_summary.json
```

Naming convention for each run folder:

- `<document_name>_<number_of_questions>`
- Example: `AugustRevolution_5`

If a naming collision occurs, a numeric suffix is appended (`_2`, `_3`, ...).

## Logging

The runner writes:

- `experiment/<method>/experiment.log`: orchestration events (start/end, per-file status, timing)
- `experiment/<method>/experiment_summary.json`: structured summary for all processed files

Each method run still writes its own per-run log at:

- `<run_root>/logs/pipeline.log`

The summary includes `manifest_path` and `log_path` for successful runs.

## Exit Codes

- `0`: all runs succeeded (or dry-run)
- `1`: at least one run failed

## Automatic Metrics Workflow

Use the metrics workflow in two steps:

1. Extract concepts from source documents and generated questions.
2. Compute coverage, breadth, and duplication from saved concepts.

### Step 1: Extract Concepts

Run concept extraction for baseline outputs:

```powershell
python experiment/metrics/extract_concepts.py --outputs-baseline-root outputs-baseline --raw-documents-root data/raw --experiment-root experiment
```

Default behavior is resume/continue mode: documents that already have both concept files are skipped, so reruns continue from unfinished docs.

Force re-run all documents:

```powershell
python experiment/metrics/extract_concepts.py --outputs-baseline-root outputs-baseline --raw-documents-root data/raw --experiment-root experiment --no-resume
```

Configure retry for rate limits/transient errors:

```powershell
python experiment/metrics/extract_concepts.py --outputs-baseline-root outputs-baseline --raw-documents-root data/raw --experiment-root experiment --max-retries 8 --initial-backoff-seconds 1.5 --max-backoff-seconds 30
```

This writes:

- `experiment/metrics/concepts/docX/document_concepts.json`
- `experiment/metrics/concepts/docX/question_concepts.json`
- `experiment/metrics/concepts/concept_extraction_summary.json`

### Step 2: Run Metrics

Compute metrics from extracted concept artifacts:

```powershell
python experiment/metrics/run_metrics.py --outputs-baseline-root outputs-baseline --experiment-root experiment --coverage-threshold 0.75
```

This writes:

- `experiment/metrics/results/docX_metrics.json`
- `experiment/metrics/results/baseline_metrics_summary.json`

### Optional: One-Shot Run

If you want to extract concepts and compute metrics in one command:

```powershell
python experiment/metrics/run_metrics.py --outputs-baseline-root outputs-baseline --raw-documents-root data/raw --experiment-root experiment --coverage-threshold 0.75 --auto-extract
```

`--auto-extract` also supports continue + retry flags:

```powershell
python experiment/metrics/run_metrics.py --outputs-baseline-root outputs-baseline --raw-documents-root data/raw --experiment-root experiment --coverage-threshold 0.75 --auto-extract --max-retries 8 --initial-backoff-seconds 1.5 --max-backoff-seconds 30
```
