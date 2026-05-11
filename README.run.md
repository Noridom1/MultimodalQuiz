# Run Guide

This document explains how to run the application locally without modifying the main [README.md](/d:/10%20Personal/10.11%20Projects/MultimodalQuiz/README.md).

## What Runs

The app has two parts:

- A FastAPI backend in `api/`
- A React + Vite frontend in `frontend/`

During local development:

- Backend runs on `http://localhost:8000`
- Frontend runs on `http://localhost:5173`
- Frontend requests to `/api` are proxied to the backend by Vite

## Prerequisites

Install:

- Python 3.10+ recommended
- Node.js 18+ recommended
- `npm`

## 1. Set Up Python

From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

## 2. Set Up Environment Variables

Copy `.env.example` to `.env` and fill in the values you need.

Minimal example:

```env
QUIZGEN_EXTRACTOR_BACKEND=langchain
QUIZGEN_LLM_PROVIDER=mistral
QUIZGEN_LLM_MODEL=devstral-medium-latest
MISTRAL_API_KEY=your_mistral_api_key
UI_ALLOWED_ORIGIN=http://localhost:5173
```

Optional Supabase-backed persistence:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key
SUPABASE_BUCKET=quiz-assets
```

Notes:

- If `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` are both set, the backend uses Supabase for notebook metadata and file/object storage.
- If those values are omitted, the app still runs in local mode.
- In local mode, notebook data is stored under `.localdata/` and uploaded source files are stored under `data/uploads/`.

## 3. Start the Backend

From the repository root with the virtual environment activated:

```powershell
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Health check:

```text
http://localhost:8000/api/health
```

The response shows whether Supabase is enabled.

## 4. Start the Frontend

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

## 5. First Run

Recommended flow after both servers are running:

1. Open the frontend in the browser.
2. Create a notebook.
3. Upload a PDF, Markdown, or text file.
4. Generate a quiz from the uploaded source.

When a source is uploaded:

- The backend saves a local copy under `data/uploads/<notebook_id>/`
- If Supabase is configured, the backend also uploads the file to the `quiz-assets` bucket
- Notebook/source metadata is stored either in Supabase or in local JSON files under `.localdata/`

## 6. Supabase Setup

If you want the Supabase-backed mode, apply the SQL in `supabase/schema.sql` to your Supabase project.

That schema creates:

- `public.notebooks`
- `public.notebook_sources`
- `public.notebook_messages`
- `public.notebook_runs`
- A public storage bucket named `quiz-assets`

Then set these values in `.env`:

```env
SUPABASE_URL=...
SUPABASE_SERVICE_ROLE_KEY=...
SUPABASE_BUCKET=quiz-assets
```

## 7. Running the Quiz Pipeline Directly

The UI ultimately calls the Python quiz pipeline. You can also run it directly:

```powershell
python scripts/run_pipeline.py data/raw/astronomy.pdf
```

Example with options:

```powershell
python scripts/run_pipeline.py data/raw/astronomy.pdf --num-questions 5
```

Run outputs are written under `outputs/`.

## 8. Troubleshooting

`ModuleNotFoundError` or import errors:

- Make sure the virtual environment is activated.
- Re-run `pip install -r requirements.txt`.

Frontend cannot reach backend:

- Confirm backend is running on port `8000`.
- Confirm frontend is running on port `5173`.
- Check `UI_ALLOWED_ORIGIN` in `.env`.

Uploads work but Supabase is not used:

- Check `/api/health`.
- `supabase_enabled` is only `true` when both `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` are set.

Quiz generation fails:

- Verify your LLM provider settings in `.env`.
- Confirm the matching API key is set.

## 9. Quick Start Summary

Terminal 1:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.
