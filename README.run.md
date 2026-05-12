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

### Frontend environment (Supabase Auth)

The login and sign-up pages (`/login`, `/signup`) use the Supabase JavaScript client. Create `frontend/.env` (or `.env.local`) with:

```env
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your_supabase_anon_public_key
```

Use the **anon** key from the Supabase dashboard (not the service role). Enable the Google, Apple, and Facebook providers under Authentication if you use those buttons.

#### Per-user notebooks (API)

The FastAPI server verifies the same Supabase session the browser uses. Add to the **repository root** `.env` (next to your other `SUPABASE_*` vars):

```env
SUPABASE_JWT_SECRET=your_jwt_secret_from_supabase_dashboard
```

Copy **Legacy JWT secret** from **Supabase → JWT Keys → Legacy JWT secret** when your project still issues **HS256** access tokens. If Supabase uses **asymmetric signing (ES256/RS256)**, the API verifies tokens with **`{SUPABASE_URL}/auth/v1/.well-known/jwks.json`** and does **not** need the legacy secret for those tokens.

When **`SUPABASE_URL`** is set, `/api/notebooks` requires `Authorization: Bearer <access_token>` and scopes notebooks by `owner_id`. Apply the `owner_id` migration at the bottom of [`supabase/schema.sql`](supabase/schema.sql) if your project was created before that column existed.

`GET /api/health` includes `auth_enabled: true` when `SUPABASE_URL` is set (JWT verification is active).

#### OAuth providers (Facebook, Google, Apple)

**Important:** Social login goes **browser → provider → Supabase → your app**. The provider (e.g. Meta) redirects to Supabase first, not to `localhost`.

1. **Supabase** — In the Supabase dashboard: **Authentication → URL configuration**. Set **Site URL** to your app origin (e.g. `http://localhost:5173`). Under **Redirect URLs**, add the same origin and paths you use after login (e.g. `http://localhost:5173` and `http://localhost:5173/**` or `http://localhost:5173/login`).

2. **Meta (Facebook) — Valid OAuth Redirect URIs** — Add **exactly** (replace with your project ref):

   ```text
   https://<your-project-ref>.supabase.co/auth/v1/callback
   ```

   You can copy the callback URL from **Supabase → Authentication → Providers → Facebook** (it is shown in the setup instructions). If this URI is missing, Meta shows errors like **“Can’t load URL”** / domain not allowed.

3. **Why Meta still complains about “localhost”** — Meta’s note that `http://localhost` redirect URIs are auto-allowed in development applies when **Facebook’s redirect_uri is on localhost**. With Supabase, that redirect_uri is **`https://….supabase.co/auth/v1/callback`**, so you must allow that URL explicitly (step 2). Your **final** return to the app still uses `redirectTo` (e.g. `http://localhost:5173/`), which Supabase handles after the callback.

4. **Meta — App Domains / JavaScript SDK** — In **App settings → Basic**, set **Website** / site URL as needed for your dev app (e.g. `http://localhost:5173`). If you use **“Allowed Domains for the JavaScript SDK”**, include the exact origin you open in the browser (e.g. `http://localhost:5173`). Use **`http://localhost:5173`**, not `http://127.0.0.1:5173`, unless you add the `127.0.0.1` origin everywhere (Meta treats them as different sites).

5. **App mode** — Keep the Meta app in **Development** while testing; the localhost redirect exception does not replace the need for the **Supabase callback** URI in step 2.

6. **“Invalid Scopes: email” (Meta)** — Facebook Login only allows scopes your app is allowed to request. The `email` permission must be enabled under **Meta → your app → Use cases / Permissions and features** (and may require App Review when live). This project’s Facebook button requests **`public_profile` only** so local sign-in works without `email`; once Meta grants `email` for your app, you can change the `scopes` option in `SupabaseAuthCard.jsx` to `public_profile,email` if you need the address on the Supabase user.

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

If you already applied an older schema, ensure `public.notebooks` has an **`owner_id uuid`** column (see the `alter table` at the bottom of `supabase/schema.sql`). The API returns a **clear JSON `detail`** from PostgREST on failure (for example unknown column). If you previously added `owner_id` with a foreign key to `auth.users` and inserts still fail with 400, drop that constraint so `owner_id` is a plain uuid: `ALTER TABLE public.notebooks DROP CONSTRAINT IF EXISTS notebooks_owner_id_fkey;`

Then set these values in `.env`:

```env
SUPABASE_URL=...
SUPABASE_SERVICE_ROLE_KEY=...
SUPABASE_JWT_SECRET=...
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

### Question formats (MCQ, true/false, fill-in-blank, matching)

- **CLI**: `--question-formats multiple_choice` for MCQ-only, or a JSON mix (escape quotes in PowerShell), for example:

```powershell
python scripts/run_pipeline.py data/raw/doc.pdf --question-formats '{\"multiple_choice\":0.34,\"true_false\":0.33,\"fill_in_blank\":0.33}'
```

- **Config**: [`configs/default.yaml`](configs/default.yaml) → `pipeline.question_format_distribution` (used when the CLI/API does not override formats).

- **HTTP API**: `POST /api/notebooks/{id}/generate` accepts optional `question_format_distribution`, for example `{"multiple_choice": 0.25, "true_false": 0.25, "fill_in_blank": 0.25, "matching": 0.25}`.

- **Studio UI**: When creating a quiz, **Question types** uses checkboxes. The selected types are split evenly; if all are unchecked, the request safely falls back to `multiple_choice` only. The player supports each format in the quiz panel.

## 8. Troubleshooting

`ModuleNotFoundError` or import errors:

- Make sure the virtual environment is activated.
- Re-run `pip install -r requirements.txt`.

Frontend cannot reach backend:

- Confirm backend is running on port `8000`.
- Confirm frontend is running on port `5173`.
- Check `UI_ALLOWED_ORIGIN` / `UI_ALLOWED_ORIGINS` in backend `.env` or host dashboard (must match your frontend origin exactly, including `https`).

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

## 10. Production deploy (Vercel frontend)

After both hosts exist, you can print the exact `UI_ALLOWED_ORIGIN`, `VITE_API_BASE_URL`, and Supabase URL lines with [`scripts/Print-DeployEnvValues.ps1`](scripts/Print-DeployEnvValues.ps1) (pass your Vercel and Render origins).

Only the **Vite SPA** under [`frontend/`](frontend/) belongs on Vercel. Quiz generation uses disk (`outputs/`, `data/uploads/`), so **do not** run the FastAPI app as Vercel serverless.

### Frontend on Vercel

1. Import this repo in Vercel.
2. Set **Root Directory** to `frontend` (required). If the root is the monorepo root, Vercel treats the FastAPI package folder [`api/`](api/) as **Python serverless** routes and `requirements.txt` as a Python app, which produces **500 / `FUNCTION_INVOCATION_FAILED` (python3.12)** on routes like `/` or `/favicon.ico`. Fix: **Root Directory → `frontend`**, then redeploy. If you must leave the project root at the repo root, [`vercel.json`](vercel.json) at the repo root builds `frontend/dist` and [`.vercelignore`](.vercelignore) excludes `api/` from the deployment bundle.
3. Framework: Vite (auto-detected); output directory `dist`. [`frontend/vercel.json`](frontend/vercel.json) rewrites all routes to `index.html` so React Router works on refresh.
4. **Environment variables** (Vercel → Project → Settings → Environment Variables). Copy names from [`frontend/.env.example`](frontend/.env.example):

   ```env
   VITE_SUPABASE_URL=https://your-project.supabase.co
   VITE_SUPABASE_ANON_KEY=your_anon_key
   ```

   Add **`VITE_API_BASE_URL`** only when you have a **reachable API** (hosted backend or a tunnel to your machine). Use the origin only, **no trailing slash**:

   ```env
   VITE_API_BASE_URL=https://your-backend-host.example.com
   ```

   Until an API URL is set, the deployed UI will not successfully call notebook/generate endpoints (Supabase auth may still work if configured).

### Supabase (production)

In **Supabase → Authentication → URL configuration**:

- Set **Site URL** to your Vercel production URL (e.g. `https://your-app.vercel.app`).
- Under **Redirect URLs**, add that origin and any paths you use after login (and preview URLs if needed).

### Optional: hosted API (Render / Railway / Fly.io)

When you want the live site to use a persistent backend instead of localhost:

1. Deploy the API with Docker (e.g. [`render.yaml`](render.yaml) Blueprint or any host running [`Dockerfile`](Dockerfile) / [`Procfile`](Procfile): `uvicorn api.main:app --host 0.0.0.0 --port $PORT`). The blueprint declares env keys (defaults for `QUIZGEN_*` / bucket; set secrets in the Render dashboard). Set remaining secrets like root `.env` (`SUPABASE_*`, LLM keys, `UI_ALLOWED_ORIGIN`, etc.).
2. **CORS**: set `UI_ALLOWED_ORIGIN` to your Vercel origin (no trailing slash), or **`UI_ALLOWED_ORIGINS`** for multiple preview/production URLs; see [`.env.example`](.env.example).
3. Set **`VITE_API_BASE_URL`** on Vercel to that API origin and redeploy.

Local Docker smoke test (optional):

```powershell
docker build -t multimodal-quiz-api .
docker run --rm -p 8000:8000 --env-file .env multimodal-quiz-api
```

### Smoke test (when API is hosted)

```powershell
python scripts/smoke_api.py --base-url https://your-backend-host.example.com
```

Then open the Vercel URL and exercise notebooks against the hosted API.
