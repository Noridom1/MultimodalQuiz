# Deploy: Vercel frontend only

## Scope

- **In scope**: Host the Vite/React app from [`frontend/`](../../frontend/) on **Vercel** (static build + SPA rewrites via [`frontend/vercel.json`](../../frontend/vercel.json)).
- **Out of scope for this plan**: Deploying FastAPI/agents to Render or elsewhere. Document that as optional follow-up when a public API URL is needed.

## Steps

1. **Vercel project**: Import repo → **Root Directory** `frontend` → Framework Vite → build outputs `dist`.
2. **Env vars on Vercel** (see [`frontend/.env.example`](../../frontend/.env.example)): `VITE_SUPABASE_*`; set `VITE_API_BASE_URL` only when a reachable API exists (hosted or tunnel).
3. **Supabase Auth**: Site URL + Redirect URLs include the Vercel origin(s).
4. **Optional later**: Host API (`uvicorn backend.main:app`), set `UI_ALLOWED_ORIGIN(S)` to Vercel URL, point `VITE_API_BASE_URL` at that host — see [`README.run.md`](../../README.run.md) §10.

## Done when

- Production Vercel URL loads the SPA; deep links refresh correctly (rewrite rule).
