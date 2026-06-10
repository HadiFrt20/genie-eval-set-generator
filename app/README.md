# Genie Eval Generator — React + FastAPI app

Frontend on top of the `genie_eval_set_generator` Databricks notebook. It
detects Genie spaces in the workspace, exposes the key notebook widget
parameters in a real form, submits eval-set generation runs via the Jobs Runs Submit API,
and renders the scorecard from MLflow + UC tables.

## Project layout

```
app/
├── app.yaml                  # Databricks Apps config (uvicorn entrypoint)
├── server/
│   ├── main.py               # FastAPI app + static mount
│   ├── databricks_client.py  # Auth + REST helpers
│   ├── requirements.txt
│   └── static/               # Built frontend (created by `npm run build`)
└── web/
    ├── package.json
    ├── vite.config.ts        # Build output → ../server/static
    ├── tailwind.config.js
    └── src/
        ├── main.tsx
        ├── App.tsx
        ├── api.ts
        ├── components/ui/    # shadcn-style primitives (hand-written)
        ├── lib/
        └── pages/
            ├── SpacesPage.tsx
            ├── ConfigurePage.tsx
            ├── RunsPage.tsx
            └── ScorecardPage.tsx
```

## Local development

Two terminals.

### Terminal 1 — backend

```bash
cd app/server
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
DATABRICKS_HOST=https://<your-workspace>.cloud.databricks.com \
  uvicorn server.main:app --reload --port 8000 --app-dir ..
```

`server.main` falls back to `databricks auth describe` + `databricks auth
token` if `DATABRICKS_HOST` / `DATABRICKS_TOKEN` are not set, so on a machine
where you've run `databricks auth login` the bare command works too.

**On-behalf-of (OBO) auth.** In production each request carries an
`X-Forwarded-Access-Token` header (the calling user's token). Locally there's no
such header, so set `DEV_USER_TOKEN` to forge one — every request then runs as
that user instead of the service principal:

```bash
DEV_USER_TOKEN=$(databricks auth token -p <your-profile> | jq -r .access_token) \
  uvicorn server.main:app --reload --port 8000 --app-dir ..
```

### Terminal 2 — frontend

```bash
cd app/web
npm install
npm run dev          # http://localhost:5173 (proxies /api → :8000)
```

Open http://localhost:5173 — Vite serves the SPA and proxies API calls to
FastAPI.

## Production build

```bash
cd app/web && npm install && npm run build
# Frontend assets are emitted to ../server/static/.
cd ../server
DATABRICKS_HOST=... uvicorn server.main:app --port 8000 --app-dir ..
```

A single uvicorn process now serves both API and the SPA on `:8000`.

## Databricks Apps deploy

`app/app.yaml` runs `uvicorn server.main:app --host 0.0.0.0 --port 8000`. Build
the frontend first so `server/static/` is populated, then sync the `app/`
directory to a workspace path and deploy:

```bash
cd app/web && npm run build && cd ..
databricks sync . /Workspace/Users/<you>/genie-eval-app
databricks apps deploy genie-eval-app \
  --source-code-path /Workspace/Users/<you>/genie-eval-app
```

`DATABRICKS_HOST` / `DATABRICKS_TOKEN` are auto-injected by the Apps runtime
(service principal OAuth).

## API surface

| Method | Path | Notes |
|---|---|---|
| GET | `/api/health` | `{ host, authed, notebook_path, obo_dev_fallback }` |
| GET | `/api/me` | OBO-resolved `{ user_name }` (defaults the experiment path) |
| GET | `/api/spaces` | Genie spaces list |
| GET | `/api/spaces/{id}` | Full space detail |
| GET | `/api/spaces/{id}/curated-count` | `{ count }` |
| GET | `/api/spaces/{id}/conversation-count` | `{ count }` |
| POST | `/api/runs` | Submit notebook run via `/api/2.2/jobs/runs/submit` |
| GET | `/api/runs/{id}` | Single run state |
| GET | `/api/runs?limit=25` | Recent workspace runs |
| GET | `/api/scorecard/{run_id}?experiment_path=...` | Shaped scorecard JSON |
| GET | `/api/eval-set?catalog=&schema=&limit=200` | Eval-set rows from `genie_eval_set` UC table |
| GET | `/api/eval-runs-index?catalog=&limit=50` | Submitted-runs index from the UC runs-index table |

## Tech stack

- **Backend** — FastAPI, requests, pydantic
- **Frontend** — React 18, TypeScript, Vite, Tailwind CSS,
  shadcn-style hand-written primitives over `@radix-ui/react-tabs`,
  React Router, TanStack Query, Zustand
