# QIDS Deployment Guide

QIDS — Quantum-Assisted Intelligent Detection & Defense System. A
security platform that lets an authorized application send security
telemetry to QIDS, where it is analyzed by classical detection, optional
quantum verification, risk assessment, automated defense, and incident
tracking.

**Not claimed**: support for arbitrary third-party websites. Only the
existing agent-driven / application-security telemetry paths built and
tested in this repository are real.

## Architecture

```
Vercel (React frontend)
      |  HTTPS
      v
QIDS API + Workers (backend PaaS)
      |
      v
Managed Redis
      |
      v
Detection -> Quantum -> Risk -> Defense -> Incident
```

The ML/quantum backend (detection, QSVM/VQC, risk, defense, incident
engines) stays on the backend platform. It is never moved to Vercel --
Vercel serves the static frontend only.

## 1. Local development (unchanged)

```
pip install -r requirements.txt
cd frontend && npm install && cd ..
python run.py
```

Opens the frontend at `http://localhost:5173`, backend at
`http://localhost:8080`. No environment variables are required for local
development -- every new production knob defaults to today's existing
local behavior when unset.

## 2. Managed Redis setup

Any Redis-compatible managed provider works (Upstash, Redis Cloud, Render
Redis, etc.). After provisioning:

1. Copy the provider's connection URL (usually `rediss://default:<password>@<host>:<port>`).
2. Set it as `REDIS_URL` on the backend deployment (see step 3). Do
   **not** also set `REDIS_HOST`/`REDIS_PORT` -- `REDIS_URL` takes
   precedence when present.

## 3. Backend deployment (Render / Railway / Fly / any Python PaaS)

1. Deploy this repository's root as the backend service.
2. Install command: `pip install -r requirements.txt`
3. Start command:
   ```
   gunicorn wsgi:app --workers 2 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT
   ```
   (A `Procfile` with this exact command is included for platforms that
   read one automatically.)
4. Set environment variables (see `.env.example` for the full list):
   - `REDIS_URL` — from step 2
   - `CORS_ALLOWED_ORIGINS` — your Vercel frontend URL(s), comma-separated
   - `EVENT_STORE_BACKEND=redis` — **recommended** if your platform's
     filesystem is ephemeral (most PaaS free/starter tiers are); keeps
     incident history across restarts/redeploys. Leave as `jsonl`
     (default) only if you have a persistent disk mounted.
5. **Workers**: the 5 distributed Redis-stream workers
   (`detection_worker`, `quantum_worker`, `risk_worker`,
   `defense_worker`, `incident_worker`) are optional for the API's own
   experiment execution today (the API runs experiments synchronously,
   in-process — see `src/api/services/experiment_service.py`'s own
   docstring). Deploy them as separate background/worker processes only
   if you want the distributed Phase 7 pipeline running too (the
   `Procfile` includes process types for each, for platforms that
   support multiple process types per app).
6. Verify: `curl https://<your-backend>/api/health` and
   `curl https://<your-backend>/api/ready`.

## 4. Frontend deployment (Vercel)

1. Import this repository into Vercel, set **Root Directory** to `frontend/`.
2. Vercel auto-detects the `vercel.json` (`buildCommand: npm run build`,
   `outputDirectory: dist`, `framework: vite`).
3. Set environment variable: `VITE_API_URL=https://<your-backend-url>`
   (no trailing slash). This must be set **before** the build runs — it
   is baked into the static JS at build time, not read at runtime.
4. Deploy. The app uses `HashRouter` (`#/incidents`, `#/agents`, etc.),
   so no server-side rewrite rules are needed for client-side routing.

## 5. Verify SSE across domains

Once both are deployed:
```
curl -N https://<your-backend>/api/events
```
should stream `event: ...` / `data: ...` frames. In the deployed
frontend, open the Dashboard page and confirm the "Live Event Stream"
connection indicator shows "Live — connected."

## Known limitations

- The distributed Redis-worker path (Phase 7) is not required for the
  API's own experiment execution; deploying it is optional, for
  demonstrating the full distributed architecture.
- `EVENT_STORE_BACKEND=jsonl` on an ephemeral filesystem will lose
  incident history on every redeploy/restart -- use `redis` in that case.
- Real interaction with the public Suzume deployment remains blocked by
  network egress restrictions in this project's own development
  environment; only the local controlled Suzume-compatible target
  (`tests/support/local_suzume_target.py`) has been verified.
