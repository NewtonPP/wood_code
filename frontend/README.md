# Wood-chip Monitor — Frontend (React + TypeScript + Vite)

This is the React re-implementation of the original single-file `index.html`. It
preserves the exact UI and data-fetching behavior (same endpoints, same polling
cadence, same RBAC gating, controls drawer, and guided tour).

## Structure

- `src/lib/` — API layer (`api.ts`), RBAC mirror (`rbac.ts`), formatters (`format.ts`)
- `src/context/` — `AuthContext` (session/login/logout), `ThemeContext` (day/night)
- `src/hooks/usePolling.ts` — interval polling with cleanup
- `src/components/` — Nav, ControlsDrawer, Tour, Histogram, MoistureBars, LiveView
- `src/pages/` — Login, Live, Events, Quality, Audit, Devices
- `src/styles/global.css` — ported verbatim from the original `<style>` block

Routing uses `HashRouter`, so URLs stay `#/live`, `#/events`, … exactly like before.

## Development

The backend (FastAPI / uvicorn) must be running on `:8000`.

```bash
npm install
npm run dev        # Vite dev server on http://localhost:5173
```

The dev server proxies `/api` and `/ping` to `http://localhost:8000` so the
session cookie stays same-origin.

## Production build

```bash
npm run build      # type-checks then builds into ../web
```

The build output lands in the repo `web/` directory, which the FastAPI backend
serves from its static mount (`app/main.py`). On the device, build once and start
the backend (`uvicorn backend_app:app`) — the SPA is served at `/`.

## Polling cadence (matches the original)

- `/api/frame?ts=…` every 200 ms (Live)
- `/api/stats` every 1000 ms (header status + Live stats)
- `/api/moisture` every 1200 ms (Live)
- `/api/hist` every 3000 ms (Live)
