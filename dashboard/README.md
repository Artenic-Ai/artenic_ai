# Artenic AI Dashboard

**[Live Demo &rarr; ai.artenic.ch](https://ai.artenic.ch)**

Admin UI for the Artenic AI ML platform. Built with React 19, Vite 7, and Tailwind CSS 4.

## Stack

| Technology | Version | Role |
|------------|---------|------|
| React | 19 | UI framework |
| Vite | 7 | Build tool + dev server |
| Tailwind CSS | 4 | Styling (`@theme` design tokens) |
| TypeScript | 5.7 | Strict mode + `noUncheckedIndexedAccess` |
| Recharts | 3 | Charts (area, bar, line) |
| TanStack React Query | 5 | Server state management |
| Lucide React | — | Icons |
| React Router | 7 | Client-side routing |
| Vitest | 4 | Testing framework |
| @testing-library/react | 16 | Component testing |

## Quick Start

```bash
npm install
npm run dev          # Dev server on http://localhost:5173
npm run build        # Production build (tsc + vite build)
npm test             # Run tests
npm run test:watch   # Watch mode
npm run test:coverage # Coverage report
```

## Demo Mode

The dashboard runs in **demo mode** by default (`VITE_DEMO_MODE=true`), serving realistic
mock data without any backend. This is what powers the live demo at
**[ai.artenic.ch](https://ai.artenic.ch)**.

To connect to a running platform server:

```bash
VITE_DEMO_MODE=false VITE_API_URL=http://localhost:9000 npm run dev
```

Demo mode is transparent: the `apiFetch` client in `lib/api-client.ts` routes requests
to mock handlers when demo mode is enabled. All hooks and components work identically
in both modes.

### Mock Data

| Entity | Count | Details |
|--------|-------|---------|
| Models | 8 | Various frameworks (LightGBM, XGBoost, PyTorch, sklearn) |
| Training Jobs | 12 | Mixed statuses (running, completed, failed, queued) |
| Ensembles | 3 | Different strategies (weighted average, stacking, voting) |
| A/B Tests | 3 | Active, concluded, paused |
| Budget Rules | 4 | Global, per-service, per-provider |
| Activity Events | 10 | Recent platform activity feed |
| Health Reports | 4 | Healthy, degraded, unhealthy models |

## Pages

| Page | Route | Description |
|------|-------|-------------|
| Overview | `/` | KPI cards, activity feed, training status, budget spending |
| Models | `/models` | Model registry with search, detail view with metadata |
| Training | `/training` | Job list with status badges, detail view with logs |
| Inference | `/inference` | Interactive JSON editor, model selector, response viewer |
| Ensembles | `/ensembles` | Ensemble list + detail with member models |
| A/B Tests | `/ab-tests` | Test list + detail with variant metrics |
| Budgets | `/budgets` | Rules management, spending charts |
| Settings | `/settings` | Schema-driven config editor with section navigation |
| Health | `/health` | Model health table with drift/error/latency indicators |

## Architecture

```
dashboard/
├── src/
│   ├── components/
│   │   └── ui/           # Shared UI: Button, Badge, Card, DataTable, Dialog, ...
│   ├── hooks/            # React Query hooks (useModels, useTraining, ...)
│   ├── lib/
│   │   ├── api-client.ts # apiFetch — demo/real mode toggle
│   │   ├── query-keys.ts # Query key factory
│   │   ├── constants.ts  # Shared constants (chart series, etc.)
│   │   └── utils.ts      # Formatting helpers
│   ├── mocks/
│   │   ├── data.ts       # All mock entities
│   │   └── handlers.ts   # Request routing for demo mode
│   ├── pages/            # One folder per page (list + detail)
│   ├── types/
│   │   └── api.ts        # TypeScript interfaces for all API responses
│   ├── App.tsx           # Router + lazy loading
│   ├── main.tsx          # Entry point (QueryClient, ToastContainer)
│   └── index.css         # Tailwind + @theme tokens
├── index.html
├── vite.config.ts
├── vitest.config.ts
├── tsconfig.json
└── package.json
```

## Dark Theme

Always-dark design inspired by Grafana and Vercel. Colors are defined as semantic tokens
in `index.css` using Tailwind 4 `@theme` syntax:

- `--color-surface-0` / `surface-1` / `surface-2` — background layers
- `--color-border` — subtle borders
- `--color-text-primary` / `text-secondary` / `text-muted` — text hierarchy
- `--color-accent` — primary action color
- `--color-chart-1` through `--color-chart-5` — chart palette

## Testing

- **Framework**: Vitest 4 + @testing-library/react 16 + jsdom
- **Tests**: 51 across 6 test suites
- **Config**: `vitest.config.ts` with `define: { "import.meta.env.VITE_DEMO_MODE": JSON.stringify("true") }`
- **CI**: runs `npm test` in the Dashboard job

## License

Apache License 2.0 — see [LICENSE](../LICENSE) for details.
