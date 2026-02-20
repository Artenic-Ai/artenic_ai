# Security Policy

## Supported Versions

Only the **latest minor version** receives security patches.

| Version | Supported          |
|---------|--------------------|
| 0.7.x   | Yes                |
| < 0.7   | No                 |

## Reporting a Vulnerability

If you discover a security issue in Artenic AI, please report it responsibly.

**Do NOT open a public GitHub issue.**

Instead, email us at **contact@artenic.ch** with:

1. A description of the issue
2. Steps to reproduce (if applicable)
3. The affected version(s)
4. Any potential impact assessment

We will acknowledge your report within **48 hours** and aim to provide a fix
or mitigation within **7 days** for confirmed issues.

## Disclosure Policy

- We follow coordinated disclosure: we ask reporters to keep details private
  until a fix is released.
- We will credit reporters in the release notes (unless they prefer anonymity).
- We publish advisories via GitHub Security Advisories once a fix is available.

## Scope

This policy covers the following components:

- `packages/sdk/core/` — Artenic AI SDK (core contracts, schemas, types)
- `packages/sdk/ensemble/` — Artenic AI SDK Ensemble (strategies)
- `packages/sdk/training/` — Artenic AI SDK Training (callbacks, serialization)
- `packages/sdk/client/` — Artenic AI SDK Client (HTTP client)
- `packages/platform/core/` — Artenic AI Platform (gateway, registry, datasets)
- `packages/platform/providers/` — Artenic AI Platform Providers (cloud integrations, catalog)
- `packages/platform/training/` — Artenic AI Platform Training (orchestration, MLflow)
- `packages/cli/` — Artenic AI CLI
- `packages/optimizer/` — Artenic Optimizer
- `dashboard/` — Artenic Dashboard

Third-party dependencies are out of scope, but we appreciate reports about
known issues in our dependency tree.
