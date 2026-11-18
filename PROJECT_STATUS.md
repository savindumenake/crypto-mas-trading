# NeuralPulse AI - Current Project Status

**Status date:** 2026-11-18
**Current branch:** `feat/saas-web-platform`
**Project mode:** Research and paper trading only (no live exchange order execution)

## Purpose of This Document

This file gives a new developer or AI assistant enough context to understand the
current state of the repository without assuming that every feature described in
the README is already complete.

## Project Summary

NeuralPulse AI is a multi-agent cryptocurrency market-analysis and paper-trading
system. The current active market is `BTCUSDT`. It uses separate Scalp,
Intraday, and Swing analysis horizons, XGBoost/SHAP predictions, technical
pattern analysis, sentiment and market-microstructure inputs, Bull/Bear debate,
a 100-point confluence engine, risk controls, and isolated SQLite paper-trading
ledgers.

The Python trading backend is implemented and its current automated test suite
passes. A FastAPI API and React/Vite web terminal have also been started, but
the web-platform work is still uncommitted and the frontend dependency install
and production build have not yet been verified.

## Implemented and Working

### Trading and analysis backend

- Market-data ingestion and feature engineering.
- BTCUSDT cached market data for the timeframes used by the scanner.
- Pattern and breakout analysis agent.
- XGBoost model training and prediction agents.
- Horizon-specific XGBoost model bundles for Scalp, Intraday, and Swing.
- SHAP-based feature explanations.
- Sentiment and market-microstructure agent.
- Bull/Bear debate agent.
- Six-node LangGraph orchestration workflow.
- 100-point master confluence engine:
  - ML and SHAP: 45 points
  - Pattern and breakout structure: 25 points
  - Historical memory and sentiment: 15 points
  - Fundamental/debate analysis: 15 points
- Risk gate and trade-veto logic.
- Live multi-horizon scanner.
- Telegram trade and lifecycle alert implementation.

### Paper-trading lifecycle

- Separate SQLite ledgers for `scalp`, `intraday`, and `swing`.
- Simulated paper-trade opening.
- Duplicate-active-trade protection.
- TP1 partial-profit lifecycle.
- Stop-loss movement to breakeven after TP1.
- TP2, stop-loss, and breakeven-close handling.
- MAE/MFE tracking.
- Isolated horizon performance summaries and aggregate portfolio roll-up.
- Equity, realized PnL, growth, win rate, profit factor, expectancy, and drawdown
  calculations.

### FastAPI layer

The API application imports successfully and currently exposes:

- `GET /api/v1/health`
- `GET /api/v1/state`
- `POST /api/v1/scan`
- `POST /api/v1/simulate-demo-trade`
- `POST /api/v1/calculate-risk`
- Swagger documentation at `/docs`

### React frontend prototype

The frontend contains navigation and views for:

1. Overview Dashboard
2. Trading Terminal
3. AI Market Analysis
4. Multi-Horizon Analysis
5. Market Intelligence
6. Trading Signals
7. Portfolio
8. Risk Management
9. Backtesting Lab
10. Trade History
11. AI Research / Explainability
12. Settings

Implemented UI behavior includes backend state loading, manual rescanning,
horizon switching, scanner signal tables, a backend-powered position-size
calculator, simulated trade submission, isolated trade-history views, and
Lightweight Charts candlestick/volume rendering when chart candles are present.

The frontend clearly identifies the system as paper trading and does not expose
backend secrets.

## Automated Verification

The Python test suite was verified on 2026-11-18:

```text
12 passed in 8.59s
```

The tests cover database isolation, ingestion and patterns, ML/SHAP behavior,
LangGraph orchestration, confluence logic, execution lifecycle, scanner behavior,
and alerts.

There are not yet dedicated FastAPI integration tests, React component tests, or
browser end-to-end tests.

## Latest Observed Runtime State

The latest cached scanner state was generated at:

```text
2026-11-18 18:37:53 Asia/Colombo
2026-11-18 13:07:53 UTC
```

The cached BTCUSDT signals were:

| Horizon | Bias | Confluence | Scanner decision |
|---|---:|---:|---|
| Scalp | LONG | 48.35 | No trade - low confluence |
| Intraday | LONG | 29.95 | No trade - low confluence |
| Swing | LONG | 59.09 | No trade - low confluence |

There was one active paper trade in the Intraday ledger:

| Field | Value |
|---|---:|
| Symbol | BTCUSDT |
| Direction | LONG |
| Status | OPEN |
| Entry | 67,412.01 |
| Stop loss | 67,024.965 |
| TP1 | 67,876.464 |
| TP2 | 68,237.706 |
| Position size | 0.02584 BTC |
| Risk amount | 10.00 USDT |
| Entry confluence | 82.4 |

At the time of inspection, there were no closed paper trades. Therefore,
realized PnL, historical win rate, and realized growth were still zero.

## Important: Active Scanner Timeframes

The current scanner implementation uses these hard-coded definitions:

| Horizon | Entry timeframe | Higher-timeframe filter |
|---|---|---|
| Scalp | 5m | 1h |
| Intraday | 15m | 4h |
| Swing | 1h | 4h |

These definitions match the current README and cached scanner output.

However, `config/assets_config.yaml` currently contains different timeframe
settings for Scalp and Swing. This is a known source-of-truth inconsistency and
must be resolved before treating the YAML configuration as authoritative.

## Incomplete or Not Yet Verified

- Frontend packages are declared in `frontend/package.json`, but `node_modules`
  is not installed in the inspected workspace.
- There is no frontend lockfile.
- The Vite production build has not been verified.
- The React application is currently concentrated in one large `App.jsx` file
  and has not yet been split into page/component modules.
- The Backtesting Lab is a labelled placeholder; there is no backtesting API.
- An external news feed and economic calendar are not connected.
- Settings are mainly status/display UI and not a complete persisted preferences
  system.
- Liquidation heatmaps, derivatives analytics, correlations, audit logs,
  configurable alerts, report export, and trading journals are future features.
- ETHUSDT and SOLUSDT are planned assets, not active validated markets.
- There is no live exchange wallet connection or live order execution.
- Deployment, authentication, authorization, rate limiting, production logging,
  and operational monitoring are not complete.

## Known Repository Issues

1. `config/assets_config.yaml` and `LiveMultiHorizonScanner` disagree on some
   horizon timeframes. Choose one authoritative configuration and make all code,
   tests, UI labels, and documentation use it.
2. `.gitignore` contains a malformed existing entry:
   `models/**/*.json.pytest_cache/`.
3. The FastAPI, frontend, README, requirements, and web-template work currently
   exists as uncommitted working-tree changes.
4. Frontend build/install verification is still pending.
5. API and frontend automated tests are still missing.

## Git State at Inspection Time

Tracked files with modifications:

- `.gitignore`
- `README.md`
- `requirements.txt`

Untracked implementation areas:

- `frontend/`
- `src/api/`
- `src/web/`

The latest checked-out commit was:

```text
cd8f0f6 feat(scanner): add LiveMultiHorizonScanner and phase 3 unit tests
```

Do not discard or overwrite the uncommitted web-platform work.

## Recommended Next Priorities

1. Resolve the timeframe/configuration source-of-truth mismatch.
2. Correct `.gitignore` and review the pending working-tree changes.
3. Install frontend dependencies and generate a lockfile.
4. Run and fix the Vite production build.
5. Smoke-test the React application against the FastAPI server.
6. Add FastAPI endpoint tests and frontend/component tests.
7. Split `App.jsx` into maintainable pages and reusable components.
8. Commit the web-platform work in logical, reviewable commits.
9. Only after the above, plan deployment/security work or additional product
   modules.

## Instructions for an AI Assistant

When assisting with this repository:

- Treat the system as paper trading and research software only.
- Do not add or imply live exchange execution unless explicitly requested and
  separately designed with appropriate safeguards.
- Preserve the three horizon ledgers and their isolation.
- Keep backend risk and confluence logic authoritative; do not duplicate it in
  the browser.
- Never expose `.env` values, API keys, Telegram credentials, or exchange
  secrets.
- Do not fabricate prices, news, model results, PnL, or account balances.
- Preserve existing uncommitted work and inspect `git diff` before editing.
- Run the existing Python tests after backend changes.
- Verify the frontend build after frontend changes once dependencies are
  installed.
- Clearly label planned or placeholder features as not implemented.


### Paper Trading and API Safety Limitations

- Demo trades can override scanner risk decisions and scores.
- Demo trades are synthetic and must not be treated as AI-approved signals.
- Risk calculator inputs need additional validation.
- API authentication and authorization are not yet implemented.
- Live exchange order execution is not enabled.
- These limitations must be resolved before production deployment.
