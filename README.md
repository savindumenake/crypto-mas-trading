# NeuralPulse AI - Multi-Horizon Crypto Trading MAS

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange)](https://xgboost.readthedocs.io/)

NeuralPulse is a multi-agent, multi-horizon cryptocurrency trading system for
research and paper trading. It combines technical pattern analysis,
XGBoost/SHAP predictions, market microstructure, historical trade memory,
adversarial debate, and ATR-based risk controls in a LangGraph workflow.

> [!WARNING]
> This project is designed for paper trading and research. It does not provide
> financial advice and should not be used for live trading without independent
> review, testing, and a compliant risk-management process.

## Key Features

- **Three trading horizons**
  - Scalp: 5-minute entries with a 1-hour trend filter
  - Intraday: 15-minute entries with a 4-hour trend filter
  - Swing: 1-hour entries with a 4-hour trend filter
- **Six-node LangGraph workflow**
  - Feature preparation
  - Pattern and breakout analysis
  - XGBoost direction and PPV prediction with SHAP explanations
  - Sentiment and microstructure analysis
  - Bull/bear debate and trade-memory review
  - Risk, veto, and execution gate
- **100-point confluence engine**
  - ML/SHAP: 45 points
  - Pattern and breakout structure: 25 points
  - Historical memory and sentiment: 15 points
  - Fundamental/debate analysis: 15 points
- **Paper-trading lifecycle**
  - Separate scalp, intraday, and swing SQLite ledgers
  - Partial profit, breakeven, stop-loss, and TP2 tracking
- **Telegram alerts**
  - New trade alerts
  - TP1, TP2, breakeven, stop-loss, and MAE/MFE lifecycle updates
- **Web terminal and REST API**
  - Interactive candlestick charts
  - Isolated horizon performance
  - Active and closed trade views
  - Position-size calculator
- **Frontend implementation**
  - React application built with Vite
  - Tailwind CSS for the web terminal interface
  - TradingView Lightweight Charts for interactive financial charts
  - Frontend consumes the existing FastAPI REST API; the trading agents,
    scanner, confluence calculations, and paper-trade lifecycle remain in the backend


## Frontend UI/UX Scope

The frontend is planned as a Binance-inspired, dark-mode cryptocurrency futures-style terminal for **BTCUSDT paper trading and research**. The visual design may resemble a professional perpetual-futures trading interface, but this does **not** mean the current backend supports live futures order execution. The existing project remains a research and paper-trading system.

The frontend is a presentation and interaction layer. It must consume the existing FastAPI backend for scanner state, risk calculations, and simulated trades. It must not reimplement or silently change the Python trading logic, LangGraph workflow, horizon definitions, confluence weights, model behavior, database structure, or existing REST API contracts.

### Main Application Pages

The planned web terminal contains the following 12 pages. A page being included in this UI/UX scope does not by itself mean that its complete backend functionality already exists. The implementation status and required data sources must be shown clearly in the UI.

| No. | Page | Main UI responsibilities | Data / implementation status |
|---:|---|---|---|
| 1 | **Overview Dashboard** | BTCUSDT market snapshot; current scan status; Scalp, Intraday, and Swing summaries; latest signals; active and closed paper trades; horizon performance; risk alerts; system status. | Use fields already returned by `/api/v1/state` and `/api/v1/health`. Do not invent live values when data is missing. |
| 2 | **Futures Trading Terminal** | Binance-inspired BTCUSDT perpetual-style layout; interactive candlestick and volume charts; timeframe selector; price and market summary; paper-trade ticket; active positions; open/closed paper-trade details. Order-book, depth, and recent-trade panels may be shown only when corresponding data is available. | Chart and market data must come from the backend or a configured market-data source. Trade submission must use the existing simulated-trade flow. No live order execution is implied. |
| 3 | **AI Market Analysis** | Six-node workflow status; ML direction and PPV output; pattern/breakout findings; sentiment and microstructure summary; Bull/Bear debate; historical memory review; final risk/veto decision; 100-point confluence breakdown. | Reflect the existing LangGraph and scanner outputs. The UI must preserve the configured scoring weights listed below. |
| 4 | **Multi-Horizon Analysis** | Separate Scalp, Intraday, and Swing views; each horizon's entry timeframe, trend filter, signal, model status, trade history, and performance. | Preserve the original README's timeframe definitions exactly: Scalp 5m entry / 1h trend filter; Intraday 15m entry / 4h trend filter; Swing 1h entry / 4h trend filter. |
| 5 | **AI News & Market Intelligence** | News and event feed layout; sentiment summary; source and timestamp; possible market impact; links to original articles; sentiment and microstructure indicators. | Sentiment analysis is part of the described workflow. A live external news feed, social feed, and economic calendar require configured providers and backend integration; do not display fabricated headlines. |
| 6 | **AI Trading Signals** | Scanned trade candidates; symbol and horizon; Long/Short direction; entry, stop-loss and take-profit levels when supplied; model output; confluence score; risk status; signal time; explanation; paper-trade action. | Use actual scanner state. A displayed signal is an analytical output, not a guaranteed result or financial advice. Actions must remain paper-trading actions. |
| 7 | **Portfolio Management** | Paper-trading balances; per-horizon account summaries; available and allocated balance if supplied; realized/unrealized PnL if supplied; open positions; allocation and performance charts. | Keep Scalp, Intraday, and Swing ledgers logically separate. Display only values returned by the backend; do not imply that this is a connected live exchange wallet. |
| 8 | **Risk Management** | Position-size calculator; notional value; ATR-based stop-loss/take-profit information; risk/reward display; risk checks; veto reason; trade lifecycle risk information. | Connect the existing `POST /api/v1/calculate-risk` endpoint and the risk results in scanner state. Keep the backend's existing risk rules authoritative. |
| 9 | **Backtesting & Strategy Lab** | Proposed workspace for selecting historical periods, horizon, model/strategy configuration, and reviewing returns, win rate, drawdown, and equity curves. | This is a proposed UI/UX module. The current documented REST API does not list a dedicated backtesting endpoint. Do not present backtesting as implemented until a tested backend workflow exists. |
| 10 | **Trade History & Performance** | Active and closed paper trades; filters by horizon and outcome; entry/exit details; TP1, TP2, breakeven, and stop-loss lifecycle; MAE/MFE; per-horizon performance summaries. | Use the existing horizon-specific SQLite ledgers and fields exposed through scanner state. Do not combine the three ledgers in a way that removes horizon identity. |
| 11 | **AI Research & Explainability** | XGBoost direction and PPV results; SHAP explanations and feature importance when available; model bundle status; walk-forward validation metrics; historical trade-memory context; Bull/Bear reasoning summary. | Explain only outputs actually produced by the models and agents. Missing SHAP or model information must be shown as unavailable, not generated as a placeholder explanation. |
| 12 | **Settings & Integrations** | Application preferences; active/planned asset visibility; paper-trading status; notification preferences; LLM provider status; Telegram integration status; chart and display preferences. | Respect `config/assets_config.yaml` and backend environment variables. Secret keys must remain on the backend and must never be exposed to browser code. |

### Confluence Engine UI Representation

The AI Market Analysis page should visualize the existing 100-point confluence engine without changing its scoring logic:

| Confluence component | Weight |
|---|---:|
| ML / SHAP | 45 points |
| Pattern and breakout structure | 25 points |
| Historical memory and sentiment | 15 points |
| Fundamental / debate analysis | 15 points |
| **Total** | **100 points** |

The interface should display the score breakdown, the source of each component, and the final risk/veto result. It must not calculate a different score in the browser.

### Trading Horizon UI Representation

The frontend must use the horizon definitions in this README and in the active project configuration. Do not replace them with different timeframe definitions as part of a visual redesign.

| Horizon | Entry timeframe | Trend filter |
|---|---|---|
| Scalp | 5 minutes | 1 hour |
| Intraday | 15 minutes | 4 hours |
| Swing | 1 hour | 4 hours |

The UI should keep horizon-specific signals, trade history, performance, and database context clearly separated.

### Additional Product Enhancements

The following ideas were recommended to make the web terminal more useful to traders. They are **additional UI/product proposals**, not claims that the current backend already implements them.

| Enhancement | Proposed value | Status / dependency |
|---|---|---|
| **Liquidation Heatmap** | Visualize long/short liquidation concentration zones. | Future integration; requires a reliable liquidation data source and backend processing. |
| **Derivatives Analytics** | Funding rate, open interest, long/short ratio, and liquidation metrics. | Future integration unless these values are already supplied by the configured market-data pipeline. |
| **Market Correlation Matrix** | Compare BTC with ETH, SOL, and other supported assets. | Future multi-asset enhancement; ETHUSDT and SOLUSDT remain planned assets in the default configuration. |
| **Smart Alert Centre** | Manage trade lifecycle alerts, signal alerts, price alerts, and notification preferences. | Telegram trade lifecycle alerts already exist; configurable alert rules and additional alert types are enhancements. |
| **Paper Trading Mode Controls** | Clearly show paper mode, simulated orders, balances, and trade lifecycle. | Paper trading is an existing core project capability; the frontend should make the mode explicit and safe by default. |
| **AI Agent Monitoring** | Show the six workflow nodes, processing state, timestamps, and errors. | UI proposal; detailed per-node telemetry may require additional backend observability fields. |
| **Watchlist Management** | Pin and monitor selected market symbols. | UI proposal; keep BTCUSDT as the active default symbol and respect the asset registry for other symbols. |
| **Custom Chart Layouts** | Save chart timeframe, panel, and indicator preferences. | Frontend enhancement; persistence requires a defined user-preference storage approach. |
| **Multi-Exchange Comparison** | Compare supported market data across exchanges. | Future enhancement. CCXT provides an exchange interface, but Binance is the documented default; do not imply all exchanges are already configured and validated. |
| **Trading Journal** | Add notes, tags, and review comments to paper trades. | Future UI/backend enhancement; keep it separate from the existing trade ledger until implemented. |
| **PDF Performance Reports** | Export horizon-specific performance and trade summaries. | Future enhancement unless a tested report-generation feature is added. |
| **System Health Monitoring** | Display API health, scanner status, and integration status. | The health endpoint already exists; richer monitoring and history may require additional telemetry. |
| **Audit Logs** | Record important user actions and system decisions for traceability. | Future enhancement; do not confuse an audit log with the existing trade database records. |
| **Responsive Mobile Interface** | Support viewing dashboards, signals, risk status, and trade history on mobile/tablet screens. | Frontend design requirement; trading-critical information must remain readable at smaller widths. |

### UI Design Principles

- **Visual direction:** Binance-inspired professional dark trading terminal. Use a distinct NeuralPulse identity; do not copy Binance logos, brand assets, or imply affiliation with Binance.
- **Trading mode:** Futures-style visual presentation for BTCUSDT, with paper trading clearly identified. The interface must not suggest that live futures orders are currently supported.
- **Layout:** Persistent navigation, clear page titles, compact market summary, consistent cards/tables, and a main content area suitable for charts and analysis.
- **Color semantics:** Use consistent positive/negative colors for market movement and PnL, while ensuring text labels and icons also communicate status.
- **Charts:** Use TradingView Lightweight Charts for financial time series. Chart data must be real backend/provider data or clearly labelled sample/demo data.
- **Responsive behaviour:** Desktop-first trading layout with tablet/mobile adaptations. On small screens, prioritize price, horizon, signal, position, and risk information.
- **Data honesty:** Show loading, stale, unavailable, and error states. Include timestamps and data-source labels. Never fabricate prices, news, model outputs, account balances, or trade results.
- **Accessibility:** Provide readable contrast, keyboard-accessible controls, clear focus states, and labels that do not depend on color alone.
- **Safety:** Paper trading remains the default. Do not store exchange secrets in local storage or frontend source files. Keep API credentials on the backend.

### Frontend and Backend Responsibility Boundary

| Responsibility | Owner |
|---|---|
| Page layout, navigation, charts, tables, forms, filters, responsive styling | React frontend |
| Frontend development server and production bundle | Vite |
| Styling utilities and responsive design | Tailwind CSS |
| Candlestick and financial chart rendering | TradingView Lightweight Charts |
| API routes, scanner, market-data ingestion, agents, ML predictions, confluence, risk checks | Existing Python / FastAPI backend |
| LangGraph workflow and six-node execution | Existing backend |
| Trade lifecycle and horizon-specific SQLite ledgers | Existing backend |
| Telegram notifications | Existing backend integration |

The frontend must call the existing API rather than duplicate backend calculations. The browser must not become the source of truth for trade approval, position sizing, risk limits, model predictions, or database state.

### Current API-to-Page Mapping

The following are the currently documented endpoints. This frontend plan does not add or rename backend routes.

| Existing endpoint | Frontend use |
|---|---|
| `GET /api/v1/health` | System health and service status |
| `GET /api/v1/state` | Dashboard, horizon summaries, signals, trade views, and available scanner outputs |
| `POST /api/v1/scan` | Request a scanner cycle / live rescan |
| `POST /api/v1/simulate-demo-trade` | Submit a simulated paper trade from cached state |
| `POST /api/v1/calculate-risk` | Calculate coin size and notional position value |

The existing API documentation remains available at `/docs`. Pages such as the proposed Backtesting Lab, live external News Feed, Liquidation Heatmap, and advanced derivatives panels may need new backend services or data providers in a future phase. They must not be represented as working features until those integrations are implemented and tested.


## Repository Layout

```text
config/                         YAML configuration and asset registry
data/cache/                     Parquet market-data cache and scanner state
data/databases/                 Scalp, intraday, and swing SQLite ledgers
models/BTCUSDT/                 Horizon-specific XGBoost model bundles
src/agents/                     ML, pattern, sentiment, and debate agents
src/api/                        FastAPI application
src/core/                       LangGraph orchestrator and live scanner
src/database/                   SQLAlchemy and SQLite lifecycle models
src/execution/                  Paper-trade lifecycle manager
src/ingestion/                  CCXT data fetching and feature engineering
src/notifications/               Telegram alert formatter and notifier
src/web/templates/              Existing FastAPI web-terminal integration templates
frontend/                       React + Vite frontend application (frontend layer)
frontend/src/                   React pages, layouts, and reusable UI components
frontend/src/pages/              Planned page modules for the 12 web-terminal sections
frontend/src/components/charts/ TradingView Lightweight Charts components
frontend/dist/                  Production frontend build output
tests/                          Unit and integration test suite
```

## Requirements

- Python 3.10 through 3.13
- Windows, Linux, or macOS
- Internet access for live Binance data or optional LLM/Telegram features
- A modern web browser
- Node.js 20.19+ or 22.12+ for the Vite frontend

## Installation

```powershell
cd C:\Users\USER\Desktop\crypto_mas_trading
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Frontend Installation

Create the React + Vite frontend in the project root if the `frontend/` directory
has not already been created:

```bash
npm create vite@latest frontend -- --template react
cd frontend
npm install
npm install tailwindcss @tailwindcss/vite lightweight-charts react-router-dom
```

Configure Tailwind CSS using the Vite plugin in `vite.config.js`:

```javascript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
```

Import Tailwind CSS in the main stylesheet (for example, `src/index.css`):

```css
@import "tailwindcss";
```

The `/api` proxy allows the React application to call the existing FastAPI
routes during local development without changing their paths. Keep all trading
logic, risk calculations, model inference, and database operations in the
Python backend. Do not place exchange API secrets in frontend environment
variables or browser code.

The Lightweight Charts license requires the public application to include the
attribution notice from the package's `NOTICE` file and a link to
[TradingView](https://www.tradingview.com/). Follow the official
[Lightweight Charts documentation](https://tradingview.github.io/lightweight-charts/docs)
when implementing chart components.

## Configuration

The main configuration is [config/assets_config.yaml](config/assets_config.yaml).
It controls:

- Binance as the default exchange
- Paper-trading mode
- Scalp, intraday, and swing databases
- Execution and trend-filter timeframes
- Confluence thresholds and risk weights
- Active and planned assets
- ATR stop-loss and take-profit multipliers

BTCUSDT is active in the default configuration. ETHUSDT and SOLUSDT are marked
as planned assets.

### Environment Variables

Create a `.env` file in the project root to enable optional integrations:

```dotenv
# Telegram notifications
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
TELEGRAM_API_BASE_URL=https://cryptomastrading.savidumenake2002.workers.dev

# Optional LLM debate providers
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
```

API keys are optional. When no LLM key is configured or `use_llm_debate` is
disabled, the deterministic Bull/Bear debate and memory score remain available.
If Telegram credentials are missing, alerts are returned as a dry-run console
payload instead of being sent.

## Running the Web Terminal

Start the API server from the project root:

```powershell
python src/api/server.py
```

The FastAPI backend remains available at:

- Terminal: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Interactive API documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health check: [http://127.0.0.1:8000/api/v1/health](http://127.0.0.1:8000/api/v1/health)

For frontend development, open a second terminal:

```powershell
cd frontend
npm run dev
```

Open the Vite development URL shown in the terminal (by default,
[http://localhost:5173](http://localhost:5173)). Configure the Vite development
server to proxy `/api` requests to `http://127.0.0.1:8000` so the React frontend
uses the existing FastAPI endpoints.

To create a production frontend bundle:

```bash
cd frontend
npm run build
```

Vite writes the production bundle to `frontend/dist/`. The production deployment
must serve this build through the existing web-terminal entry point or the
configured web host. This changes the presentation layer only; the FastAPI
routes and backend trading workflow remain unchanged.

The terminal loads cached state by default. Use **Live Rescan (Binance)** or
`POST /api/v1/scan?fetch_live=true` to request fresh closed candles.

## Running a Scanner Cycle

Run the local scanner with cached data:

```powershell
python src/core/live_scanner.py
```

To fetch live Binance candles and send Telegram alerts:

```python
import asyncio

from src.core.live_scanner import LiveMultiHorizonScanner


async def main():
    scanner = LiveMultiHorizonScanner(account_balance_per_horizon_usdt=1000)
    state = await scanner.run_full_scan_cycle(
        fetch_live=True,
        use_llm_debate=False,
        send_telegram=True,
    )
    print(state["scan_timestamp_utc"])


asyncio.run(main())
```

## Training Models

The included BTCUSDT bundles can be used immediately. To retrain all three
horizon models from historical Binance data:

```powershell
python src/agents/ml_trainer.py
```

Training targets:

| Horizon | Entry timeframe | Forward bars | ATR threshold |
|---|---|---:|---:|
| Scalp | 5m | 4 | 0.30 |
| Intraday | 1h | 4 | 0.45 |
| Swing | 4h | 6 | 0.60 |

The scanner automatically trains a missing horizon model before prediction.
Saved bundles contain the XGBoost classifier, ATR-variation regressor, feature
columns, and walk-forward validation metrics.

## REST API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Web trading terminal |
| `GET` | `/api/v1/health` | Service and Telegram configuration status |
| `GET` | `/api/v1/state` | Full cached or freshly scanned platform state |
| `POST` | `/api/v1/scan` | Run a scan across configured symbols and horizons |
| `POST` | `/api/v1/simulate-demo-trade` | Open a simulated signal from cached state |
| `POST` | `/api/v1/calculate-risk` | Calculate coin size and notional position value |

Example:

```powershell
Invoke-RestMethod -Method Post "http://127.0.0.1:8000/api/v1/scan?fetch_live=false&send_telegram=false"
```

## Testing

Run the full test suite:

```powershell
python -m pytest
```

Run one test module:

```powershell
python -m pytest tests/test_phase3_scanner_and_alerts.py
```

Tests cover database setup, ingestion and patterns, LangGraph execution,
confluence, ML/SHAP behavior, live scanning, and Telegram formatting.

## Data and Persistent State

- Market candles are cached in `data/cache/*.parquet`.
- The latest complete platform snapshot is stored in
  `data/cache/latest_scanner_state.json`.
- Paper-trade records are isolated in
  `data/databases/scalp_trading.db`,
  `data/databases/intraday_trading.db`, and
  `data/databases/swing_trading.db`.
- Model bundles are stored under `models/BTCUSDT/`.

Deleting cached databases or state files resets the corresponding local paper
trading history and cached scanner output.

## Troubleshooting

- **Missing Parquet support:** install `pyarrow`.
- **Missing web server packages:** install `fastapi` and `uvicorn`.
- **No Telegram messages:** verify `TELEGRAM_BOT_TOKEN` and
  `TELEGRAM_CHAT_ID`, then inspect the health endpoint.
- **Live data unavailable:** the scanner can fall back to existing local
  Parquet caches when `fetch_live=false` or when an exchange fetch fails.
- **Missing model bundle:** run the ML trainer or let the live scanner train
  the absent BTCUSDT horizon model.
- **Windows PowerShell blocks activation:** run
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate
  the virtual environment again.
- **Frontend dependencies unavailable:** check the installed Node.js version,
  then run `npm install` from the `frontend/` directory.
- **Frontend cannot reach the API:** confirm that the FastAPI server is running
  on `http://127.0.0.1:8000` and that the Vite `/api` proxy is configured.
- **Charts do not render:** confirm that `lightweight-charts` is installed and
  that the chart container has a measurable width and height.

## Frontend Documentation References

- [Vite Guide](https://vite.dev/guide/) — project setup, development server, and build process.
- [Tailwind CSS with Vite](https://tailwindcss.com/docs/installation/using-vite) — Tailwind CSS Vite-plugin setup.
- [TradingView Lightweight Charts](https://tradingview.github.io/lightweight-charts/docs) — chart installation, API, and attribution requirements.
- [React Documentation](https://react.dev/learn) — components, props, state, and hooks.

## License

No license file is currently included. Add a license before distributing or
deploying this project publicly.
