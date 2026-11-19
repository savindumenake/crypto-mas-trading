import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.live_scanner import LiveMultiHorizonScanner

app = FastAPI(
    title="NeuralPulse AI Quant Terminal API",
    description="Commercial Multi-Horizon Crypto Multi-Agent SaaS Backend (Scalp 5m | Intraday 15m | Swing 1h)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

scanner = LiveMultiHorizonScanner(account_balance_per_horizon_usdt=1000.0)
TEMPLATE_PATH = PROJECT_ROOT / "src" / "web" / "templates" / "index.html"
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"

app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets", check_dir=False), name="frontend-assets")



class RiskCalcRequest(BaseModel):
    account_balance_usdt: float = Field(
        default=1000.0,
        gt=0,
        allow_inf_nan=False
    )
    risk_pct: float = Field(
        default=1.0,
        gt=0,
        le=100,
        allow_inf_nan=False
    )
    entry_price: float = Field(
        gt=0,
        allow_inf_nan=False
    )
    stop_loss: float = Field(
        gt=0,
        allow_inf_nan=False
    )



def _resolve_horizon_db_path(horizon: str) -> Path:
    """Safely resolves the segregated SQLite database path for a given horizon."""
    db_mgr = getattr(scanner.lifecycle, "db_manager", None)
    if db_mgr is not None:
        for attr in ("db_paths", "databases", "paths"):
            mapping = getattr(db_mgr, attr, None)
            if isinstance(mapping, dict) and horizon in mapping:
                return Path(mapping[horizon])
        for method_name in ("_get_db_path", "get_database_path", "resolve_db_path"):
            if hasattr(db_mgr, method_name):
                return Path(getattr(db_mgr, method_name)(horizon))
    return PROJECT_ROOT / "data" / "databases" / f"{horizon}_trading.db"


def _get_recent_closed_trades(horizon: str, limit: int = 25) -> List[Dict[str, Any]]:
    """Reads recently closed paper trades from the isolated horizon database."""
    db_path = _resolve_horizon_db_path(horizon)
    if not db_path.exists():
        return []
    try:
        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM paper_trades
                WHERE status LIKE 'CLOSED%'
                ORDER BY trade_id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
    except Exception:
        return []


def _enrich_state_with_ledgers(state: Dict[str, Any]) -> Dict[str, Any]:
    """Attaches live isolated performance, active trades, and closed trade history."""
    state["isolated_performance"] = scanner.lifecycle.get_multi_horizon_dashboard_payload(
        initial_capital_per_horizon_usdt=scanner.account_balance_usdt
    )
    state["active_trades_by_horizon"] = {
        h: scanner.lifecycle.get_active_trades(h) for h in ("scalp", "intraday", "swing")
    }
    state["closed_trades_by_horizon"] = {
        h: _get_recent_closed_trades(h, limit=25) for h in ("scalp", "intraday", "swing")
    }
    return state


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_saas_terminal():
    """Serves the built React terminal, falling back to the legacy template."""
    frontend_index = FRONTEND_DIST / "index.html"
    if frontend_index.exists():
        return FileResponse(frontend_index)
    if not TEMPLATE_PATH.exists():
        return HTMLResponse("<h3>Frontend is not built. Run npm run build from frontend/.</h3>", status_code=404)
    return FileResponse(TEMPLATE_PATH)


@app.get("/api/v1/health")
async def health_check() -> Dict[str, Any]:
    return {
        "status": "online",
        "service": "NeuralPulse Multi-Horizon Quant SaaS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "telegram_configured": scanner.notifier.is_configured,
    }


@app.get("/api/v1/state")
async def get_platform_state(force_refresh: bool = Query(False)) -> Dict[str, Any]:
    """Returns the complete multi-horizon state snapshot and isolated PnL metrics."""
    if force_refresh or not scanner.state_cache_path.exists():
        state = await scanner.run_full_scan_cycle(
            fetch_live=False,
            use_llm_debate=False,
            send_telegram=False,
        )
        return _enrich_state_with_ledgers(state)

    try:
        with open(scanner.state_cache_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        return _enrich_state_with_ledgers(state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load scanner state: {exc}")


@app.post("/api/v1/scan")
async def trigger_live_scan(
    fetch_live: bool = Query(False, description="Fetch live candles from Binance if True, else use local Parquet"),
    send_telegram: bool = Query(True, description="Send Telegram alerts on new qualifying trades"),
) -> Dict[str, Any]:
    """Triggers an on-demand multi-horizon scan across all symbols and horizons."""
    state = await scanner.run_full_scan_cycle(
        fetch_live=fetch_live,
        use_llm_debate=False,
        send_telegram=send_telegram,
    )
    return _enrich_state_with_ledgers(state)


@app.post("/api/v1/simulate-demo-trade")
async def simulate_vip_signal(
    horizon: str = Query("scalp", description="Target horizon: scalp, intraday, or swing"),
    symbol: str = Query("BTCUSDT", description="Target symbol"),
    send_telegram: bool = Query(True, description="Dispatch real Telegram alert to channel"),
) -> Dict[str, Any]:
    """Executes a simulated institutional A+ setup on the selected horizon."""
    h_key = horizon.lower()
    if h_key not in ("scalp", "intraday", "swing"):
        raise HTTPException(status_code=400, detail="Invalid horizon. Choose scalp, intraday, or swing.")

    state = await get_platform_state(force_refresh=False)
    sig = state.get("signals_by_horizon", {}).get(h_key, {}).get(symbol)
    if not sig:
        raise HTTPException(status_code=404, detail=f"No cached signal for {symbol} ({h_key})")

    demo_decision = dict(sig)
    rp = dict(demo_decision.get("risk_plan", {}))
    price = float(demo_decision.get("latest_price") or rp.get("entry_price") or 90000.0)
    atr = float(rp.get("atr_value") or (price * 0.004))
    bias = demo_decision.get("evaluated_bias") or "LONG"

    if bias == "LONG":
        sl = round(price - 1.5 * atr, 4)
        tp1 = round(price + 1.8 * atr, 4)
        tp2 = round(price + 3.2 * atr, 4)
    else:
        sl = round(price + 1.5 * atr, 4)
        tp1 = round(price - 1.8 * atr, 4)
        tp2 = round(price - 3.2 * atr, 4)

    rp.update(
        {
            "entry_price": price,
            "stop_loss": sl,
            "tp1_price": tp1,
            "tp2_price": tp2,
            "atr_value": round(atr, 4),
            "sl_atr_mult": 1.5,
            "rr_ratio_tp1": 1.2,
            "rr_ratio_tp2": 2.13,
            "risk_amount_usdt": 10.0,
            "position_size_coins": round(10.0 / max(abs(price - sl), 1e-6), 5),
        }
    )
    demo_decision["risk_plan"] = rp
    demo_decision["direction"] = bias
    demo_decision["evaluated_bias"] = bias
    demo_decision["should_execute"] = True
    demo_decision["total_confluence_score"] = max(float(demo_decision.get("total_confluence_score", 0)), 82.4)
    demo_decision["signal_tier"] = "A_PLUS_INSTITUTIONAL"

    open_res = scanner.lifecycle.open_paper_trade(demo_decision)
    tg_res = {"sent": False, "mode": "SKIPPED"}
    if send_telegram:
        msg = scanner.notifier.format_new_trade_alert(demo_decision, open_res)
        tg_res = scanner.notifier.send_message(msg)

    return {
        "opened_trade": open_res,
        "telegram_dispatch": tg_res,
        "updated_performance": scanner.lifecycle.get_multi_horizon_dashboard_payload(
            initial_capital_per_horizon_usdt=scanner.account_balance_usdt
        ),
        "active_trades": scanner.lifecycle.get_active_trades(h_key),
    }


@app.post("/api/v1/calculate-risk")
async def calculate_custom_position_size(req: RiskCalcRequest) -> Dict[str, Any]:
    """Interactive position sizing calculator for SaaS subscribers."""
    risk_usdt = round(req.account_balance_usdt * (req.risk_pct / 100.0), 2)
    stop_dist = abs(req.entry_price - req.stop_loss)
    if stop_dist <= 0:
        raise HTTPException(status_code=400, detail="Entry and Stop-Loss cannot be identical.")
    size_coins = round(risk_usdt / stop_dist, 6)
    notional_usdt = round(size_coins * req.entry_price, 2)
    implied_leverage = round(notional_usdt / max(req.account_balance_usdt, 1.0), 2)
    return {
        "risk_amount_usdt": risk_usdt,
        "stop_distance_usdt": round(stop_dist, 4),
        "position_size_coins": size_coins,
        "notional_value_usdt": notional_usdt,
        "recommended_max_leverage": max(1, int(round(implied_leverage + 0.5))),
    }


@app.get("/{frontend_path:path}", include_in_schema=False)
async def serve_frontend_routes(frontend_path: str):
    """Allows browser refreshes on client-side React routes after a frontend build."""
    frontend_index = FRONTEND_DIST / "index.html"
    if frontend_index.exists():
        return FileResponse(frontend_index)
    raise HTTPException(status_code=404, detail="Not found")


if __name__ == "__main__":
    print("\n" + "=" * 78)
    print(" 🚀 NEURALPULSE QUANT AI — COMMERCIAL SAAS WEB PLATFORM STARTING")
    print(" 🌐 Live Web Terminal : http://127.0.0.1:8000")
    print(" 📘 Swagger API Docs  : http://127.0.0.1:8000/docs")
    print("=" * 78 + "\n")
    uvicorn.run("src.api.server:app", host="127.0.0.1", port=8000, reload=False)
