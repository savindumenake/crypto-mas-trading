import json
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import ConfigLoader
from src.database.db_manager import SegregatedDatabaseManager


class PaperTradeLifecycleManager:
    """
    Manages the full lifecycle of Paper/Simulated Trades across segregated horizon SQLite DBs
    (`scalp`, `intraday`, `swing` strictly isolated):
      1. Opens approved setups while preventing duplicate concurrent positions per (symbol, horizon).
      2. Tracks running MAE and MFE in ATR units on every candle update.
      3. Executes 50% partial take-profit at TP1 and moves Stop-Loss to Breakeven (Entry).
      4. Computes per-horizon isolated PnL (USDT & R-multiple), Account Growth Rate (%),
         Profit Factor, Max Drawdown (%), and Equity Curve series for the UI Dashboard.
    """

    HORIZONS = ("scalp", "intraday", "swing")

    def __init__(
        self,
        config_path: str = "config/assets_config.yaml",
        default_horizon_capital_usdt: float = 1000.0,
    ):
        self.config_loader = ConfigLoader(config_path)
        self.db_manager = SegregatedDatabaseManager(config_path)
        self.default_capital_usdt = float(default_horizon_capital_usdt)
        risk_cfg = self.config_loader.full_config.get("risk_management", {})
        self.base_risk_pct = float(risk_cfg.get("max_risk_per_trade_pct", 1.0)) / 100.0

    @contextmanager
    def _db_conn(self, horizon: str) -> Iterator[sqlite3.Connection]:
        """Resolves the horizon SQLite database connection with Row factory and schema migrations."""
        for method_name in ("get_connection", "_get_connection", "connect", "_connect"):
            if hasattr(self.db_manager, method_name):
                fn = getattr(self.db_manager, method_name)
                with fn(horizon) as conn:
                    conn.row_factory = sqlite3.Row
                    self._ensure_trade_table(conn)
                    yield conn
                return

        db_path = None
        if hasattr(self.db_manager, "get_db_path"):
            db_path = self.db_manager.get_db_path(horizon)
        elif hasattr(self.db_manager, "db_paths") and isinstance(self.db_manager.db_paths, dict):
            db_path = self.db_manager.db_paths.get(horizon)
        else:
            db_path = self.config_loader.get_db_path(horizon)

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            self._ensure_trade_table(conn)
            yield conn
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _ensure_trade_table(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc DATETIME DEFAULT CURRENT_TIMESTAMP,
                closed_timestamp_utc DATETIME,
                symbol TEXT NOT NULL,
                horizon TEXT NOT NULL,
                timeframe TEXT,
                direction TEXT NOT NULL,
                pattern_type TEXT,
                market_regime TEXT,
                confluence_score REAL,
                ml_probability REAL,
                technical_score REAL,
                sentiment_score REAL,
                debate_score REAL,
                entry_price REAL,
                stop_loss REAL,
                tp1_price REAL,
                tp2_price REAL,
                exit_price REAL,
                entry_atr REAL,
                position_size REAL,
                risk_amount_usdt REAL DEFAULT 10.0,
                realized_pnl_r REAL,
                realized_pnl_usdt REAL,
                mae_atr REAL DEFAULT 0.0,
                mfe_atr REAL DEFAULT 0.0,
                feature_vector_json TEXT,
                shap_top_features_json TEXT,
                status TEXT DEFAULT 'OPEN'
            )
            """
        )
        # Safe column migration for existing Sprint 1 databases
        existing_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(trade_memory)").fetchall()
        }
        if "risk_amount_usdt" not in existing_cols:
            conn.execute("ALTER TABLE trade_memory ADD COLUMN risk_amount_usdt REAL DEFAULT 10.0")
        if "realized_pnl_usdt" not in existing_cols:
            conn.execute("ALTER TABLE trade_memory ADD COLUMN realized_pnl_usdt REAL")
        if "closed_timestamp_utc" not in existing_cols:
            conn.execute("ALTER TABLE trade_memory ADD COLUMN closed_timestamp_utc DATETIME")

    def get_active_trades(
        self, horizon: str, symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Returns all currently active trades (`OPEN` or `TP1_HIT`) in the specified horizon DB."""
        with self._db_conn(horizon) as conn:
            if symbol:
                rows = conn.execute(
                    """
                    SELECT * FROM trade_memory
                    WHERE symbol = ? AND status IN ('OPEN', 'TP1_HIT')
                    ORDER BY id ASC
                    """,
                    (symbol,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM trade_memory
                    WHERE status IN ('OPEN', 'TP1_HIT')
                    ORDER BY id ASC
                    """
                ).fetchall()
        return [dict(r) for r in rows]

    def open_paper_trade(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Registers a new paper trade in the horizon's segregated SQLite database if
        `should_execute` is True and no concurrent trade is open for (symbol, horizon).
        """
        if not decision.get("should_execute", False):
            return {
                "opened": False,
                "reason": "SIGNAL_NOT_APPROVED_FOR_EXECUTION",
                "trade_id": None,
            }

        symbol = str(decision["symbol"])
        horizon = str(decision["horizon"])
        existing = self.get_active_trades(horizon=horizon, symbol=symbol)
        if existing:
            return {
                "opened": False,
                "reason": f"DUPLICATE_ACTIVE_TRADE_EXISTS (id={existing[0]['id']}, status={existing[0]['status']})",
                "trade_id": existing[0]["id"],
            }

        rp = decision["risk_plan"]
        comp = decision.get("component_scores", {})
        reports = decision.get("agent_reports", {})
        pat_rep = reports.get("pattern", {})
        ml_rep = reports.get("ml_shap", {})
        sent_rep = reports.get("sentiment", {})

        enriched_features = dict(ml_rep.get("latest_feature_vector", {}))
        enriched_features.update(sent_rep.get("sentiment_memory_snapshot", {}))

        horizon_tf_map = {"scalp": "5m", "intraday": "15m", "swing": "1h"}
        timeframe = horizon_tf_map.get(horizon, "5m")

        entry_price = float(rp["entry_price"])
        stop_loss = float(rp["stop_loss"])
        pos_size = float(rp["position_size_coins"])
        risk_usdt = float(
            rp.get("risk_amount_usdt")
            or round(abs(entry_price - stop_loss) * pos_size, 2)
            or (self.default_capital_usdt * self.base_risk_pct)
        )

        with self._db_conn(horizon) as conn:
            cur = conn.execute(
                """
                INSERT INTO trade_memory (
                    timestamp_utc, symbol, horizon, timeframe, direction,
                    pattern_type, market_regime, confluence_score, ml_probability,
                    technical_score, sentiment_score, debate_score,
                    entry_price, stop_loss, tp1_price, tp2_price,
                    entry_atr, position_size, risk_amount_usdt, mae_atr, mfe_atr,
                    feature_vector_json, shap_top_features_json, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0.0, 0.0, ?, ?, 'OPEN')
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    symbol,
                    horizon,
                    timeframe,
                    str(decision.get("evaluated_bias") or decision["direction"]),
                    str(pat_rep.get("pattern_type", "No_Clear_Pattern")),
                    str(pat_rep.get("market_regime", "Ranging")),
                    float(decision["total_confluence_score"]),
                    float(ml_rep.get("directional_confidence", 0.5)),
                    float(comp.get("pattern_structure_score_25", 0.0)),
                    float(comp.get("sentiment_microstructure_score_15", 0.0)),
                    float(comp.get("debate_memory_score_15", 0.0)),
                    entry_price,
                    stop_loss,
                    float(rp["tp1_price"]),
                    float(rp["tp2_price"]),
                    float(rp["atr_value"]),
                    pos_size,
                    risk_usdt,
                    json.dumps(enriched_features),
                    json.dumps(ml_rep.get("shap_top_features_json", {})),
                ),
            )
            trade_id = int(cur.lastrowid)

        return {
            "opened": True,
            "reason": "PAPER_TRADE_OPENED",
            "trade_id": trade_id,
            "symbol": symbol,
            "horizon": horizon,
            "direction": decision.get("evaluated_bias") or decision["direction"],
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "tp1_price": rp["tp1_price"],
            "tp2_price": rp["tp2_price"],
            "position_size_coins": pos_size,
            "risk_amount_usdt": risk_usdt,
        }

    def update_open_trades(
        self,
        horizon: str,
        symbol: str,
        candle_high: float,
        candle_low: float,
        candle_close: float,
    ) -> List[Dict[str, Any]]:
        """
        Evaluates a new candle (high, low, close) against active trades in `horizon`:
          - Updates running `mae_atr` and `mfe_atr`.
          - Transitions `OPEN` -> `TP1_HIT` (50% partial profit locked + SL moved to Breakeven).
          - Transitions `OPEN` or `TP1_HIT` -> `CLOSED` with exact `realized_pnl_r` and `realized_pnl_usdt`.
        """
        active_trades = self.get_active_trades(horizon=horizon, symbol=symbol)
        events: List[Dict[str, Any]] = []

        for trade in active_trades:
            t_id = int(trade["id"])
            direction = str(trade["direction"])
            status = str(trade["status"])
            entry = float(trade["entry_price"])
            current_sl = float(trade["stop_loss"])
            tp1 = float(trade["tp1_price"])
            tp2 = float(trade["tp2_price"])
            atr = max(float(trade["entry_atr"] or 1.0), 1e-8)
            pos_size = float(trade["position_size"] or 0.0)

            prev_mae = float(trade["mae_atr"] or 0.0)
            prev_mfe = float(trade["mfe_atr"] or 0.0)

            if status == "OPEN" and abs(entry - current_sl) > 1e-8:
                initial_risk_dist = abs(entry - current_sl)
            else:
                initial_risk_dist = atr * 1.5

            risk_usdt = float(
                trade.get("risk_amount_usdt")
                or (initial_risk_dist * pos_size if pos_size > 0 else self.default_capital_usdt * self.base_risk_pct)
            )

            tp1_r = abs(tp1 - entry) / max(initial_risk_dist, 1e-8)
            tp2_r = abs(tp2 - entry) / max(initial_risk_dist, 1e-8)

            if direction == "LONG":
                adverse_move_atr = max(0.0, (entry - candle_low) / atr)
                favorable_move_atr = max(0.0, (candle_high - entry) / atr)
            else:
                adverse_move_atr = max(0.0, (candle_high - entry) / atr)
                favorable_move_atr = max(0.0, (entry - candle_low) / atr)

            new_mae = round(max(prev_mae, adverse_move_atr), 4)
            new_mfe = round(max(prev_mfe, favorable_move_atr), 4)

            new_status = status
            new_sl = current_sl
            exit_price: Optional[float] = None
            realized_pnl_r: Optional[float] = None
            realized_pnl_usdt: Optional[float] = None
            closed_ts: Optional[str] = None
            event_type = "MAE_MFE_UPDATED"

            if direction == "LONG":
                if candle_low <= current_sl:
                    new_status = "CLOSED"
                    exit_price = current_sl
                    closed_ts = datetime.now(timezone.utc).isoformat()
                    if status == "OPEN":
                        realized_pnl_r = -1.0
                        event_type = "STOP_LOSS_HIT"
                    else:
                        realized_pnl_r = round(0.5 * tp1_r, 3)
                        event_type = "BREAKEVEN_STOP_AFTER_TP1"
                elif candle_high >= tp2:
                    new_status = "CLOSED"
                    exit_price = tp2
                    closed_ts = datetime.now(timezone.utc).isoformat()
                    realized_pnl_r = round((0.5 * tp1_r) + (0.5 * tp2_r), 3)
                    event_type = "TP2_FULL_TARGET_HIT"
                elif status == "OPEN" and candle_high >= tp1:
                    new_status = "TP1_HIT"
                    new_sl = entry  # Move Stop-Loss to Breakeven
                    event_type = "TP1_PARTIAL_AND_BREAKEVEN_SET"
            else:
                if candle_high >= current_sl:
                    new_status = "CLOSED"
                    exit_price = current_sl
                    closed_ts = datetime.now(timezone.utc).isoformat()
                    if status == "OPEN":
                        realized_pnl_r = -1.0
                        event_type = "STOP_LOSS_HIT"
                    else:
                        realized_pnl_r = round(0.5 * tp1_r, 3)
                        event_type = "BREAKEVEN_STOP_AFTER_TP1"
                elif candle_low <= tp2:
                    new_status = "CLOSED"
                    exit_price = tp2
                    closed_ts = datetime.now(timezone.utc).isoformat()
                    realized_pnl_r = round((0.5 * tp1_r) + (0.5 * tp2_r), 3)
                    event_type = "TP2_FULL_TARGET_HIT"
                elif status == "OPEN" and candle_low <= tp1:
                    new_status = "TP1_HIT"
                    new_sl = entry  # Move Stop-Loss to Breakeven
                    event_type = "TP1_PARTIAL_AND_BREAKEVEN_SET"

            if realized_pnl_r is not None:
                realized_pnl_usdt = round(realized_pnl_r * risk_usdt, 2)

            with self._db_conn(horizon) as conn:
                conn.execute(
                    """
                    UPDATE trade_memory
                    SET status = ?,
                        stop_loss = ?,
                        exit_price = COALESCE(?, exit_price),
                        realized_pnl_r = COALESCE(?, realized_pnl_r),
                        realized_pnl_usdt = COALESCE(?, realized_pnl_usdt),
                        closed_timestamp_utc = COALESCE(?, closed_timestamp_utc),
                        mae_atr = ?,
                        mfe_atr = ?
                    WHERE id = ?
                    """,
                    (
                        new_status,
                        round(new_sl, 4),
                        round(exit_price, 4) if exit_price is not None else None,
                        realized_pnl_r,
                        realized_pnl_usdt,
                        closed_ts,
                        new_mae,
                        new_mfe,
                        t_id,
                    ),
                )

            events.append(
                {
                    "trade_id": t_id,
                    "symbol": symbol,
                    "horizon": horizon,
                    "direction": direction,
                    "previous_status": status,
                    "new_status": new_status,
                    "event_type": event_type,
                    "stop_loss": round(new_sl, 2),
                    "exit_price": round(exit_price, 2) if exit_price is not None else None,
                    "realized_pnl_r": realized_pnl_r,
                    "realized_pnl_usdt": realized_pnl_usdt,
                    "mae_atr": new_mae,
                    "mfe_atr": new_mfe,
                    "mark_price": round(candle_close, 2),
                }
            )

        return events

    def get_horizon_performance_summary(
        self, horizon: str, initial_capital_usdt: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Computes comprehensive, isolated performance metrics for a single horizon (`scalp`, `intraday`, or `swing`)
        including Net PnL ($ & R), Growth Rate (%), Win Rate (%), Profit Factor, Expectancy,
        Max Drawdown (%), and an Equity Curve series ready for UI charting.
        """
        start_cap = float(initial_capital_usdt or self.default_capital_usdt)

        with self._db_conn(horizon) as conn:
            closed_rows = conn.execute(
                """
                SELECT id, timestamp_utc, closed_timestamp_utc, symbol, direction,
                       pattern_type, confluence_score, realized_pnl_r, realized_pnl_usdt,
                       risk_amount_usdt, mae_atr, mfe_atr
                FROM trade_memory
                WHERE status = 'CLOSED'
                ORDER BY id ASC
                """
            ).fetchall()

            open_cnt = conn.execute(
                "SELECT COUNT(*) as cnt FROM trade_memory WHERE status IN ('OPEN', 'TP1_HIT')"
            ).fetchone()["cnt"]

        total_closed = len(closed_rows)
        wins = 0
        losses = 0
        gross_profit_r = 0.0
        gross_loss_r = 0.0
        net_pnl_usdt = 0.0
        mae_list: List[float] = []
        mfe_list: List[float] = []

        running_equity = start_cap
        peak_equity = start_cap
        max_drawdown_pct = 0.0
        equity_curve: List[Dict[str, Any]] = [
            {"trade_num": 0, "symbol": "START", "pnl_r": 0.0, "pnl_usdt": 0.0, "equity_usdt": round(start_cap, 2)}
        ]

        for idx, r in enumerate(closed_rows, start=1):
            pnl_r = float(r["realized_pnl_r"] or 0.0)
            risk_u = float(r["risk_amount_usdt"] or (start_cap * self.base_risk_pct))
            pnl_u = (
                float(r["realized_pnl_usdt"])
                if r["realized_pnl_usdt"] is not None
                else round(pnl_r * risk_u, 2)
            )

            if pnl_r > 0:
                wins += 1
                gross_profit_r += pnl_r
            elif pnl_r < 0:
                losses += 1
                gross_loss_r += abs(pnl_r)

            net_pnl_usdt += pnl_u
            running_equity += pnl_u
            if running_equity > peak_equity:
                peak_equity = running_equity
            dd_pct = ((peak_equity - running_equity) / max(peak_equity, 1e-8)) * 100.0
            if dd_pct > max_drawdown_pct:
                max_drawdown_pct = dd_pct

            mae_list.append(float(r["mae_atr"] or 0.0))
            mfe_list.append(float(r["mfe_atr"] or 0.0))

            equity_curve.append(
                {
                    "trade_num": idx,
                    "trade_id": int(r["id"]),
                    "symbol": str(r["symbol"]),
                    "direction": str(r["direction"]),
                    "pnl_r": round(pnl_r, 3),
                    "pnl_usdt": round(pnl_u, 2),
                    "equity_usdt": round(running_equity, 2),
                }
            )

        net_r = round(gross_profit_r - gross_loss_r, 3)
        win_rate_pct = round((wins / total_closed) * 100.0, 2) if total_closed > 0 else 0.0
        expectancy_r = round(net_r / total_closed, 3) if total_closed > 0 else 0.0
        profit_factor = (
            round(gross_profit_r / gross_loss_r, 2)
            if gross_loss_r > 0
            else (round(gross_profit_r, 2) if gross_profit_r > 0 else 0.0)
        )
        growth_rate_pct = round(((running_equity - start_cap) / max(start_cap, 1e-8)) * 100.0, 2)

        avg_mae = round(sum(mae_list) / len(mae_list), 3) if mae_list else 0.0
        avg_mfe = round(sum(mfe_list) / len(mfe_list), 3) if mfe_list else 0.0

        return {
            "horizon": horizon,
            "initial_capital_usdt": round(start_cap, 2),
            "current_equity_usdt": round(running_equity, 2),
            "net_pnl_usdt": round(net_pnl_usdt, 2),
            "growth_rate_pct": growth_rate_pct,
            "net_realized_r": net_r,
            "expectancy_r": expectancy_r,
            "profit_factor": profit_factor,
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "active_trades_count": int(open_cnt or 0),
            "closed_trades_count": total_closed,
            "wins_count": wins,
            "losses_count": losses,
            "win_rate_pct": win_rate_pct,
            "avg_mae_atr": avg_mae,
            "avg_mfe_atr": avg_mfe,
            "equity_curve": equity_curve,
        }

    def get_multi_horizon_dashboard_payload(
        self, initial_capital_per_horizon_usdt: float = 1000.0
    ) -> Dict[str, Any]:
        """
        Returns a complete UI-ready payload with strictly segregated analytics for
        `scalp`, `intraday`, and `swing` horizons, plus an aggregate portfolio overview.
        """
        horizons_data: Dict[str, Any] = {}
        total_start = 0.0
        total_equity = 0.0
        total_closed = 0
        total_active = 0

        for h in self.HORIZONS:
            h_summary = self.get_horizon_performance_summary(
                horizon=h, initial_capital_usdt=initial_capital_per_horizon_usdt
            )
            horizons_data[h] = h_summary
            total_start += h_summary["initial_capital_usdt"]
            total_equity += h_summary["current_equity_usdt"]
            total_closed += h_summary["closed_trades_count"]
            total_active += h_summary["active_trades_count"]

        portfolio_growth_pct = round(
            ((total_equity - total_start) / max(total_start, 1e-8)) * 100.0, 2
        )

        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "horizons": horizons_data,  # Strictly separated: horizons['scalp'], horizons['intraday'], horizons['swing']
            "portfolio_Rollup": {
                "total_allocated_capital_usdt": round(total_start, 2),
                "total_current_equity_usdt": round(total_equity, 2),
                "total_net_pnl_usdt": round(total_equity - total_start, 2),
                "portfolio_growth_rate_pct": portfolio_growth_pct,
                "total_active_trades": total_active,
                "total_closed_trades": total_closed,
            },
        }


def _demo_paper_lifecycle():
    manager = PaperTradeLifecycleManager(default_horizon_capital_usdt=1000.0)

    mock_decision = {
        "symbol": "DEMO_BTC",
        "horizon": "scalp",
        "direction": "LONG",
        "evaluated_bias": "LONG",
        "should_execute": True,
        "total_confluence_score": 82.5,
        "component_scores": {
            "ml_shap_score_45": 36.0,
            "pattern_structure_score_25": 22.0,
            "sentiment_microstructure_score_15": 12.5,
            "debate_memory_score_15": 12.0,
        },
        "risk_plan": {
            "entry_price": 90000.0,
            "atr_value": 100.0,
            "stop_loss": 89850.0,
            "tp1_price": 90180.0,
            "tp2_price": 90320.0,
            "position_size_coins": 0.0667,
            "risk_amount_usdt": 10.0,
        },
        "agent_reports": {
            "pattern": {"pattern_type": "Volatility_Squeeze_Breakout_Bull", "market_regime": "Trending_Bull"},
            "ml_shap": {"directional_confidence": 0.68, "latest_feature_vector": {"rsi_14": 58.0}, "shap_top_features_json": {}},
            "sentiment": {"sentiment_memory_snapshot": {"funding_regime": "HEALTHY_BULL_FUNDING"}},
        },
    }

    open_res = manager.open_paper_trade(mock_decision)
    print("1. Open Trade Result:", open_res)

    step1 = manager.update_open_trades(
        horizon="scalp", symbol="DEMO_BTC", candle_high=90200.0, candle_low=89950.0, candle_close=90190.0
    )
    print("2. Candle 1 (TP1 Hit & SL moved to Breakeven):", step1)

    step2 = manager.update_open_trades(
        horizon="scalp", symbol="DEMO_BTC", candle_high=90350.0, candle_low=90150.0, candle_close=90330.0
    )
    print("3. Candle 2 (TP2 Full Target Hit & Trade CLOSED):", step2)

    dashboard_data = manager.get_multi_horizon_dashboard_payload(initial_capital_per_horizon_usdt=1000.0)
    print("\n=======================================================")
    print(" MULTI-HORIZON ISOLATED PERFORMANCE SUMMARY (UI READY)")
    print("=======================================================")
    for h_name, stats in dashboard_data["horizons"].items():
        print(
            f" [{h_name.upper():8s}] Equity: ${stats['current_equity_usdt']:,.2f} | "
            f"PnL: ${stats['net_pnl_usdt']:+,.2f} ({stats['net_realized_r']:+.2f}R) | "
            f"Growth: {stats['growth_rate_pct']:+.2f}% | WinRate: {stats['win_rate_pct']}% | "
            f"PF: {stats['profit_factor']} | MaxDD: {stats['max_drawdown_pct']}%"
        )
    print("=======================================================\n")


if __name__ == "__main__":
    _demo_paper_lifecycle()