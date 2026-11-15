import asyncio
import json
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional
import ccxt.async_support as ccxt
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import ConfigLoader
from src.database.db_manager import SegregatedDatabaseManager


class CryptoSentimentAndMicrostructureAgent:
    """
    Evaluates crypto derivatives sentiment & orderbook microstructure (Max 15.0 Confluence Points).
    Includes:
    1. Live Binance USDT-M Futures Funding, Open Interest & L2 Orderbook Imbalance.
    2. Persistent Post-Shock Cooldown Tracking via Segregated SQLite (`sentiment_cache`).
       Distinguishes chaotic whipsaw/adverse shocks from clean directional breakouts.
    3. Sentiment-Matched Historical Trade Lookup (queries last 15 similar trades
       matching both technical regime AND historical sentiment/funding state).
    """

    COOLDOWN_CONFIG = {
        "scalp": {"window_minutes": 60, "hard_veto_minutes": 30},
        "intraday": {"window_minutes": 180, "hard_veto_minutes": 60},
        "swing": {"window_minutes": 360, "hard_veto_minutes": 120},
    }

    def __init__(self, config_path: str = "config/assets_config.yaml"):
        self.config_loader = ConfigLoader(config_path)
        self.db_manager = SegregatedDatabaseManager(config_path)
        self._exchange: Optional[ccxt.binanceusdm] = None

    @staticmethod
    def _ensure_tables(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sentiment_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT NOT NULL,
                funding_rate REAL,
                open_interest_change_pct REAL,
                long_short_ratio REAL,
                social_sentiment_score REAL,
                news_shock_active INTEGER DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc DATETIME DEFAULT CURRENT_TIMESTAMP,
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
                realized_pnl_r REAL,
                mae_atr REAL,
                mfe_atr REAL,
                feature_vector_json TEXT,
                shap_top_features_json TEXT,
                status TEXT DEFAULT 'OPEN'
            )
            """
        )

    @contextmanager
    def _db_conn(self, horizon: str) -> Iterator[sqlite3.Connection]:
        """
        Safely resolves the horizon SQLite connection and ensures required tables exist.
        """
        for method_name in ("get_connection", "_get_connection", "connect", "_connect"):
            if hasattr(self.db_manager, method_name):
                fn = getattr(self.db_manager, method_name)
                with fn(horizon) as conn:
                    conn.row_factory = sqlite3.Row
                    self._ensure_tables(conn)
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
            self._ensure_tables(conn)
            yield conn
            conn.commit()
        finally:
            conn.close()

    async def _get_exchange(self) -> ccxt.binanceusdm:
        if self._exchange is None:
            self._exchange = ccxt.binanceusdm(
                {"enableRateLimit": True, "options": {"defaultType": "future"}}
            )
        return self._exchange

    async def close_exchange(self) -> None:
        if self._exchange is not None:
            await self._exchange.close()
            self._exchange = None

    async def fetch_live_microstructure(self, symbol: str) -> Dict[str, float]:
        """
        Fetches live Funding Rate, Open Interest, and L2 Orderbook Bid/Ask Imbalance
        from Binance USDT-M Perpetual Futures.
        """
        exchange = await self._get_exchange()
        ccxt_symbol = (
            f"{symbol[:-4]}/USDT:USDT"
            if symbol.endswith("USDT") and "/" not in symbol
            else symbol
        )

        funding_rate = 0.0
        open_interest = 0.0
        orderbook_imbalance = 0.0
        spread_bps = 1.0

        try:
            funding_info = await exchange.fetch_funding_rate(ccxt_symbol)
            funding_rate = float(funding_info.get("fundingRate") or 0.0)
        except Exception:
            funding_rate = 0.0

        try:
            oi_info = await exchange.fetch_open_interest(ccxt_symbol)
            open_interest = float(
                oi_info.get("openInterestAmount")
                or oi_info.get("openInterestValue")
                or 0.0
            )
        except Exception:
            open_interest = 0.0

        try:
            ob = await exchange.fetch_order_book(ccxt_symbol, limit=20)
            bids = ob.get("bids", [])
            asks = ob.get("asks", [])
            bid_vol = sum(float(b[1]) for b in bids[:20]) if bids else 0.0
            ask_vol = sum(float(a[1]) for a in asks[:20]) if asks else 0.0
            total_vol = bid_vol + ask_vol
            if total_vol > 0:
                orderbook_imbalance = (bid_vol - ask_vol) / total_vol
            if bids and asks and bids[0][0] > 0:
                spread_bps = ((asks[0][0] - bids[0][0]) / bids[0][0]) * 10000.0
        except Exception:
            orderbook_imbalance = 0.0

        return {
            "funding_rate": round(funding_rate, 6),
            "open_interest": round(open_interest, 2),
            "orderbook_imbalance": round(orderbook_imbalance, 4),
            "spread_bps": round(spread_bps, 3),
        }

    # -----------------------------------------------------------------
    # 1. PERSISTENT SHOCK COOLDOWN TRACKING (SQLITE SENTIMENT_CACHE)
    # -----------------------------------------------------------------
    def record_sentiment_snapshot(
        self,
        horizon: str,
        symbol: str,
        funding_rate: float,
        open_interest: float,
        orderbook_imbalance: float,
        social_sentiment: float,
        news_shock_active: bool,
    ) -> None:
        """Persists live derivatives & shock status into the horizon's SQLite DB."""
        with self._db_conn(horizon) as conn:
            conn.execute(
                """
                INSERT INTO sentiment_cache (
                    symbol, funding_rate, open_interest_change_pct,
                    long_short_ratio, social_sentiment_score, news_shock_active
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    float(funding_rate),
                    float(open_interest),
                    float(orderbook_imbalance),
                    float(social_sentiment),
                    1 if news_shock_active else 0,
                ),
            )

    def clear_symbol_shocks(self, horizon: str, symbol: str) -> None:
        """Clears active shock flags for a symbol in the horizon DB (useful for reset/tests)."""
        with self._db_conn(horizon) as conn:
            conn.execute(
                "UPDATE sentiment_cache SET news_shock_active = 0 WHERE symbol = ?",
                (symbol,),
            )

    def get_post_shock_cooldown_state(
        self, horizon: str, symbol: str
    ) -> Dict[str, Any]:
        """
        Checks the horizon SQLite database for any news/volatility shock recorded
        within the horizon's cooldown window and computes a time-decay stability factor.
        """
        cfg = self.COOLDOWN_CONFIG.get(horizon, self.COOLDOWN_CONFIG["scalp"])
        window_mins = cfg["window_minutes"]
        veto_mins = cfg["hard_veto_minutes"]

        with self._db_conn(horizon) as conn:
            row = conn.execute(
                """
                SELECT timestamp_utc
                FROM sentiment_cache
                WHERE symbol = ? AND news_shock_active = 1
                ORDER BY id DESC LIMIT 1
                """,
                (symbol,),
            ).fetchone()

        if not row or not row["timestamp_utc"]:
            return {
                "in_cooldown_window": False,
                "hard_veto_active": False,
                "minutes_since_shock": None,
                "cooldown_stability_factor": 1.0,
            }

        ts_str = str(row["timestamp_utc"])
        try:
            shock_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if shock_dt.tzinfo is None:
                shock_dt = shock_dt.replace(tzinfo=timezone.utc)
        except ValueError:
            shock_dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc
            )

        now_utc = datetime.now(timezone.utc)
        elapsed_mins = max(0.0, (now_utc - shock_dt).total_seconds() / 60.0)

        if elapsed_mins <= veto_mins:
            return {
                "in_cooldown_window": True,
                "hard_veto_active": True,
                "minutes_since_shock": round(elapsed_mins, 1),
                "cooldown_stability_factor": 0.0,
            }
        elif elapsed_mins < window_mins:
            recovery_span = max(1.0, float(window_mins - veto_mins))
            factor = (elapsed_mins - veto_mins) / recovery_span
            return {
                "in_cooldown_window": True,
                "hard_veto_active": False,
                "minutes_since_shock": round(elapsed_mins, 1),
                "cooldown_stability_factor": round(float(np.clip(factor, 0.1, 1.0)), 3),
            }

        return {
            "in_cooldown_window": False,
            "hard_veto_active": False,
            "minutes_since_shock": round(elapsed_mins, 1),
            "cooldown_stability_factor": 1.0,
        }

    # -----------------------------------------------------------------
    # 2. SENTIMENT-MATCHED 15 HISTORICAL TRADES LOOKUP
    # -----------------------------------------------------------------
    def get_sentiment_matched_historical_trades(
        self,
        horizon: str,
        symbol: str,
        pattern_type: str,
        market_regime: str,
        current_sentiment_score: float,
        current_funding_regime: str,
        top_k: int = 15,
    ) -> Dict[str, Any]:
        """
        Queries the last closed trades from the horizon DB and ranks the top `top_k` (15)
        historical setups that match both the technical pattern/regime AND similar
        sentiment/funding conditions.
        """
        with self._db_conn(horizon) as conn:
            rows = conn.execute(
                """
                SELECT id, direction, pattern_type, market_regime, sentiment_score,
                       realized_pnl_r, mae_atr, mfe_atr, feature_vector_json
                FROM trade_memory
                WHERE symbol = ? AND status = 'CLOSED'
                ORDER BY id DESC LIMIT 60
                """,
                (symbol,),
            ).fetchall()

        if not rows:
            return {
                "matched_trades_count": 0,
                "sentiment_matched_win_rate": 0.50,
                "avg_historical_sentiment": 0.0,
                "sentiment_similarity_confidence": 0.0,
            }

        scored_matches = []
        for r in rows:
            hist_sent = float(r["sentiment_score"] or 7.5)
            sent_diff = abs(current_sentiment_score - hist_sent)
            sent_sim = max(0.0, 1.0 - (sent_diff / 15.0))

            pat_bonus = 0.25 if r["pattern_type"] == pattern_type else 0.0
            reg_bonus = 0.15 if r["market_regime"] == market_regime else 0.0

            fund_bonus = 0.0
            try:
                fv = json.loads(r["feature_vector_json"] or "{}")
                if fv.get("funding_regime") == current_funding_regime:
                    fund_bonus = 0.20
            except Exception:
                pass

            total_sim = min(1.0, (sent_sim * 0.60) + pat_bonus + reg_bonus + fund_bonus)
            scored_matches.append((total_sim, r))

        scored_matches.sort(key=lambda x: x[0], reverse=True)
        selected = [item[1] for item in scored_matches[:top_k]]
        sims = [item[0] for item in scored_matches[:top_k]]

        wins = sum(1 for r in selected if float(r["realized_pnl_r"] or 0.0) > 0)
        win_rate = wins / len(selected)
        avg_sent = float(np.mean([float(r["sentiment_score"] or 7.5) for r in selected]))

        return {
            "matched_trades_count": len(selected),
            "sentiment_matched_win_rate": round(win_rate, 4),
            "avg_historical_sentiment": round(avg_sent, 2),
            "sentiment_similarity_confidence": round(float(np.mean(sims)), 3),
        }

    # -----------------------------------------------------------------
    # 3. MAIN 15-POINT SENTIMENT CONFLUENCE EVALUATOR
    # -----------------------------------------------------------------
    def evaluate_sentiment_confluence(
        self,
        symbol: str,
        target_direction: str,
        horizon: str = "scalp",
        df: Optional[pd.DataFrame] = None,
        microstructure: Optional[Dict[str, float]] = None,
        external_news_sentiment: float = 0.0,
        news_shock_flag: bool = False,
        max_points: float = 15.0,
    ) -> Dict[str, Any]:
        """
        Computes the 15-point Sentiment & Microstructure Confluence Score and updates
        persistent cooldown state in SQLite.
        """
        if microstructure is None:
            if df is not None and not df.empty:
                last = df.iloc[-1]
                ob_proxy = float(
                    np.clip(
                        (last.get("close", 1.0) - last.get("open", 1.0))
                        / (last.get("atr_14", 1.0) + 1e-8)
                        * 0.25,
                        -0.6,
                        0.6,
                    )
                )
                microstructure = {
                    "funding_rate": 0.0001,
                    "open_interest": float(last.get("volume", 0.0)),
                    "orderbook_imbalance": round(ob_proxy, 4),
                    "spread_bps": 1.2,
                }
            else:
                microstructure = {
                    "funding_rate": 0.0001,
                    "open_interest": 0.0,
                    "orderbook_imbalance": 0.0,
                    "spread_bps": 1.0,
                }

        funding = float(microstructure.get("funding_rate", 0.0))
        oi_val = float(microstructure.get("open_interest", 0.0))
        ob_imb = float(microstructure.get("orderbook_imbalance", 0.0))
        spread_bps = float(microstructure.get("spread_bps", 1.0))

        # Detect anomalous volatility shock vs. clean directional breakout
        detected_shock = bool(news_shock_flag)
        if df is not None and len(df) >= 20:
            recent_ranges = (df["high"] - df["low"]).tail(20)
            median_range = float(recent_ranges.iloc[:-1].median()) + 1e-8
            latest_range = float(recent_ranges.iloc[-1])

            if latest_range > (median_range * 4.2):
                last_open = float(df["open"].iloc[-1])
                last_close = float(df["close"].iloc[-1])
                body_size = abs(last_close - last_open)
                body_ratio = body_size / (latest_range + 1e-8)

                is_bull_bar = last_close > last_open
                aligned_move = (target_direction == "LONG" and is_bull_bar) or (
                    target_direction == "SHORT" and not is_bull_bar
                )

                # Clean directional breakout (body >= 45% in trade direction) is NOT a shock veto
                if not (aligned_move and body_ratio >= 0.45):
                    detected_shock = True

        self.record_sentiment_snapshot(
            horizon=horizon,
            symbol=symbol,
            funding_rate=funding,
            open_interest=oi_val,
            orderbook_imbalance=ob_imb,
            social_sentiment=external_news_sentiment,
            news_shock_active=detected_shock,
        )
        cooldown_state = self.get_post_shock_cooldown_state(horizon, symbol)

        # Component A: Orderbook Imbalance Alignment (0.0 to 6.0 pts)
        aligned_ob = ob_imb if target_direction == "LONG" else -ob_imb
        ob_score = float(np.clip((aligned_ob + 0.30) / 0.60, 0.0, 1.0) * 6.0)
        if spread_bps > 8.0:
            ob_score *= 0.5

        # Component B: Derivatives Funding & Crowding Filter (0.0 to 5.0 pts)
        funding_regime = "NEUTRAL_POSITIONING"
        if target_direction == "LONG":
            if funding < -0.0002:
                funding_score = 5.0
                funding_regime = "SHORT_SQUEEZE_FUEL"
            elif funding <= 0.0003:
                funding_score = 4.0
                funding_regime = "HEALTHY_BULL_FUNDING"
            elif funding <= 0.0008:
                funding_score = 2.2
                funding_regime = "ELEVATED_LONG_FUNDING"
            else:
                funding_score = 0.5
                funding_regime = "CROWDED_LONG_SQUEEZE_WARNING"
        else:
            if funding > 0.0003:
                funding_score = 5.0
                funding_regime = "LONG_SQUEEZE_FUEL"
            elif funding >= -0.0002:
                funding_score = 4.0
                funding_regime = "HEALTHY_BEAR_FUNDING"
            elif funding >= -0.0006:
                funding_score = 2.2
                funding_regime = "ELEVATED_SHORT_FUNDING"
            else:
                funding_score = 0.5
                funding_regime = "CROWDED_SHORT_SQUEEZE_WARNING"

        # Component C: Macro Stability & Post-Shock Cooldown Recovery (0.0 to 4.0 pts)
        aligned_news = (
            external_news_sentiment
            if target_direction == "LONG"
            else -external_news_sentiment
        )
        raw_macro_score = float(np.clip(2.5 + (aligned_news * 1.5), 0.5, 4.0))
        macro_score = raw_macro_score * cooldown_state["cooldown_stability_factor"]

        hard_veto = bool(detected_shock or cooldown_state["hard_veto_active"])
        if hard_veto:
            macro_score = 0.0

        total_score = round(min(max_points, ob_score + funding_score + macro_score), 2)

        return {
            "symbol": symbol,
            "horizon": horizon,
            "evaluated_direction": target_direction,
            "sentiment_confluence_score": total_score,
            "funding_regime": funding_regime,
            "funding_rate": funding,
            "orderbook_imbalance": ob_imb,
            "spread_bps": spread_bps,
            "news_shock_veto": hard_veto,
            "cooldown_state": cooldown_state,
            "score_breakdown": {
                "orderbook_score": round(ob_score, 2),
                "funding_score": round(funding_score, 2),
                "macro_stability_score": round(macro_score, 2),
            },
            "sentiment_memory_snapshot": {
                "sentiment_score": total_score,
                "funding_regime": funding_regime,
                "funding_rate": funding,
                "orderbook_imbalance": ob_imb,
            },
        }


async def _demo():
    agent = CryptoSentimentAndMicrostructureAgent()
    try:
        ms = await agent.fetch_live_microstructure("BTCUSDT")
        res = agent.evaluate_sentiment_confluence(
            symbol="BTCUSDT",
            target_direction="LONG",
            horizon="scalp",
            microstructure=ms,
            external_news_sentiment=0.25,
            news_shock_flag=False,
        )
        print("Live Sentiment & Cooldown Confluence Output:")
        for k, v in res.items():
            print(f"  {k}: {v}")
    finally:
        await agent.close_exchange()


if __name__ == "__main__":
    asyncio.run(_demo())