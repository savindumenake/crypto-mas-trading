import asyncio
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import ccxt.async_support as ccxt
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import ConfigLoader
from src.database.db_manager import SegregatedDatabaseManager
from src.database.models import FundamentalStateLog, TradeExecutionLog


class CryptoSentimentAndMicrostructureAgent:
    """
    Evaluates real-time Crypto Derivatives Sentiment, L2 Orderbook Imbalance,
    Perpetual Funding Rate positioning, Persistent Macro/Volatility Shock Cooldowns
    (stored in SQLite FundamentalStateLog), and Sentiment-Matched Historical Trade Lookups.
    Outputs a 15.0-point Confluence Score + Circuit Breaker Veto flag.
    """

    SHOCK_COOLDOWN_MINUTES: Dict[str, int] = {
        "scalp": 45,      # 45 mins cooldown for 1m/5m/15m scalping
        "intraday": 180,  # 3 hours cooldown for 1h intraday
        "swing": 480,     # 8 hours cooldown for 4h/1d swing
    }

    HARD_VETO_MINUTES: Dict[str, int] = {
        "scalp": 15,
        "intraday": 45,
        "swing": 120,
    }

    def __init__(self, config_path: str = "config/assets_config.yaml"):
        self.config_loader = ConfigLoader(config_path)
        sys_cfg = self.config_loader.full_config.get("system", {})
        self.exchange_id = sys_cfg.get("default_exchange", "binance")
        self.exchange: Optional[ccxt.Exchange] = None
        self.db_manager = SegregatedDatabaseManager(config_path)

    async def initialize_exchange(self) -> None:
        if self.exchange is None:
            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange = exchange_class(
                {
                    "enableRateLimit": True,
                    "options": {"defaultType": "future"},
                }
            )

    async def close_exchange(self) -> None:
        if self.exchange is not None:
            await self.exchange.close()
            self.exchange = None

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        if "/" in symbol:
            return symbol
        if symbol.endswith("USDT"):
            return f"{symbol[:-4]}/USDT"
        return symbol

    async def fetch_live_microstructure(self, symbol: str) -> Dict[str, float]:
        """
        Concurrently fetches live Perpetual Funding Rate and Top-20 L2 Orderbook
        Bid/Ask volumes from Binance Futures.
        """
        await self.initialize_exchange()
        ccxt_symbol = self._normalize_symbol(symbol)

        funding_rate = 0.0001
        ob_imbalance = 0.0
        spread_bps = 1.0

        try:
            funding_task = self.exchange.fetch_funding_rate(ccxt_symbol)
            ob_task = self.exchange.fetch_order_book(ccxt_symbol, limit=20)
            funding_res, ob_res = await asyncio.gather(
                funding_task, ob_task, return_exceptions=True
            )

            if isinstance(funding_res, dict):
                raw_fr = funding_res.get("fundingRate")
                if raw_fr is not None:
                    funding_rate = float(raw_fr)

            if isinstance(ob_res, dict):
                bids = ob_res.get("bids", [])
                asks = ob_res.get("asks", [])
                bid_vol = sum(float(b[1]) for b in bids[:20]) if bids else 0.0
                ask_vol = sum(float(a[1]) for a in asks[:20]) if asks else 0.0
                total_vol = bid_vol + ask_vol
                if total_vol > 0:
                    ob_imbalance = (bid_vol - ask_vol) / total_vol

                if bids and asks and float(bids[0][0]) > 0:
                    best_bid = float(bids[0][0])
                    best_ask = float(asks[0][0])
                    spread_bps = ((best_ask - best_bid) / best_bid) * 10000.0
        except Exception:
            pass

        return {
            "funding_rate": round(funding_rate, 6),
            "orderbook_imbalance": round(ob_imbalance, 4),
            "spread_bps": round(spread_bps, 3),
        }

    # -----------------------------------------------------------------
    # 1. PERSISTENT SHOCK MEMORY & COOLDOWN DECAY (DATABASE-BACKED)
    # -----------------------------------------------------------------
    def persist_sentiment_and_check_cooldown(
        self,
        symbol: str,
        horizon: str,
        funding_rate: float,
        ob_imbalance: float,
        sentiment_score_norm: float,
        current_shock_detected: bool,
        summary_text: str = "",
    ) -> Dict[str, Any]:
        """
        Saves the current sentiment state to `fundamental_state_logs` in the horizon DB
        and checks if ANY shock occurred within the horizon's cooldown window.
        Applies an exponential time-decay penalty as the shock ages.
        """
        cooldown_mins = self.SHOCK_COOLDOWN_MINUTES.get(horizon, 60)
        hard_veto_mins = self.HARD_VETO_MINUTES.get(horizon, 15)
        now_utc = datetime.now(timezone.utc)
        cutoff_utc = now_utc - timedelta(minutes=cooldown_mins)

        # Derive sentiment_label from score & orderbook pressure
        composite_signal = sentiment_score_norm + (ob_imbalance * 0.5)
        if composite_signal > 0.10:
            sentiment_label = "BULLISH"
        elif composite_signal < -0.10:
            sentiment_label = "BEARISH"
        else:
            sentiment_label = "NEUTRAL"

        summary_val = summary_text or f"OB_Imb={ob_imbalance:.3f}, FR={funding_rate:.5f}"

        valid_cols = {c.name for c in FundamentalStateLog.__table__.columns}
        candidate_kwargs = {
            "symbol": symbol,
            "timestamp": now_utc,
            "sentiment_score": sentiment_score_norm,
            "sentiment_label": sentiment_label,
            "impact_horizon": horizon,
            "funding_rate": funding_rate,
            "open_interest_change_pct": round(ob_imbalance * 100.0, 2),
            "news_shock_active": current_shock_detected,
            "summary_text": summary_val,
            "retrieval_query_text": f"{symbol}_{horizon}_{sentiment_label}",
            "dominant_narrative": summary_val,
            "summary_context": summary_val,
        }
        filtered_kwargs = {k: v for k, v in candidate_kwargs.items() if k in valid_cols}

        with self.db_manager._get_session(horizon) as session:
            log_entry = FundamentalStateLog(**filtered_kwargs)
            session.add(log_entry)
            session.commit()

            recent_shock = (
                session.query(FundamentalStateLog)
                .filter(
                    FundamentalStateLog.symbol == symbol,
                    FundamentalStateLog.news_shock_active.is_(True),
                    FundamentalStateLog.timestamp >= cutoff_utc,
                )
                .order_by(FundamentalStateLog.timestamp.desc())
                .first()
            )

            if recent_shock is None:
                return {
                    "in_cooldown_window": False,
                    "hard_veto_active": False,
                    "minutes_since_shock": None,
                    "shock_decay_penalty_factor": 0.0,
                }

            shock_ts = recent_shock.timestamp
            if shock_ts.tzinfo is None:
                shock_ts = shock_ts.replace(tzinfo=timezone.utc)

            elapsed_mins = max(0.0, (now_utc - shock_ts).total_seconds() / 60.0)
            hard_veto = elapsed_mins <= hard_veto_mins

            decay_lambda = 2.3 / max(cooldown_mins, 1)
            penalty_factor = round(math.exp(-decay_lambda * elapsed_mins), 4)

            return {
                "in_cooldown_window": True,
                "hard_veto_active": hard_veto,
                "minutes_since_shock": round(elapsed_mins, 1),
                "shock_decay_penalty_factor": penalty_factor,
            }

    # -----------------------------------------------------------------
    # 2. SENTIMENT-MATCHED 15-TRADE HISTORICAL MEMORY LOOKUP
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
        Retrieves closed historical trades matching `pattern_type` and `market_regime`,
        then ranks and selects the `top_k` (15) trades whose historical sentiment score
        and funding regime at trade entry were closest to the current live sentiment.
        """
        with self.db_manager._get_session(horizon) as session:
            candidates: List[TradeExecutionLog] = (
                session.query(TradeExecutionLog)
                .filter(
                    TradeExecutionLog.symbol == symbol,
                    TradeExecutionLog.pattern_type == pattern_type,
                    TradeExecutionLog.market_regime == market_regime,
                    TradeExecutionLog.status == "CLOSED",
                )
                .order_by(TradeExecutionLog.exit_timestamp.desc())
                .limit(60)
                .all()
            )

            if not candidates:
                return {
                    "matched_trades_count": 0,
                    "sentiment_matched_win_rate": 0.50,
                    "avg_historical_sentiment": 0.0,
                    "sentiment_similarity_confidence": 0.0,
                }

            scored_candidates = []
            for t in candidates:
                hist_sent = float(t.sentiment_score if t.sentiment_score is not None else 7.5)
                hist_meta = t.feature_vector_json or {}
                hist_f_regime = hist_meta.get("funding_regime", "UNKNOWN")

                sent_diff = abs(current_sentiment_score - hist_sent)
                regime_bonus = 0.0 if hist_f_regime == current_funding_regime else 2.5
                total_distance = sent_diff + regime_bonus

                scored_candidates.append((total_distance, t))

            scored_candidates.sort(key=lambda x: x[0])
            selected = [t for _, t in scored_candidates[:top_k]]

            wins = sum(1 for t in selected if t.is_win)
            win_rate = wins / len(selected)
            avg_sent = sum(float(t.sentiment_score or 0.0) for t in selected) / len(selected)
            avg_dist = sum(d for d, _ in scored_candidates[:top_k]) / len(selected)
            sim_conf = max(0.0, 1.0 - (avg_dist / 15.0))

            return {
                "matched_trades_count": len(selected),
                "sentiment_matched_win_rate": round(win_rate, 4),
                "avg_historical_sentiment": round(avg_sent, 2),
                "sentiment_similarity_confidence": round(sim_conf, 4),
            }

    # -----------------------------------------------------------------
    # 3. 15-POINT SENTIMENT & MICROSTRUCTURE CONFLUENCE EVALUATION
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
        if target_direction not in ("LONG", "SHORT"):
            target_direction = "LONG"

        ms = microstructure or {
            "funding_rate": 0.0001,
            "orderbook_imbalance": 0.0,
            "spread_bps": 1.0,
        }
        funding_rate = float(ms.get("funding_rate", 0.0001))
        ob_imb = float(ms.get("orderbook_imbalance", 0.0))
        spread_bps = float(ms.get("spread_bps", 1.0))

        # 1. Orderbook Imbalance Score (0 to 6.0 points)
        aligned_ob = ob_imb if target_direction == "LONG" else -ob_imb
        ob_score = min(6.0, max(0.0, ((aligned_ob + 0.25) / 0.60) * 6.0))

        # 2. Perpetual Funding Rate & Squeeze Filter (0 to 5.0 points)
        funding_regime = "NEUTRAL_POSITIONING"
        if target_direction == "LONG":
            if funding_rate < 0.0:
                funding_score = 5.0
                funding_regime = "SHORT_SQUEEZE_FUEL"
            elif funding_rate <= 0.00025:
                funding_score = 4.0
                funding_regime = "HEALTHY_BULL_FUNDING"
            elif funding_rate <= 0.0005:
                funding_score = 2.0
                funding_regime = "ELEVATED_LONG_FUNDING"
            else:
                funding_score = 0.0
                funding_regime = "OVERCROWDED_LONGS_WARNING"
        else:
            if funding_rate > 0.00035:
                funding_score = 5.0
                funding_regime = "LONG_LIQUIDATION_FUEL"
            elif funding_rate >= 0.0:
                funding_score = 3.5
                funding_regime = "NORMAL_FUNDING"
            elif funding_rate >= -0.00025:
                funding_score = 2.0
                funding_regime = "ELEVATED_SHORT_FUNDING"
            else:
                funding_score = 0.0
                funding_regime = "OVERCROWDED_SHORTS_WARNING"

        # 3. Point-in-Time Volatility Shock Detection + Persistent DB Cooldown Check
        instant_shock = bool(news_shock_flag)
        if df is not None and len(df) >= 25:
            high_low_range = (df["high"] - df["low"]).astype("float64")
            latest_range = float(high_low_range.iloc[-1])
            median_range = float(high_low_range.iloc[-25:-1].median())
            if median_range > 0 and (latest_range / median_range) >= 3.8:
                instant_shock = True

        if spread_bps > 15.0:
            instant_shock = True

        aligned_news = (
            external_news_sentiment
            if target_direction == "LONG"
            else -external_news_sentiment
        )
        raw_macro_score = min(4.0, max(0.0, 2.5 + (aligned_news * 3.0)))

        # Persist state & check if we are still inside the post-news shock cooldown window
        cooldown_info = self.persist_sentiment_and_check_cooldown(
            symbol=symbol,
            horizon=horizon,
            funding_rate=funding_rate,
            ob_imbalance=ob_imb,
            sentiment_score_norm=external_news_sentiment,
            current_shock_detected=instant_shock,
        )

        penalty_factor = cooldown_info["shock_decay_penalty_factor"]
        macro_score = raw_macro_score * (1.0 - penalty_factor)
        if cooldown_info["hard_veto_active"]:
            macro_score = 0.0

        total_sentiment_score = round(
            min(max_points, ob_score + funding_score + macro_score), 2
        )

        return {
            "symbol": symbol,
            "horizon": horizon,
            "evaluated_direction": target_direction,
            "sentiment_confluence_score": total_sentiment_score,
            "funding_rate": funding_rate,
            "funding_regime": funding_regime,
            "orderbook_imbalance": ob_imb,
            "spread_bps": spread_bps,
            "news_shock_veto": cooldown_info["hard_veto_active"],
            "cooldown_state": cooldown_info,
            "score_breakdown": {
                "orderbook_pressure_score": round(ob_score, 2),
                "funding_positioning_score": round(funding_score, 2),
                "macro_stability_score": round(macro_score, 2),
            },
            "sentiment_memory_snapshot": {
                "sentiment_score": total_sentiment_score,
                "funding_rate": funding_rate,
                "funding_regime": funding_regime,
                "orderbook_imbalance": ob_imb,
            },
        }

    async def analyze_live(
        self,
        symbol: str,
        target_direction: str = "LONG",
        horizon: str = "scalp",
        df: Optional[pd.DataFrame] = None,
        external_news_sentiment: float = 0.0,
    ) -> Dict[str, Any]:
        ms = await self.fetch_live_microstructure(symbol)
        return self.evaluate_sentiment_confluence(
            symbol=symbol,
            target_direction=target_direction,
            horizon=horizon,
            df=df,
            microstructure=ms,
            external_news_sentiment=external_news_sentiment,
        )


async def _demo_run():
    agent = CryptoSentimentAndMicrostructureAgent()
    try:
        cache_file = PROJECT_ROOT / "data" / "cache" / "BTCUSDT_5m.parquet"
        df = pd.read_parquet(cache_file).tail(50) if cache_file.exists() else None
        res = await agent.analyze_live("BTCUSDT", target_direction="LONG", horizon="scalp", df=df)
        print("Live BTCUSDT Sentiment, Cooldown Memory & Orderbook Output:")
        for k, v in res.items():
            print(f"  {k:<28}: {v}")
    finally:
        await agent.close_exchange()


if __name__ == "__main__":
    asyncio.run(_demo_run())