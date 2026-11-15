import asyncio
import inspect
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import ConfigLoader
from src.database.db_manager import SegregatedDatabaseManager
from src.ingestion.feature_engineer import CryptoFeatureEngineer
from src.agents.pattern_agent import PatternBreakoutAgent
from src.agents.ml_prediction_agent import MLPredictionAndShapAgent
from src.agents.sentiment_agent import CryptoSentimentAndMicrostructureAgent
from src.agents.debate_agent import BullBearDebateAgent


class MasterConfluenceEngine:
    """
    100-Point Multi-Agent Confluence & Dynamic Risk Management Engine.
    Orchestrates:
      - PatternBreakoutAgent (25 pts)
      - MLPredictionAndShapAgent (45 pts)
      - CryptoSentimentAndMicrostructureAgent (15 pts)
      - BullBearDebateAgent + 15-Trade Memory + Gemini LLM Judge (15 pts)
    Computes dynamic MAE/MFE-calibrated ATR Stop-Loss, TP1, TP2, and Position Sizing
    ONLY when a trade is approved for execution (`should_execute=True`).
    """

    def __init__(self, config_path: str = "config/assets_config.yaml"):
        self.config_loader = ConfigLoader(config_path)
        self.db_manager = SegregatedDatabaseManager(config_path)
        self.pattern_agent = PatternBreakoutAgent()
        self.ml_agent = MLPredictionAndShapAgent(config_path)
        self.sentiment_agent = CryptoSentimentAndMicrostructureAgent(config_path)
        self.debate_agent = BullBearDebateAgent(config_path)

        full_cfg = self.config_loader.full_config
        self.confluence_cfg = full_cfg.get("confluence", {})
        self.risk_cfg = full_cfg.get("risk_management", {})
        self.min_entry_score = float(
            self.confluence_cfg.get("min_entry_threshold", 65.0)
        )
        self.high_conviction_score = float(
            self.confluence_cfg.get("high_conviction_threshold", 80.0)
        )

    def _run_pattern_agent(
        self,
        df: pd.DataFrame,
        htf_df: Optional[pd.DataFrame],
        max_points: float = 25.0,
    ) -> Dict[str, Any]:
        """
        Dynamically invokes PatternBreakoutAgent's primary evaluation method and
        normalizes output keys so downstream agents receive a consistent schema.
        """
        candidate_methods = [
            "evaluate_structure",
            "analyze_pattern_and_structure",
            "analyze_structure",
            "evaluate_pattern",
            "analyze",
            "evaluate",
        ]
        target_fn = None
        for name in candidate_methods:
            if hasattr(self.pattern_agent, name) and callable(getattr(self.pattern_agent, name)):
                target_fn = getattr(self.pattern_agent, name)
                break

        if target_fn is None:
            for name, fn in inspect.getmembers(self.pattern_agent, predicate=callable):
                if not name.startswith("_"):
                    target_fn = fn
                    break

        sig = inspect.signature(target_fn)
        kwargs: Dict[str, Any] = {}
        for param in sig.parameters:
            if param in ("exec_df", "df", "primary_df", "ohlcv_df"):
                kwargs[param] = df
            elif param in ("htf_df", "higher_tf_df", "context_df"):
                kwargs[param] = htf_df
            elif param in ("max_points", "max_score"):
                kwargs[param] = max_points

        raw_rep = target_fn(**kwargs) if kwargs else target_fn(df)

        direction = (
            raw_rep.get("direction")
            or raw_rep.get("signal_direction")
            or raw_rep.get("dominant_direction")
            or "NEUTRAL"
        )
        score = float(
            raw_rep.get(
                "pattern_confluence_score",
                raw_rep.get(
                    "technical_confluence_score",
                    raw_rep.get("confluence_score", raw_rep.get("score", 0.0)),
                ),
            )
        )
        pattern_type = str(
            raw_rep.get("pattern_type", raw_rep.get("pattern", "No_Clear_Pattern"))
        )
        market_regime = str(
            raw_rep.get("market_regime", raw_rep.get("regime", "Ranging"))
        )
        vol_ratio = float(
            raw_rep.get("volume_ratio", raw_rep.get("vol_ratio_20", 1.0))
        )
        htf_trend = str(raw_rep.get("htf_trend", "NEUTRAL"))
        htf_aligned = bool(
            raw_rep.get("htf_aligned", htf_trend in (direction, "BULLISH" if direction == "LONG" else "BEARISH"))
        )
        htf_veto = bool(
            raw_rep.get("htf_counter_trend_veto", raw_rep.get("counter_trend_veto", False))
        )
        is_sweep = bool(
            raw_rep.get("is_liquidity_sweep", "Sweep" in pattern_type)
        )

        normalized = dict(raw_rep)
        normalized.update(
            {
                "direction": direction,
                "pattern_confluence_score": round(min(max_points, score), 2),
                "pattern_type": pattern_type,
                "market_regime": market_regime,
                "volume_ratio": round(vol_ratio, 2),
                "htf_trend": htf_trend,
                "htf_aligned": htf_aligned,
                "htf_counter_trend_veto": htf_veto,
                "is_liquidity_sweep": is_sweep,
            }
        )
        return normalized

    def _compute_dynamic_risk_plan(
        self,
        symbol: str,
        horizon: str,
        direction: str,
        entry_price: float,
        atr_val: float,
        pattern_type: str,
        market_regime: str,
        total_score: float,
        should_execute: bool,
        account_balance_usdt: float = 1000.0,
    ) -> Dict[str, Any]:
        """
        If `should_execute` is False, returns an inactive risk plan with `None` for SL/TP levels.
        If `should_execute` is True, queries historical MAE/MFE multipliers and computes exact
        SL, TP1, TP2 prices, R:R ratios, and risk-adjusted position size.
        """
        safe_atr = max(atr_val, entry_price * 0.0015)

        if not should_execute:
            return {
                "active": False,
                "status_reason": "NO_ACTIVE_TRADE_SETUP",
                "reference_price": round(entry_price, 2),
                "atr_value": round(safe_atr, 2),
                "entry_price": None,
                "stop_loss": None,
                "tp1_price": None,
                "tp2_price": None,
                "sl_atr_mult": None,
                "tp1_atr_mult": None,
                "tp2_atr_mult": None,
                "rr_ratio_tp1": None,
                "rr_ratio_tp2": None,
                "risk_amount_usdt": 0.0,
                "position_size_coins": 0.0,
                "notional_position_usdt": 0.0,
                "effective_leverage": 0.0,
                "memory_optimization_meta": None,
            }

        if hasattr(self.db_manager, "get_dynamic_sl_tp_multipliers"):
            dyn_mults = self.db_manager.get_dynamic_sl_tp_multipliers(
                horizon=horizon,
                symbol=symbol,
                pattern_type=pattern_type,
                market_regime=market_regime,
            )
        elif hasattr(self.db_manager, "calculate_dynamic_sl_tp"):
            dyn_mults = self.db_manager.calculate_dynamic_sl_tp(
                horizon=horizon,
                symbol=symbol,
                pattern_type=pattern_type,
                market_regime=market_regime,
            )
        else:
            dyn_mults = {
                "sl_atr_multiplier": 1.5,
                "tp1_atr_multiplier": 1.8,
                "tp2_atr_multiplier": 3.2,
                "sample_count": 0,
                "source": "DEFAULT_BASELINE",
            }

        sl_mult = float(dyn_mults.get("sl_atr_multiplier", dyn_mults.get("sl_multiplier", 1.5)))
        tp1_mult = float(dyn_mults.get("tp1_atr_multiplier", dyn_mults.get("tp1_multiplier", 1.8)))
        tp2_mult = float(dyn_mults.get("tp2_atr_multiplier", dyn_mults.get("tp2_multiplier", 3.2)))

        sl_dist = safe_atr * sl_mult
        tp1_dist = safe_atr * tp1_mult
        tp2_dist = safe_atr * tp2_mult

        if direction == "LONG":
            sl_price = entry_price - sl_dist
            tp1_price = entry_price + tp1_dist
            tp2_price = entry_price + tp2_dist
        else:
            sl_price = entry_price + sl_dist
            tp1_price = entry_price - tp1_dist
            tp2_price = entry_price - tp2_dist

        base_risk_pct = float(self.risk_cfg.get("max_risk_per_trade_pct", 1.0)) / 100.0
        tier_risk_multiplier = 1.0 if total_score >= self.high_conviction_score else 0.75

        risk_usdt = round(account_balance_usdt * base_risk_pct * tier_risk_multiplier, 2)
        position_size_coins = round(risk_usdt / max(sl_dist, 1e-8), 6)
        notional_usdt = round(position_size_coins * entry_price, 2)
        effective_leverage = round(notional_usdt / max(account_balance_usdt, 1.0), 2)

        rr_tp1 = round(tp1_dist / max(sl_dist, 1e-8), 2)
        rr_tp2 = round(tp2_dist / max(sl_dist, 1e-8), 2)

        return {
            "active": True,
            "status_reason": "TRADE_APPROVED",
            "reference_price": round(entry_price, 2),
            "entry_price": round(entry_price, 2),
            "atr_value": round(safe_atr, 2),
            "stop_loss": round(sl_price, 2),
            "tp1_price": round(tp1_price, 2),
            "tp2_price": round(tp2_price, 2),
            "sl_atr_mult": round(sl_mult, 3),
            "tp1_atr_mult": round(tp1_mult, 3),
            "tp2_atr_mult": round(tp2_mult, 3),
            "rr_ratio_tp1": rr_tp1,
            "rr_ratio_tp2": rr_tp2,
            "risk_amount_usdt": risk_usdt,
            "position_size_coins": position_size_coins,
            "notional_position_usdt": notional_usdt,
            "effective_leverage": effective_leverage,
            "memory_optimization_meta": dyn_mults,
        }

    def evaluate_confluence(
        self,
        symbol: str,
        horizon: str,
        df: pd.DataFrame,
        htf_df: Optional[pd.DataFrame] = None,
        microstructure: Optional[Dict[str, float]] = None,
        external_news_sentiment: float = 0.0,
        news_shock_flag: bool = False,
        use_llm_debate: bool = True,
        account_balance_usdt: float = 1000.0,
        auto_log_to_db: bool = False,
    ) -> Dict[str, Any]:
        """
        Runs all 4 specialized agents, computes the 100-point confluence score,
        checks institutional safety vetoes, and builds the complete dynamic risk plan
        ONLY when `should_execute` is True.
        """
        feat_df = (
            df
            if "atr_14" in df.columns
            else CryptoFeatureEngineer.compute_features(df, drop_na=True)
        )
        latest_close = float(feat_df["close"].iloc[-1])
        latest_atr = float(feat_df["atr_14"].iloc[-1])

        # 1. Pattern & Breakout Agent (Max 25.0 Points)
        pattern_rep = self._run_pattern_agent(
            df=feat_df, htf_df=htf_df, max_points=25.0
        )
        pat_dir = pattern_rep.get("direction", "NEUTRAL")

        # 2. ML & SHAP Agent (Max 45.0 Points)
        target_dir = pat_dir if pat_dir in ("LONG", "SHORT") else None
        ml_rep = self.ml_agent.predict_and_explain(
            symbol=symbol,
            horizon=horizon,
            df=feat_df,
            target_direction=target_dir,
            max_points=45.0,
        )
        final_direction = ml_rep["evaluated_direction"]

        pattern_score = (
            float(pattern_rep["pattern_confluence_score"])
            if pat_dir == final_direction
            else round(float(pattern_rep["pattern_confluence_score"]) * 0.35, 2)
        )
        ml_score = float(ml_rep["ml_confluence_score"])

        # 3. Derivatives Sentiment & Microstructure Agent (Max 15.0 Points)
        sentiment_rep = self.sentiment_agent.evaluate_sentiment_confluence(
            symbol=symbol,
            target_direction=final_direction,
            horizon=horizon,
            df=feat_df,
            microstructure=microstructure,
            external_news_sentiment=external_news_sentiment,
            news_shock_flag=news_shock_flag,
            max_points=15.0,
        )
        sentiment_score = float(sentiment_rep["sentiment_confluence_score"])

        # 4. Bull vs. Bear Debate & Historical Memory Agent (Max 15.0 Points)
        debate_rep = self.debate_agent.conduct_debate(
            symbol=symbol,
            horizon=horizon,
            target_direction=final_direction,
            pattern_report=pattern_rep,
            ml_report=ml_rep,
            sentiment_report=sentiment_rep,
            use_llm=use_llm_debate,
            max_points=15.0,
        )
        debate_score = float(debate_rep["debate_confluence_score"])

        # 5. Aggregate 100-Point Score & Check Institutional Vetoes
        total_score = round(
            min(100.0, ml_score + pattern_score + sentiment_score + debate_score), 2
        )

        veto_reasons = []
        if sentiment_rep.get("news_shock_veto"):
            veto_reasons.append("MACRO_OR_VOLATILITY_SHOCK_VETO")
        if pattern_rep.get("htf_counter_trend_veto"):
            veto_reasons.append("STRONG_HTF_COUNTER_TREND_VETO")
        if pattern_rep.get("is_liquidity_sweep") and pat_dir not in ("NEUTRAL", final_direction):
            veto_reasons.append("OPPOSING_LIQUIDITY_SWEEP_FAKEOUT_VETO")

        is_vetoed = len(veto_reasons) > 0

        if is_vetoed:
            signal_tier = "VETOED_NO_TRADE"
            should_execute = False
        elif total_score >= self.high_conviction_score:
            signal_tier = "A_PLUS_INSTITUTIONAL"
            should_execute = True
        elif total_score >= self.min_entry_score:
            signal_tier = "B_STANDARD_CONFLUENCE"
            should_execute = True
        else:
            signal_tier = "NO_TRADE_LOW_CONFLUENCE"
            should_execute = False

        # 6. Dynamic Risk Plan (Calculated ONLY if should_execute is True)
        risk_plan = self._compute_dynamic_risk_plan(
            symbol=symbol,
            horizon=horizon,
            direction=final_direction,
            entry_price=latest_close,
            atr_val=latest_atr,
            pattern_type=pattern_rep.get("pattern_type", "No_Clear_Pattern"),
            market_regime=pattern_rep.get("market_regime", "Ranging"),
            total_score=total_score,
            should_execute=should_execute,
            account_balance_usdt=account_balance_usdt,
        )

        logged_trade_id: Optional[int] = None
        if auto_log_to_db and should_execute and hasattr(self.db_manager, "log_new_trade"):
            enriched_features = dict(ml_rep.get("latest_feature_vector", {}))
            enriched_features.update(sentiment_rep.get("sentiment_memory_snapshot", {}))

            trade_payload = {
                "symbol": symbol,
                "horizon": horizon,
                "timeframe": self.ml_agent.trainer.HORIZON_TRAINING_SPECS[horizon]["timeframe"],
                "direction": final_direction,
                "pattern_type": pattern_rep.get("pattern_type", "No_Clear_Pattern"),
                "market_regime": pattern_rep.get("market_regime", "Ranging"),
                "confluence_score": total_score,
                "ml_probability": ml_rep.get("directional_confidence", 0.5),
                "technical_score": pattern_score,
                "sentiment_score": sentiment_score,
                "debate_score": debate_score,
                "entry_price": risk_plan["entry_price"],
                "stop_loss": risk_plan["stop_loss"],
                "tp1_price": risk_plan["tp1_price"],
                "tp2_price": risk_plan["tp2_price"],
                "entry_atr": risk_plan["atr_value"],
                "position_size": risk_plan["position_size_coins"],
                "feature_vector_json": enriched_features,
                "shap_top_features_json": ml_rep.get("shap_top_features_json", {}),
                "status": "OPEN",
            }
            logged_trade_id = self.db_manager.log_new_trade(horizon, trade_payload)

        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "horizon": horizon,
            "direction": final_direction if should_execute else f"WATCH_{final_direction}",
            "evaluated_bias": final_direction,
            "should_execute": should_execute,
            "signal_tier": signal_tier,
            "total_confluence_score": total_score,
            "component_scores": {
                "ml_shap_score_45": ml_score,
                "pattern_structure_score_25": pattern_score,
                "sentiment_microstructure_score_15": sentiment_score,
                "debate_memory_score_15": debate_score,
            },
            "veto_active": is_vetoed,
            "veto_reasons": veto_reasons,
            "risk_plan": risk_plan,
            "logged_trade_id": logged_trade_id,
            "agent_reports": {
                "pattern": pattern_rep,
                "ml_shap": ml_rep,
                "sentiment": sentiment_rep,
                "debate": debate_rep,
            },
        }

    async def evaluate_live_symbol_horizon(
        self,
        symbol: str,
        horizon: str,
        df: pd.DataFrame,
        htf_df: Optional[pd.DataFrame] = None,
        use_llm_debate: bool = True,
        account_balance_usdt: float = 1000.0,
    ) -> Dict[str, Any]:
        """Fetches live Binance Futures microstructure and runs the full 100-pt evaluation."""
        try:
            ms = await self.sentiment_agent.fetch_live_microstructure(symbol)
            return self.evaluate_confluence(
                symbol=symbol,
                horizon=horizon,
                df=df,
                htf_df=htf_df,
                microstructure=ms,
                use_llm_debate=use_llm_debate,
                account_balance_usdt=account_balance_usdt,
            )
        finally:
            await self.sentiment_agent.close_exchange()


async def _demo_master_engine():
    engine = MasterConfluenceEngine()
    p5m = PROJECT_ROOT / "data" / "cache" / "BTCUSDT_5m.parquet"
    p1h = PROJECT_ROOT / "data" / "cache" / "BTCUSDT_1h.parquet"

    if p5m.exists() and p1h.exists():
        df_5m = pd.read_parquet(p5m).tail(250).reset_index(drop=True)
        df_1h = pd.read_parquet(p1h).tail(250).reset_index(drop=True)

        decision = await engine.evaluate_live_symbol_horizon(
            symbol="BTCUSDT",
            horizon="scalp",
            df=df_5m,
            htf_df=df_1h,
            use_llm_debate=True,
            account_balance_usdt=1000.0,
        )

        print("\n=======================================================")
        print(f" MASTER 100-POINT CONFLUENCE ENGINE: {decision['symbol']} ({decision['horizon'].upper()})")
        print("=======================================================")
        print(f"  Signal Decision      : {decision['direction']}")
        print(f"  Signal Tier          : {decision['signal_tier']} (Execute: {decision['should_execute']})")
        print(f"  Total Score          : {decision['total_confluence_score']} / 100.0 (Min Required: {engine.min_entry_score})")
        print(f"  Score Breakdown      : {decision['component_scores']}")
        print(f"  Veto Status          : {decision['veto_active']} {decision['veto_reasons']}")
        print("-------------------------------------------------------")
        rp = decision["risk_plan"]
        if decision["should_execute"]:
            print(f"  Entry Price          : ${rp['entry_price']:,.2f} (ATR: ${rp['atr_value']:,.2f})")
            print(f"  Dynamic Stop-Loss    : ${rp['stop_loss']:,.2f} ({rp['sl_atr_mult']}x ATR)")
            print(f"  Dynamic TP1 (50% BE) : ${rp['tp1_price']:,.2f} ({rp['tp1_atr_mult']}x ATR | R:R 1:{rp['rr_ratio_tp1']})")
            print(f"  Dynamic TP2 (Runner) : ${rp['tp2_price']:,.2f} ({rp['tp2_atr_mult']}x ATR | R:R 1:{rp['rr_ratio_tp2']})")
            print(f"  Position Sizing      : {rp['position_size_coins']} BTC (${rp['notional_position_usdt']:,.2f} Notional | Risk: ${rp['risk_amount_usdt']})")
        else:
            print(f"  Market Reference     : ${rp['reference_price']:,.2f} (ATR: ${rp['atr_value']:,.2f})")
            print("  Execution Plan       : STAND ASIDE (No SL / TP / Position generated — Confluence below 65.0 threshold)")
        print("-------------------------------------------------------")
        llm_rev = decision["agent_reports"]["debate"]["llm_executive_review"]
        print(f"  Judge Synthesis ({llm_rev['llm_provider']}):")
        print(f"    \"{llm_rev['judge_synthesis']}\"")
        print("=======================================================\n")


if __name__ == "__main__":
    asyncio.run(_demo_master_engine())