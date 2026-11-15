import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langgraph.graph import END, START, StateGraph
from src.core.confluence_engine import MasterConfluenceEngine
from src.ingestion.feature_engineer import CryptoFeatureEngineer


class TradingGraphState(TypedDict, total=False):
    """Typed state dictionary flowing through the LangGraph multi-agent pipeline."""
    symbol: str
    horizon: str
    df: pd.DataFrame
    htf_df: Optional[pd.DataFrame]
    microstructure: Optional[Dict[str, float]]
    external_news_sentiment: float
    news_shock_flag: bool
    use_llm_debate: bool
    account_balance_usdt: float
    auto_log_to_db: bool

    # Enriched data & intermediate agent outputs
    feat_df: pd.DataFrame
    latest_close: float
    latest_atr: float
    pattern_report: Dict[str, Any]
    ml_report: Dict[str, Any]
    sentiment_report: Dict[str, Any]
    debate_report: Dict[str, Any]

    # Final aggregated outputs
    evaluated_bias: str
    pattern_score: float
    ml_score: float
    sentiment_score: float
    debate_score: float
    total_confluence_score: float
    veto_active: bool
    veto_reasons: List[str]
    should_execute: bool
    signal_tier: str
    risk_plan: Dict[str, Any]
    logged_trade_id: Optional[int]
    final_decision: Dict[str, Any]


class LangGraphTradingOrchestrator:
    """
    Stateful Multi-Agent Workflow built on LangGraph (`StateGraph`).
    Nodes:
      1. feature_prep_node       -> Scale-invariant indicators & ATR extraction
      2. pattern_structure_node  -> Market structure, breakout & HTF alignment (25 pts)
      3. ml_shap_node            -> XGBoost probability, PPV & TreeSHAP drivers (45 pts)
      4. sentiment_cooldown_node -> Orderbook, funding & SQLite shock cooldown (15 pts)
      5. debate_memory_node      -> Bull/Bear debate, 15-trade memory & Gemini LLM (15 pts)
      6. risk_and_veto_gate_node -> Veto checks, dynamic MAE/MFE SL/TP & DB logging
    """

    def __init__(self, config_path: str = "config/assets_config.yaml"):
        self.engine = MasterConfluenceEngine(config_path)
        self.graph = self._build_compiled_graph()

    def _feature_prep_node(self, state: TradingGraphState) -> Dict[str, Any]:
        raw_df = state["df"]
        feat_df = (
            raw_df
            if "atr_14" in raw_df.columns
            else CryptoFeatureEngineer.compute_features(raw_df, drop_na=True)
        )
        return {
            "feat_df": feat_df,
            "latest_close": float(feat_df["close"].iloc[-1]),
            "latest_atr": float(feat_df["atr_14"].iloc[-1]),
        }

    def _pattern_structure_node(self, state: TradingGraphState) -> Dict[str, Any]:
        pat_rep = self.engine._run_pattern_agent(
            df=state["feat_df"],
            htf_df=state.get("htf_df"),
            max_points=25.0,
        )
        return {"pattern_report": pat_rep}

    def _ml_shap_node(self, state: TradingGraphState) -> Dict[str, Any]:
        pat_rep = state["pattern_report"]
        pat_dir = pat_rep.get("direction", "NEUTRAL")
        target_dir = pat_dir if pat_dir in ("LONG", "SHORT") else None

        ml_rep = self.engine.ml_agent.predict_and_explain(
            symbol=state["symbol"],
            horizon=state["horizon"],
            df=state["feat_df"],
            target_direction=target_dir,
            max_points=45.0,
        )
        final_dir = ml_rep["evaluated_direction"]
        pat_score = (
            float(pat_rep["pattern_confluence_score"])
            if pat_dir == final_dir
            else round(float(pat_rep["pattern_confluence_score"]) * 0.35, 2)
        )
        return {
            "ml_report": ml_rep,
            "evaluated_bias": final_dir,
            "pattern_score": pat_score,
            "ml_score": float(ml_rep["ml_confluence_score"]),
        }

    def _sentiment_cooldown_node(self, state: TradingGraphState) -> Dict[str, Any]:
        sent_rep = self.engine.sentiment_agent.evaluate_sentiment_confluence(
            symbol=state["symbol"],
            target_direction=state["evaluated_bias"],
            horizon=state["horizon"],
            df=state["feat_df"],
            microstructure=state.get("microstructure"),
            external_news_sentiment=float(state.get("external_news_sentiment", 0.0)),
            news_shock_flag=bool(state.get("news_shock_flag", False)),
            max_points=15.0,
        )
        return {
            "sentiment_report": sent_rep,
            "sentiment_score": float(sent_rep["sentiment_confluence_score"]),
        }

    def _debate_memory_node(self, state: TradingGraphState) -> Dict[str, Any]:
        # Conditional optimization: if a hard shock veto is already active, skip external LLM API call
        shock_veto = bool(state["sentiment_report"].get("news_shock_veto", False))
        htf_veto = bool(state["pattern_report"].get("htf_counter_trend_veto", False))
        allow_llm = bool(state.get("use_llm_debate", True)) and not (shock_veto or htf_veto)

        deb_rep = self.engine.debate_agent.conduct_debate(
            symbol=state["symbol"],
            horizon=state["horizon"],
            target_direction=state["evaluated_bias"],
            pattern_report=state["pattern_report"],
            ml_report=state["ml_report"],
            sentiment_report=state["sentiment_report"],
            use_llm=allow_llm,
            max_points=15.0,
        )
        return {
            "debate_report": deb_rep,
            "debate_score": float(deb_rep["debate_confluence_score"]),
        }

    def _risk_and_veto_gate_node(self, state: TradingGraphState) -> Dict[str, Any]:
        ml_score = state["ml_score"]
        pattern_score = state["pattern_score"]
        sentiment_score = state["sentiment_score"]
        debate_score = state["debate_score"]
        final_direction = state["evaluated_bias"]

        total_score = round(
            min(100.0, ml_score + pattern_score + sentiment_score + debate_score), 2
        )

        pat_rep = state["pattern_report"]
        sent_rep = state["sentiment_report"]
        ml_rep = state["ml_report"]
        deb_rep = state["debate_report"]
        pat_dir = pat_rep.get("direction", "NEUTRAL")

        veto_reasons: List[str] = []
        if sent_rep.get("news_shock_veto"):
            veto_reasons.append("MACRO_OR_VOLATILITY_SHOCK_VETO")
        if pat_rep.get("htf_counter_trend_veto"):
            veto_reasons.append("STRONG_HTF_COUNTER_TREND_VETO")
        if pat_rep.get("is_liquidity_sweep") and pat_dir not in ("NEUTRAL", final_direction):
            veto_reasons.append("OPPOSING_LIQUIDITY_SWEEP_FAKEOUT_VETO")

        is_vetoed = len(veto_reasons) > 0

        if is_vetoed:
            signal_tier = "VETOED_NO_TRADE"
            should_execute = False
        elif total_score >= self.engine.high_conviction_score:
            signal_tier = "A_PLUS_INSTITUTIONAL"
            should_execute = True
        elif total_score >= self.engine.min_entry_score:
            signal_tier = "B_STANDARD_CONFLUENCE"
            should_execute = True
        else:
            signal_tier = "NO_TRADE_LOW_CONFLUENCE"
            should_execute = False

        risk_plan = self.engine._compute_dynamic_risk_plan(
            symbol=state["symbol"],
            horizon=state["horizon"],
            direction=final_direction,
            entry_price=state["latest_close"],
            atr_val=state["latest_atr"],
            pattern_type=pat_rep.get("pattern_type", "No_Clear_Pattern"),
            market_regime=pat_rep.get("market_regime", "Ranging"),
            total_score=total_score,
            should_execute=should_execute,
            account_balance_usdt=float(state.get("account_balance_usdt", 1000.0)),
        )

        logged_trade_id: Optional[int] = None
        if state.get("auto_log_to_db") and should_execute and hasattr(self.engine.db_manager, "log_new_trade"):
            enriched_features = dict(ml_rep.get("latest_feature_vector", {}))
            enriched_features.update(sent_rep.get("sentiment_memory_snapshot", {}))
            trade_payload = {
                "symbol": state["symbol"],
                "horizon": state["horizon"],
                "timeframe": self.engine.ml_agent.trainer.HORIZON_TRAINING_SPECS[state["horizon"]]["timeframe"],
                "direction": final_direction,
                "pattern_type": pat_rep.get("pattern_type", "No_Clear_Pattern"),
                "market_regime": pat_rep.get("market_regime", "Ranging"),
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
            logged_trade_id = self.engine.db_manager.log_new_trade(state["horizon"], trade_payload)

        decision = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "orchestrator": "LangGraph_StateGraph_v1",
            "symbol": state["symbol"],
            "horizon": state["horizon"],
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
                "pattern": pat_rep,
                "ml_shap": ml_rep,
                "sentiment": sent_rep,
                "debate": deb_rep,
            },
        }
        return {
            "total_confluence_score": total_score,
            "veto_active": is_vetoed,
            "veto_reasons": veto_reasons,
            "should_execute": should_execute,
            "signal_tier": signal_tier,
            "risk_plan": risk_plan,
            "logged_trade_id": logged_trade_id,
            "final_decision": decision,
        }

    def _build_compiled_graph(self):
        workflow = StateGraph(TradingGraphState)
        workflow.add_node("feature_prep", self._feature_prep_node)
        workflow.add_node("pattern_structure", self._pattern_structure_node)
        workflow.add_node("ml_shap", self._ml_shap_node)
        workflow.add_node("sentiment_cooldown", self._sentiment_cooldown_node)
        workflow.add_node("debate_memory", self._debate_memory_node)
        workflow.add_node("risk_and_veto_gate", self._risk_and_veto_gate_node)

        workflow.add_edge(START, "feature_prep")
        workflow.add_edge("feature_prep", "pattern_structure")
        workflow.add_edge("pattern_structure", "ml_shap")
        workflow.add_edge("ml_shap", "sentiment_cooldown")
        workflow.add_edge("sentiment_cooldown", "debate_memory")
        workflow.add_edge("debate_memory", "risk_and_veto_gate")
        workflow.add_edge("risk_and_veto_gate", END)

        return workflow.compile()

    def run_graph(
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
        initial_state: TradingGraphState = {
            "symbol": symbol,
            "horizon": horizon,
            "df": df,
            "htf_df": htf_df,
            "microstructure": microstructure,
            "external_news_sentiment": external_news_sentiment,
            "news_shock_flag": news_shock_flag,
            "use_llm_debate": use_llm_debate,
            "account_balance_usdt": account_balance_usdt,
            "auto_log_to_db": auto_log_to_db,
        }
        final_state = self.graph.invoke(initial_state)
        return final_state["final_decision"]


async def _demo_graph():
    orchestrator = LangGraphTradingOrchestrator()
    p5m = PROJECT_ROOT / "data" / "cache" / "BTCUSDT_5m.parquet"
    p1h = PROJECT_ROOT / "data" / "cache" / "BTCUSDT_1h.parquet"

    if p5m.exists() and p1h.exists():
        df_5m = pd.read_parquet(p5m).tail(250).reset_index(drop=True)
        df_1h = pd.read_parquet(p1h).tail(250).reset_index(drop=True)
        try:
            ms = await orchestrator.engine.sentiment_agent.fetch_live_microstructure("BTCUSDT")
            res = orchestrator.run_graph(
                symbol="BTCUSDT",
                horizon="scalp",
                df=df_5m,
                htf_df=df_1h,
                microstructure=ms,
                use_llm_debate=True,
            )
            print("LangGraph StateGraph Execution Complete:")
            print(f"  Orchestrator  : {res['orchestrator']}")
            print(f"  Signal        : {res['direction']} ({res['signal_tier']})")
            print(f"  Total Score   : {res['total_confluence_score']} / 100.0")
            print(f"  Components    : {res['component_scores']}")
        finally:
            await orchestrator.engine.sentiment_agent.close_exchange()


if __name__ == "__main__":
    asyncio.run(_demo_graph())