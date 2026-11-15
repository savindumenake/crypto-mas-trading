import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_phase2_confluence import _make_synthetic_df
from src.core.langgraph_orchestrator import LangGraphTradingOrchestrator
from src.execution.paper_executor import PaperTradeLifecycleManager


def test_langgraph_orchestrator_end_to_end(tmp_path, monkeypatch):
    """
    Verifies that LangGraphTradingOrchestrator executes all 6 StateGraph nodes
    and produces a valid 100-point decision dictionary.
    """
    orchestrator = LangGraphTradingOrchestrator()
    df_5m = _make_synthetic_df(160, "5min")
    df_1h = _make_synthetic_df(160, "1h")

    model_path = orchestrator.engine.ml_agent.trainer.get_model_path("BTCUSDT", "scalp")
    if not model_path.exists():
        monkeypatch.setattr(orchestrator.engine.ml_agent.trainer, "get_model_dir", lambda sym: tmp_path)
        orchestrator.engine.ml_agent.trainer.train_horizon_model("BTCUSDT", "scalp", df_5m)

    orchestrator.engine.sentiment_agent.clear_symbol_shocks("scalp", "BTCUSDT")

    res = orchestrator.run_graph(
        symbol="BTCUSDT",
        horizon="scalp",
        df=df_5m,
        htf_df=df_1h,
        microstructure={"funding_rate": -0.0001, "orderbook_imbalance": 0.25, "spread_bps": 1.0},
        use_llm_debate=False,
        account_balance_usdt=1500.0,
    )

    assert res["orchestrator"] == "LangGraph_StateGraph_v1"
    assert res["symbol"] == "BTCUSDT"
    assert res["horizon"] == "scalp"
    assert 0.0 <= res["total_confluence_score"] <= 100.0
    assert "pattern" in res["agent_reports"]
    assert "ml_shap" in res["agent_reports"]
    assert "sentiment" in res["agent_reports"]
    assert "debate" in res["agent_reports"]


def test_paper_executor_lifecycle_and_horizon_isolation():
    """
    Verifies:
      1. Opening a paper trade and blocking duplicate concurrent setups on the same horizon.
      2. TP1 partial profit + automatic Breakeven Stop-Loss movement.
      3. Strict isolation of PnL, Growth Rate %, and Equity Curves across scalp, intraday, and swing.
    """
    manager = PaperTradeLifecycleManager(default_horizon_capital_usdt=1000.0)

    # Clean up any leftover test records
    for h in ("scalp", "intraday", "swing"):
        with manager._db_conn(h) as conn:
            conn.execute("DELETE FROM trade_memory WHERE symbol = 'TEST_ISOLATION_COIN'")

    try:
        baseline_dash = manager.get_multi_horizon_dashboard_payload(1000.0)
        base_intraday_eq = baseline_dash["horizons"]["intraday"]["current_equity_usdt"]
        base_swing_eq = baseline_dash["horizons"]["swing"]["current_equity_usdt"]

        mock_scalp_decision = {
            "symbol": "TEST_ISOLATION_COIN",
            "horizon": "scalp",
            "direction": "LONG",
            "evaluated_bias": "LONG",
            "should_execute": True,
            "total_confluence_score": 84.0,
            "component_scores": {
                "ml_shap_score_45": 38.0,
                "pattern_structure_score_25": 22.0,
                "sentiment_microstructure_score_15": 12.0,
                "debate_memory_score_15": 12.0,
            },
            "risk_plan": {
                "entry_price": 50000.0,
                "atr_value": 100.0,
                "stop_loss": 49850.0,
                "tp1_price": 50180.0,
                "tp2_price": 50320.0,
                "position_size_coins": 0.1,
                "risk_amount_usdt": 15.0,
            },
            "agent_reports": {},
        }

        # 1. Open trade on scalp
        opened = manager.open_paper_trade(mock_scalp_decision)
        assert opened["opened"] is True

        # 2. Attempt duplicate open on scalp -> must be rejected
        dup = manager.open_paper_trade(mock_scalp_decision)
        assert dup["opened"] is False
        assert "DUPLICATE_ACTIVE_TRADE_EXISTS" in dup["reason"]

        # 3. Candle hits TP1 (50,180) -> status becomes TP1_HIT and SL moves to 50,000 (Breakeven)
        ev1 = manager.update_open_trades(
            horizon="scalp",
            symbol="TEST_ISOLATION_COIN",
            candle_high=50200.0,
            candle_low=49950.0,
            candle_close=50190.0,
        )
        assert len(ev1) == 1
        assert ev1[0]["new_status"] == "TP1_HIT"
        assert ev1[0]["stop_loss"] == 50000.0

        # 4. Candle pulls back to 50,000 (Breakeven SL hit after TP1) -> closes with positive 50% TP1 profit!
        ev2 = manager.update_open_trades(
            horizon="scalp",
            symbol="TEST_ISOLATION_COIN",
            candle_high=50210.0,
            candle_low=49990.0,
            candle_close=50000.0,
        )
        assert len(ev2) == 1
        assert ev2[0]["new_status"] == "CLOSED"
        assert ev2[0]["event_type"] == "BREAKEVEN_STOP_AFTER_TP1"
        assert ev2[0]["realized_pnl_r"] == 0.6  # 0.5 * (180 / 150) = +0.6R
        assert ev2[0]["realized_pnl_usdt"] == 9.0  # 0.6R * $15 risk = +$9.00

        # 5. Verify multi-horizon dashboard isolation (intraday & swing untouched)
        after_dash = manager.get_multi_horizon_dashboard_payload(1000.0)
        assert after_dash["horizons"]["scalp"]["current_equity_usdt"] >= 1009.0
        assert after_dash["horizons"]["intraday"]["current_equity_usdt"] == base_intraday_eq
        assert after_dash["horizons"]["swing"]["current_equity_usdt"] == base_swing_eq

    finally:
        for h in ("scalp", "intraday", "swing"):
            with manager._db_conn(h) as conn:
                conn.execute("DELETE FROM trade_memory WHERE symbol = 'TEST_ISOLATION_COIN'")