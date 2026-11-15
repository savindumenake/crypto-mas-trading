import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.sentiment_agent import CryptoSentimentAndMicrostructureAgent
from src.core.confluence_engine import MasterConfluenceEngine


def _make_synthetic_df(periods: int = 160, freq: str = "5min") -> pd.DataFrame:
    idx = pd.date_range("2026-02-01", periods=periods, freq=freq, tz="UTC")
    np.random.seed(7)
    prices = 92000 + np.cumsum(np.random.normal(8, 45, periods))
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "open": prices - 12,
            "high": prices + 40,
            "low": prices - 40,
            "close": prices,
            "volume": np.random.uniform(200, 500, periods),
        }
    )
    df.loc[df.index[-1], "close"] = float(df["high"].iloc[-25:-1].max()) + 120.0
    df.loc[df.index[-1], "high"] = float(df["close"].iloc[-1]) + 25.0
    df.loc[df.index[-1], "volume"] = float(df["volume"].iloc[-25:-1].mean()) * 2.8
    return df


def test_sentiment_shock_cooldown_persistence():
    """
    Verifies that a detected volatility/news shock is persisted in SQLite
    and triggers hard_veto_active within the horizon's cooldown window.
    """
    agent = CryptoSentimentAndMicrostructureAgent()
    ms = {"funding_rate": -0.0002, "orderbook_imbalance": 0.30, "spread_bps": 1.2}

    shock_res = agent.evaluate_sentiment_confluence(
        symbol="TESTBTC",
        target_direction="LONG",
        horizon="scalp",
        microstructure=ms,
        news_shock_flag=True,
    )
    assert shock_res["news_shock_veto"] is True
    assert shock_res["cooldown_state"]["in_cooldown_window"] is True
    assert shock_res["score_breakdown"]["macro_stability_score"] == 0.0

    follow_up = agent.evaluate_sentiment_confluence(
        symbol="TESTBTC",
        target_direction="LONG",
        horizon="scalp",
        microstructure=ms,
        news_shock_flag=False,
    )
    assert follow_up["cooldown_state"]["in_cooldown_window"] is True
    assert follow_up["news_shock_veto"] is True


def test_bull_bear_debate_and_master_confluence_engine(tmp_path, monkeypatch):
    """
    Tests MasterConfluenceEngine end-to-end:
    1. When should_execute is False, SL/TP1/TP2 must be None and direction is WATCH_LONG/SHORT.
    2. When should_execute is True, SL/TP1/TP2 must be valid prices matching direction.
    """
    engine = MasterConfluenceEngine()
    df_5m = _make_synthetic_df(160, "5min")
    df_1h = _make_synthetic_df(160, "1h")

    model_path = engine.ml_agent.trainer.get_model_path("BTCUSDT", "scalp")
    if not model_path.exists():
        monkeypatch.setattr(engine.ml_agent.trainer, "get_model_dir", lambda sym: tmp_path)
        engine.ml_agent.trainer.train_horizon_model("BTCUSDT", "scalp", df_5m)

    ms_bullish = {
        "funding_rate": -0.0001,
        "orderbook_imbalance": 0.28,
        "spread_bps": 0.8,
    }

    # Case 1: Normal evaluation
    decision = engine.evaluate_confluence(
        symbol="BTCUSDT",
        horizon="scalp",
        df=df_5m,
        htf_df=df_1h,
        microstructure=ms_bullish,
        use_llm_debate=False,
        account_balance_usdt=2000.0,
    )

    assert decision["symbol"] == "BTCUSDT"
    assert decision["evaluated_bias"] in ("LONG", "SHORT")
    assert 0.0 <= decision["total_confluence_score"] <= 100.0

    comp = decision["component_scores"]
    assert 0.0 <= comp["ml_shap_score_45"] <= 45.0
    assert 0.0 <= comp["pattern_structure_score_25"] <= 25.0
    assert 0.0 <= comp["sentiment_microstructure_score_15"] <= 15.0
    assert 0.0 <= comp["debate_memory_score_15"] <= 15.0

    rp = decision["risk_plan"]
    if not decision["should_execute"]:
        assert decision["direction"] in ("WATCH_LONG", "WATCH_SHORT")
        assert rp["active"] is False
        assert rp["stop_loss"] is None
        assert rp["tp1_price"] is None
        assert rp["tp2_price"] is None
        assert rp["position_size_coins"] == 0.0

    # Case 2: Force threshold to 30.0 to verify active trade SL/TP calculation
    engine.min_entry_score = 30.0
    active_decision = engine.evaluate_confluence(
        symbol="BTCUSDT",
        horizon="scalp",
        df=df_5m,
        htf_df=df_1h,
        microstructure=ms_bullish,
        use_llm_debate=False,
        account_balance_usdt=2000.0,
    )
    active_rp = active_decision["risk_plan"]
    assert active_decision["should_execute"] is True
    assert active_decision["direction"] in ("LONG", "SHORT")
    assert active_rp["active"] is True
    assert active_rp["position_size_coins"] > 0.0

    if active_decision["direction"] == "LONG":
        assert active_rp["stop_loss"] < active_rp["entry_price"] < active_rp["tp1_price"] < active_rp["tp2_price"]
    else:
        assert active_rp["stop_loss"] > active_rp["entry_price"] > active_rp["tp1_price"] > active_rp["tp2_price"]