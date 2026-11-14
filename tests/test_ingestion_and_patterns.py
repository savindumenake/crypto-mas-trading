import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.market_data import MultiTimeframeDataFetcher
from src.ingestion.feature_engineer import CryptoFeatureEngineer
from src.agents.pattern_agent import PatternBreakoutAgent


def test_boundary_aligned_closed_timeframes():
    """Verifies that exact minute/hour boundaries trigger the right timeframes."""
    watched = ["1m", "5m", "15m", "1h", "4h", "1d"]

    # 3600s = 01:00:00 UTC -> 1m, 5m, 15m, 1h should all trigger, but not 4h or 1d
    closed_at_1h = MultiTimeframeDataFetcher.get_closed_timeframes_at_timestamp(3600, watched)
    assert set(closed_at_1h) == {"1m", "5m", "15m", "1h"}

    # 86400s = 00:00:00 UTC (Midnight) -> All 6 timeframes close simultaneously
    closed_at_1d = MultiTimeframeDataFetcher.get_closed_timeframes_at_timestamp(86400, watched)
    assert set(closed_at_1d) == {"1m", "5m", "15m", "1h", "4h", "1d"}


def test_feature_engineer_scale_invariance_and_json_snapshot():
    """Verifies that all 25 ML features compute cleanly without NaNs or Infs."""
    idx = pd.date_range("2026-01-01", periods=80, freq="5min", tz="UTC")
    np.random.seed(7)
    prices = 90000 + np.cumsum(np.random.normal(0, 50, 80))

    df = pd.DataFrame({
        "timestamp": idx,
        "open": prices - 15,
        "high": prices + 45,
        "low": prices - 45,
        "close": prices,
        "volume": np.random.uniform(100, 500, 80),
    })

    feat_df = CryptoFeatureEngineer.compute_features(df, drop_na=True)
    snapshot = CryptoFeatureEngineer.extract_latest_feature_snapshot(feat_df)

    assert len(snapshot) == len(CryptoFeatureEngineer.ML_FEATURE_COLUMNS)
    for col, val in snapshot.items():
        assert isinstance(val, float), f"Feature {col} is not float"
        assert not np.isnan(val) and not np.isinf(val), f"Invalid value in {col}"


def test_pattern_agent_breakout_vs_liquidity_sweep():
    """
    Tests that PatternBreakoutAgent accurately distinguishes between:
    1. A clean Bullish Resistance Breakout (LONG)
    2. A Bearish Liquidity Sweep / Stop-Hunt Fakeout above Resistance (SHORT)
    """
    idx = pd.date_range("2026-01-01", periods=50, freq="15min", tz="UTC")
    np.random.seed(101)
    base = np.linspace(90000, 90100, 49)

    # 1. Test Clean Bullish Breakout on Bar 50
    df_breakout = pd.DataFrame({
        "timestamp": idx,
        "open": np.append(base - 10, 90120.0),
        "high": np.append(base + 30, 90360.0),   # Prior resistance is ~90130
        "low": np.append(base - 30, 90110.0),
        "close": np.append(base, 90340.0),       # Strong body close above resistance
        "volume": np.append(np.full(49, 200.0), 500.0),
    })

    agent = PatternBreakoutAgent(lookback_bars=20)
    res_breakout = agent.analyze(df_breakout, htf_df=df_breakout)

    assert res_breakout["direction"] == "LONG"
    assert "Breakout" in res_breakout["pattern_type"]
    assert res_breakout["pattern_confluence_score"] >= 20.0

    # 2. Test Bearish Liquidity Sweep (Stop-Hunt Wick above 90130, closes back at 90080)
    df_sweep = pd.DataFrame({
        "timestamp": idx,
        "open": np.append(base - 10, 90100.0),
        "high": np.append(base + 30, 90300.0),   # Huge wick pierces above 90130 resistance
        "low": np.append(base - 30, 90060.0),
        "close": np.append(base, 90080.0),       # Closes back BELOW resistance (Fakeout!)
        "volume": np.append(np.full(49, 200.0), 450.0),
    })

    res_sweep = agent.analyze(df_sweep)
    assert res_sweep["direction"] == "SHORT"
    assert res_sweep["pattern_type"] == "Bearish_Liquidity_Sweep_Reversal"