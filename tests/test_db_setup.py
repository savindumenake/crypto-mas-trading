import sys
import uuid
from pathlib import Path
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.db_manager import SegregatedDatabaseManager


@pytest.fixture
def temp_db_manager(tmp_path: Path) -> SegregatedDatabaseManager:
    """
    Creates a SegregatedDatabaseManager pointed at temporary test databases
    so real production databases in data/databases/ are never polluted.
    """
    mgr = SegregatedDatabaseManager()
    mgr.db_paths = {
        "scalp": tmp_path / "test_scalp.db",
        "intraday": tmp_path / "test_intraday.db",
        "swing": tmp_path / "test_swing.db",
    }
    mgr._initialize_databases()
    return mgr


def test_horizon_database_isolation(temp_db_manager: SegregatedDatabaseManager):
    """Verifies that a Scalp trade is isolated from Intraday and Swing databases."""
    trade_id = f"TEST_SCALP_{uuid.uuid4().hex[:8]}"
    sample_trade = {
        "trade_id": trade_id,
        "symbol": "BTCUSDT",
        "timeframe": "5m",
        "direction": "LONG",
        "pattern_type": "Micro_Range_Breakout",
        "market_regime": "Trending_Bull",
        "confluence_score": 84.5,
        "risk_tier": "TIER_A",
        "ml_up_probability": 0.78,
        "sentiment_score": 0.45,
        "entry_price": 90000.0,
        "position_size": 0.05,
        "atr_at_entry": 150.0,
        "initial_sl": 89775.0,
        "initial_tp1": 90300.0,
        "initial_tp2": 90525.0,
        "feature_vector_json": {"rsi_14": 62.1, "vol_ratio": 1.85, "atr": 150.0},
        "shap_top_features_json": {"vol_ratio": 0.31, "rsi_14": 0.18},
    }

    # Insert into SCALP database
    inserted = temp_db_manager.insert_trade("scalp", sample_trade)
    assert inserted["trade_id"] == trade_id
    assert inserted["status"] == "OPEN"

    # Close trade with MAE/MFE outcome
    updated = temp_db_manager.update_trade_outcome(
        "scalp",
        trade_id,
        {
            "actual_exit_price": 90300.0,
            "pnl_amount": 15.0,
            "pnl_percent": 0.33,
            "is_win": True,
            "exit_reason": "TP1_HIT",
            "mae_atr_ratio": 0.9,
            "mfe_atr_ratio": 2.4,
        },
    )
    assert updated is not None
    assert updated["status"] == "CLOSED"
    assert updated["is_win"] is True

    # Verify SCALP has 1 closed record, while INTRADAY and SWING have 0
    scalp_records = temp_db_manager.fetch_retraining_data("scalp", "BTCUSDT")
    intraday_records = temp_db_manager.fetch_retraining_data("intraday", "BTCUSDT")
    swing_records = temp_db_manager.fetch_retraining_data("swing", "BTCUSDT")

    assert len(scalp_records) == 1
    assert len(intraday_records) == 0
    assert len(swing_records) == 0


def test_dynamic_mae_mfe_sl_tp_optimization(temp_db_manager: SegregatedDatabaseManager):
    """
    Simulates 5 historical Intraday trades where winning breakouts experienced
    stop-hunt wicks up to 1.8x ATR before rallying to 3.2x-4.5x ATR.
    Verifies that the Memory Optimizer widens SL above 1.8x ATR.
    """
    simulated_outcomes = [
        {"is_win": True, "mae": 1.60, "mfe": 3.00},
        {"is_win": True, "mae": 1.75, "mfe": 3.40},
        {"is_win": True, "mae": 1.80, "mfe": 4.20},
        {"is_win": True, "mae": 1.50, "mfe": 2.80},
        {"is_win": False, "mae": 2.10, "mfe": 0.50},
    ]

    for idx, outcome in enumerate(simulated_outcomes):
        t_id = f"HIST_INTRA_{idx}"
        temp_db_manager.insert_trade(
            "intraday",
            {
                "trade_id": t_id,
                "symbol": "BTCUSDT",
                "timeframe": "15m",
                "direction": "LONG",
                "pattern_type": "Bull_Flag_Breakout",
                "market_regime": "Trending_Bull",
                "confluence_score": 82.0,
                "risk_tier": "TIER_A",
                "ml_up_probability": 0.75,
                "entry_price": 90000.0,
                "position_size": 0.1,
                "atr_at_entry": 200.0,
                "initial_sl": 89640.0,
                "initial_tp1": 90500.0,
                "feature_vector_json": {"rsi_14": 58.0, "vol_ratio": 1.9},
            },
        )
        temp_db_manager.update_trade_outcome(
            "intraday",
            t_id,
            {
                "actual_exit_price": 90600.0 if outcome["is_win"] else 89580.0,
                "pnl_percent": 0.66 if outcome["is_win"] else -0.46,
                "is_win": outcome["is_win"],
                "exit_reason": "TP1_HIT" if outcome["is_win"] else "SL_HIT",
                "mae_atr_ratio": outcome["mae"],
                "mfe_atr_ratio": outcome["mfe"],
            },
        )

    # Query Memory Stats for the next Bull_Flag_Breakout on BTCUSDT
    stats = temp_db_manager.get_historical_memory_stats(
        horizon="intraday",
        symbol="BTCUSDT",
        pattern_type="Bull_Flag_Breakout",
        market_regime="Trending_Bull",
        top_k=15,
    )

    print("\n[Memory Optimizer Output]:", stats)

    assert stats["sample_count"] == 5
    assert stats["historical_win_rate"] == 0.80  # 4 wins out of 5
    assert stats["memory_status"] == "ACTIVE_MEMORY_OPTIMIZED"
    # Since winning trades dipped up to 1.80 ATR, recommended SL with 10% buffer should be > 1.80
    assert stats["recommended_sl_atr_mult"] >= 1.85
    assert stats["recommended_tp2_atr_mult"] > stats["recommended_tp1_atr_mult"]