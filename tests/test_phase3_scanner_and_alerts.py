import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.notifications.telegram_notifier import TelegramAlertNotifier
from src.core.live_scanner import LiveMultiHorizonScanner


def test_telegram_alert_formatting():
    """Verifies HTML formatting for new trade alerts, lifecycle events, and multi-horizon digests."""
    notifier = TelegramAlertNotifier(bot_token="123456:ABC-DEF", chat_id="@TestChannel")
    assert notifier.is_configured is True

    sample_decision = {
        "symbol": "BTCUSDT",
        "horizon": "scalp",
        "direction": "LONG",
        "evaluated_bias": "LONG",
        "signal_tier": "A_PLUS_INSTITUTIONAL",
        "total_confluence_score": 85.0,
        "component_scores": {
            "ml_shap_score_45": 36.0,
            "pattern_structure_score_25": 23.0,
            "sentiment_microstructure_score_15": 13.0,
            "debate_memory_score_15": 13.0,
        },
        "risk_plan": {
            "entry_price": 90000.0,
            "atr_value": 120.0,
            "stop_loss": 89820.0,
            "tp1_price": 90216.0,
            "tp2_price": 90360.0,
            "sl_atr_mult": 1.5,
            "rr_ratio_tp1": 1.2,
            "rr_ratio_tp2": 2.0,
            "position_size_coins": 0.055,
            "risk_amount_usdt": 10.0,
        },
        "agent_reports": {
            "pattern": {"pattern_type": "Breakout_Bull", "market_regime": "Trending_Bull"},
            "ml_shap": {"shap_top_features_json": {"rsi_14": 0.15}},
            "debate": {"llm_executive_review": {"judge_synthesis": "Valid breakout."}},
        },
    }

    alert_html = notifier.format_new_trade_alert(sample_decision, {"trade_id": 77})
    assert "SCALP (5m) | NEW TRADE OPENED (#77)" in alert_html
    assert "BTCUSDT" in alert_html
    assert "85.00 / 100.0" in alert_html

    lifecycle_html = notifier.format_lifecycle_event_alert(
        {
            "horizon": "scalp",
            "event_type": "TP1_PARTIAL_AND_BREAKEVEN_SET",
            "trade_id": 77,
            "symbol": "BTCUSDT",
            "direction": "LONG",
            "previous_status": "OPEN",
            "new_status": "TP1_HIT",
            "stop_loss": 90000.0,
            "mae_atr": 0.25,
            "mfe_atr": 1.4,
        },
        {"current_equity_usdt": 1006.0, "growth_rate_pct": 0.6, "win_rate_pct": 100.0},
    )
    assert "TP1 HIT & BREAKEVEN LOCKED (#77)" in lifecycle_html
    assert "$1,006.00" in lifecycle_html


def test_live_multi_horizon_scanner_cycle():
    """Verifies that LiveMultiHorizonScanner evaluates all 3 segregated horizons and caches UI state."""
    scanner = LiveMultiHorizonScanner(account_balance_per_horizon_usdt=1000.0)
    state = asyncio.run(
        scanner.run_full_scan_cycle(
            symbols=["BTCUSDT"],
            horizons=["scalp", "intraday", "swing"],
            fetch_live=False,
            use_llm_debate=False,
            send_telegram=False,
        )
    )

    assert "scan_timestamp_utc" in state
    assert state["horizons_scanned"] == ["scalp", "intraday", "swing"]
    assert len(state["all_signals"]) == 3

    # Verify each horizon has chart candles and 100-pt confluence structure
    for h in ("scalp", "intraday", "swing"):
        sig = state["signals_by_horizon"][h]["BTCUSDT"]
        assert sig["horizon"] == h
        assert 0.0 <= float(sig["total_confluence_score"]) <= 100.0
        assert len(sig["chart_candles"]) > 10

    # Verify isolated performance dictionary has segregated books
    horizons_perf = state["isolated_performance"]["horizons"]
    assert set(horizons_perf.keys()) == {"scalp", "intraday", "swing"}

    # Verify JSON state cache was written to disk
    assert scanner.state_cache_path.exists()
    cached_json = json.loads(scanner.state_cache_path.read_text(encoding="utf-8"))
    assert len(cached_json["all_signals"]) == 3