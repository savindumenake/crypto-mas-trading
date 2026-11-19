import asyncio
from copy import deepcopy
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from src.api import server


VALID_SIGNAL = {
    "should_execute": True,
    "direction": "LONG",
    "evaluated_bias": "LONG",
    "total_confluence_score": 76.5,
    "signal_tier": "B",
    "risk_plan": {
        "entry_price": 90000,
        "stop_loss": 89000,
        "tp1_price": 91200,
        "tp2_price": 92500,
        "position_size_coins": 0.01,
    },
}


@pytest.fixture
def setup_trade(monkeypatch):
    lifecycle = MagicMock()
    lifecycle.open_paper_trade.return_value = {
        "status": "OPEN"
    }

    lifecycle.get_multi_horizon_dashboard_payload.return_value = {}
    lifecycle.get_active_trades.return_value = []

    monkeypatch.setattr(server.scanner, "lifecycle", lifecycle)

    async def fake_state(force_refresh=False):
        return {
            "signals_by_horizon": {
                "scalp": {
                    "BTCUSDT": deepcopy(VALID_SIGNAL)
                }
            }
        }

    monkeypatch.setattr(
        server,
        "get_platform_state",
        fake_state
    )

    return lifecycle


def run_trade():
    return asyncio.run(
        server.simulate_vip_signal(
            horizon="scalp",
            symbol="BTCUSDT",
            send_telegram=False
        )
    )


def test_approved_trade_preserves_ai_decision(setup_trade):
    run_trade()

    setup_trade.open_paper_trade.assert_called_once()

    decision = setup_trade.open_paper_trade.call_args.args[0]

    assert decision["should_execute"] is True
    assert decision["total_confluence_score"] == 76.5
    assert decision["signal_tier"] == "B"
    assert decision["risk_plan"]["entry_price"] == 90000


def test_rejected_signal_cannot_open_trade(
    setup_trade,
    monkeypatch
):
    async def rejected_state(force_refresh=False):
        signal = deepcopy(VALID_SIGNAL)
        signal["should_execute"] = False

        return {
            "signals_by_horizon": {
                "scalp": {"BTCUSDT": signal}
            }
        }

    monkeypatch.setattr(
        server,
        "get_platform_state",
        rejected_state
    )

    with pytest.raises(HTTPException) as exc:
        run_trade()

    assert exc.value.status_code == 409
    setup_trade.open_paper_trade.assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    [
        ("entry_price", -100),
        ("stop_loss", 0),
        ("tp1_price", float("nan")),
        ("tp2_price", float("inf")),
        ("position_size_coins", -1),
    ]
)
def test_invalid_risk_plan_cannot_open_trade(
    setup_trade,
    monkeypatch,
    field,
    value
):
    async def invalid_state(force_refresh=False):
        signal = deepcopy(VALID_SIGNAL)
        signal["risk_plan"][field] = value

        return {
            "signals_by_horizon": {
                "scalp": {"BTCUSDT": signal}
            }
        }

    monkeypatch.setattr(
        server,
        "get_platform_state",
        invalid_state
    )

    with pytest.raises(HTTPException) as exc:
        run_trade()

    assert exc.value.status_code == 422
    setup_trade.open_paper_trade.assert_not_called()


def test_inconsistent_trade_levels_rejected(
    setup_trade,
    monkeypatch
):
    async def invalid_state(force_refresh=False):
        signal = deepcopy(VALID_SIGNAL)
        signal["risk_plan"]["stop_loss"] = 92000

        return {
            "signals_by_horizon": {
                "scalp": {"BTCUSDT": signal}
            }
        }

    monkeypatch.setattr(
        server,
        "get_platform_state",
        invalid_state
    )

    with pytest.raises(HTTPException) as exc:
        run_trade()

    assert exc.value.status_code == 422
    setup_trade.open_paper_trade.assert_not_called()
