
import pytest
from pydantic import ValidationError

from src.api.server import RiskCalcRequest


def test_valid_risk_request():
    request = RiskCalcRequest(
        account_balance_usdt=1000,
        risk_pct=1,
        entry_price=90000,
        stop_loss=89000
    )

    assert request.risk_pct == 1


@pytest.mark.parametrize("field,value", [
    ("account_balance_usdt", -100),
    ("account_balance_usdt", 0),
    ("risk_pct", -1),
    ("risk_pct", 0),
    ("risk_pct", 101),
    ("entry_price", -90000),
    ("entry_price", 0),
    ("stop_loss", -89000),
    ("stop_loss", 0),
    ("risk_pct", float("nan")),
    ("entry_price", float("inf")),
])
def test_invalid_risk_inputs(field, value):
    data = {
        "account_balance_usdt": 1000,
        "risk_pct": 1,
        "entry_price": 90000,
        "stop_loss": 89000
    }

    data[field] = value

    with pytest.raises(ValidationError):
        RiskCalcRequest(**data)
