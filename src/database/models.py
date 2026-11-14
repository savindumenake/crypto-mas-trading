from datetime import datetime, timezone
from typing import Any, Dict, Optional
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    JSON,
    String,
    Text,
    Index,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models across Scalp, Intraday, and Swing DBs."""
    pass


class TradeExecutionLog(Base):
    """
    Stores trade executions, ML feature snapshots (for periodic retraining),
    and MAE/MFE metrics (for dynamic SL/TP optimization).
    Instantiated separately inside scalp_trading.db, intraday_trading.db, and swing_trading.db.
    """

    __tablename__ = "trade_execution_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)

    # --- Asset & Horizon Metadata ---
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)       # e.g., BTCUSDT, ETHUSDT, SOLUSDT
    horizon: Mapped[str] = mapped_column(String(20), nullable=False, index=True)      # scalp, intraday, swing
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False, index=True)    # 1m, 5m, 15m, 1h, 4h, 1d
    direction: Mapped[str] = mapped_column(String(10), nullable=False)                # LONG or SHORT
    status: Mapped[str] = mapped_column(String(20), default="OPEN", index=True)       # OPEN, CLOSED, CANCELLED

    # --- Pattern & Market Context (Used for Memory Similarity Lookup) ---
    pattern_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True) # e.g., Bull_Flag_Breakout, Support_Sweep
    market_regime: Mapped[str] = mapped_column(String(32), nullable=False, index=True)# e.g., Trending_Bull, Ranging, High_Vol

    # --- Confluence & Agent Scores at Entry ---
    confluence_score: Mapped[float] = mapped_column(Float, nullable=False)            # 0.0 to 100.0
    risk_tier: Mapped[str] = mapped_column(String(10), nullable=False)                # TIER_A (1.0x) or TIER_B (0.5x)
    ml_up_probability: Mapped[float] = mapped_column(Float, nullable=False)           # Model predicted probability
    sentiment_score: Mapped[float] = mapped_column(Float, default=0.0)                # -1.0 to +1.0 from FinBERT/VADER

    # --- Price, Risk & SL/TP Execution Details ---
    entry_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    exit_timestamp: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    position_size: Mapped[float] = mapped_column(Float, nullable=False)
    atr_at_entry: Mapped[float] = mapped_column(Float, nullable=False)

    initial_sl: Mapped[float] = mapped_column(Float, nullable=False)
    initial_tp1: Mapped[float] = mapped_column(Float, nullable=False)
    initial_tp2: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    actual_exit_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # --- Post-Trade Outcome & MAE / MFE (For Dynamic SL/TP Optimizer) ---
    pnl_amount: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pnl_percent: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_win: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True, index=True)
    exit_reason: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)     # TP1_HIT, TP2_HIT, SL_HIT, TRAILING_SL

    # MAE (Max Adverse Excursion) & MFE (Max Favorable Excursion) in ATR multiples
    mae_atr_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)      # How far price dipped against trade (in ATRs)
    mfe_atr_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)      # How far price moved in favor (in ATRs)

    # --- Retraining & Explainability Payloads (JSON) ---
    feature_vector_json: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False) # Full ML features for weekly retraining
    shap_top_features_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    agent_debate_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Composite index for fast pre-trade historical memory retrieval
    __table_args__ = (
        Index("idx_memory_lookup", "symbol", "pattern_type", "market_regime", "status"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Converts the ORM record into a clean dictionary."""
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "horizon": self.horizon,
            "timeframe": self.timeframe,
            "direction": self.direction,
            "status": self.status,
            "pattern_type": self.pattern_type,
            "market_regime": self.market_regime,
            "confluence_score": self.confluence_score,
            "risk_tier": self.risk_tier,
            "entry_price": self.entry_price,
            "initial_sl": self.initial_sl,
            "initial_tp1": self.initial_tp1,
            "initial_tp2": self.initial_tp2,
            "actual_exit_price": self.actual_exit_price,
            "pnl_percent": self.pnl_percent,
            "is_win": self.is_win,
            "exit_reason": self.exit_reason,
            "mae_atr_ratio": self.mae_atr_ratio,
            "mfe_atr_ratio": self.mfe_atr_ratio,
            "feature_vector_json": self.feature_vector_json,
            "shap_top_features_json": self.shap_top_features_json,
        }


class FundamentalStateLog(Base):
    """
    Stores asynchronous crypto news, FinBERT/VADER sentiment, and derivatives state
    (Funding Rate, Open Interest) so Scalp/Intraday/Swing engines can read it in <2ms.
    """

    __tablename__ = "fundamental_state_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    sentiment_score: Mapped[float] = mapped_column(Float, nullable=False)             # -1.0 to +1.0
    sentiment_label: Mapped[str] = mapped_column(String(20), nullable=False)          # BULLISH, BEARISH, NEUTRAL
    impact_horizon: Mapped[str] = mapped_column(String(32), nullable=False)           # SCALP_NOISE, INTRADAY, MACRO_SWING
    news_shock_active: Mapped[bool] = mapped_column(Boolean, default=False)           # True during CPI/FOMC/Major hack
    funding_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    open_interest_change_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    summary_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    retrieval_query_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # FinAgent-style diversified query


if __name__ == "__main__":
    print("SQLAlchemy Models loaded successfully!")
    print("Registered Tables:", list(Base.metadata.tables.keys()))