import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np
from sqlalchemy import create_engine, select, desc
from sqlalchemy.orm import sessionmaker, Session

# Ensure project root is in sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import ConfigLoader
from src.database.models import Base, TradeExecutionLog, FundamentalStateLog


class SegregatedDatabaseManager:
    """
    Manages the three physically segregated SQLite databases:
    - scalp_trading.db (1m, 5m)
    - intraday_trading.db (15m, 1h)
    - swing_trading.db (4h, 1d)
    """

    def __init__(self, config_path: str = "config/assets_config.yaml"):
        self.config_loader = ConfigLoader(config_path)
        self.db_paths = self.config_loader.get_database_paths()
        self.engines: Dict[str, Any] = {}
        self.session_factories: Dict[str, sessionmaker] = {}
        self._initialize_databases()

    def _initialize_databases(self) -> None:
        """Creates database files and initializes tables for scalp, intraday, and swing."""
        for horizon, db_path in self.db_paths.items():
            db_path.parent.mkdir(parents=True, exist_ok=True)
            db_url = f"sqlite:///{db_path}"
            engine = create_engine(db_url, echo=False, future=True)
            Base.metadata.create_all(engine)
            self.engines[horizon] = engine
            self.session_factories[horizon] = sessionmaker(
                bind=engine, expire_on_commit=False
            )

    def _get_session(self, horizon: str) -> Session:
        if horizon not in self.session_factories:
            raise ValueError(
                f"Invalid horizon '{horizon}'. Must be one of: {list(self.session_factories.keys())}"
            )
        return self.session_factories[horizon]()

    # -----------------------------------------------------------------
    # 1. INSERT NEW TRADE (AT ENTRY)
    # -----------------------------------------------------------------
    def insert_trade(self, horizon: str, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Inserts a new OPEN trade into the specified horizon database."""
        with self._get_session(horizon) as session:
            new_trade = TradeExecutionLog(
                trade_id=trade_data["trade_id"],
                symbol=trade_data["symbol"],
                horizon=horizon,
                timeframe=trade_data["timeframe"],
                direction=trade_data["direction"],
                status=trade_data.get("status", "OPEN"),
                pattern_type=trade_data["pattern_type"],
                market_regime=trade_data["market_regime"],
                confluence_score=trade_data["confluence_score"],
                risk_tier=trade_data["risk_tier"],
                ml_up_probability=trade_data["ml_up_probability"],
                sentiment_score=trade_data.get("sentiment_score", 0.0),
                entry_price=trade_data["entry_price"],
                position_size=trade_data["position_size"],
                atr_at_entry=trade_data["atr_at_entry"],
                initial_sl=trade_data["initial_sl"],
                initial_tp1=trade_data["initial_tp1"],
                initial_tp2=trade_data.get("initial_tp2"),
                feature_vector_json=trade_data["feature_vector_json"],
                shap_top_features_json=trade_data.get("shap_top_features_json"),
                agent_debate_summary=trade_data.get("agent_debate_summary"),
            )
            session.add(new_trade)
            session.commit()
            session.refresh(new_trade)
            return new_trade.to_dict()

    # -----------------------------------------------------------------
    # 2. UPDATE TRADE OUTCOME & MAE/MFE (AT EXIT)
    # -----------------------------------------------------------------
    def update_trade_outcome(
        self, horizon: str, trade_id: str, exit_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Updates a trade when it closes with PnL, MAE, and MFE ratios."""
        with self._get_session(horizon) as session:
            stmt = select(TradeExecutionLog).where(TradeExecutionLog.trade_id == trade_id)
            trade = session.scalar(stmt)
            if not trade:
                return None

            trade.status = "CLOSED"
            trade.exit_timestamp = exit_data.get(
                "exit_timestamp", datetime.now(timezone.utc)
            )
            trade.actual_exit_price = exit_data["actual_exit_price"]
            trade.pnl_amount = exit_data.get("pnl_amount", 0.0)
            trade.pnl_percent = exit_data["pnl_percent"]
            trade.is_win = exit_data["is_win"]
            trade.exit_reason = exit_data["exit_reason"]
            trade.mae_atr_ratio = exit_data["mae_atr_ratio"]
            trade.mfe_atr_ratio = exit_data["mfe_atr_ratio"]

            session.commit()
            session.refresh(trade)
            return trade.to_dict()

    # -----------------------------------------------------------------
    # 3. HISTORICAL MEMORY RETRIEVAL & DYNAMIC SL/TP RECOMMENDATION
    # -----------------------------------------------------------------
    def get_historical_memory_stats(
        self,
        horizon: str,
        symbol: str,
        pattern_type: str,
        market_regime: Optional[str] = None,
        top_k: int = 15,
    ) -> Dict[str, Any]:
        """
        Retrieves the Top-K most recent closed trades for the same symbol & pattern
        to compute Historical Win Rate and optimal MAE/MFE-based SL/TP multipliers.
        """
        with self._get_session(horizon) as session:
            stmt = (
                select(TradeExecutionLog)
                .where(
                    TradeExecutionLog.symbol == symbol,
                    TradeExecutionLog.pattern_type == pattern_type,
                    TradeExecutionLog.status == "CLOSED",
                )
            )
            if market_regime:
                stmt = stmt.where(TradeExecutionLog.market_regime == market_regime)

            stmt = stmt.order_by(desc(TradeExecutionLog.entry_timestamp)).limit(top_k)
            past_trades: List[TradeExecutionLog] = list(session.scalars(stmt).all())

            horizon_cfg = self.config_loader.get_horizon_config(horizon)
            default_sl = horizon_cfg["default_sl_atr_mult"]
            default_tp1 = horizon_cfg["default_tp1_atr_mult"]
            default_tp2 = horizon_cfg["default_tp2_atr_mult"]

            # If fewer than 3 historical trades exist, return default config multipliers
            if len(past_trades) < 3:
                return {
                    "sample_count": len(past_trades),
                    "historical_win_rate": None,
                    "recommended_sl_atr_mult": default_sl,
                    "recommended_tp1_atr_mult": default_tp1,
                    "recommended_tp2_atr_mult": default_tp2,
                    "memory_status": "COLD_START_DEFAULTS",
                }

            wins = [t for t in past_trades if t.is_win]
            win_rate = len(wins) / len(past_trades)

            # Calculate MAE 80th percentile of winning trades to prevent stop-hunts
            win_maes = [t.mae_atr_ratio for t in wins if t.mae_atr_ratio is not None]
            win_mfes = [t.mfe_atr_ratio for t in wins if t.mfe_atr_ratio is not None]

            if win_maes and win_mfes:
                # Add a 10% safety buffer above the 80th percentile MAE wick
                rec_sl = round(float(np.percentile(win_maes, 80) * 1.10), 2)
                rec_tp1 = round(float(np.percentile(win_mfes, 50)), 2)
                rec_tp2 = round(float(np.percentile(win_mfes, 80)), 2)
            else:
                rec_sl, rec_tp1, rec_tp2 = default_sl, default_tp1, default_tp2

            return {
                "sample_count": len(past_trades),
                "historical_win_rate": round(win_rate, 4),
                "recommended_sl_atr_mult": max(rec_sl, 0.8),
                "recommended_tp1_atr_mult": max(rec_tp1, rec_sl * 1.2),
                "recommended_tp2_atr_mult": max(rec_tp2, rec_tp1 * 1.3),
                "memory_status": "ACTIVE_MEMORY_OPTIMIZED",
            }

    # -----------------------------------------------------------------
    # 4. FETCH RETRAINING DATASET FOR WEEKLY ML UPDATES
    # -----------------------------------------------------------------
    def fetch_retraining_data(self, horizon: str, symbol: str) -> List[Dict[str, Any]]:
        """Extracts closed trade feature vectors and outcomes for weekly model retraining."""
        with self._get_session(horizon) as session:
            stmt = select(TradeExecutionLog).where(
                TradeExecutionLog.symbol == symbol,
                TradeExecutionLog.status == "CLOSED",
            )
            records = session.scalars(stmt).all()
            return [r.to_dict() for r in records]


if __name__ == "__main__":
    db_mgr = SegregatedDatabaseManager()
    print("Initialized Segregated Databases:")
    for hz, path in db_mgr.db_paths.items():
        print(f" - {hz.upper()}: {path.name} (Exists: {path.exists()})")