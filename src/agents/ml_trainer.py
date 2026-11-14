import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import ConfigLoader
from src.ingestion.market_data import MultiTimeframeDataFetcher
from src.ingestion.feature_engineer import CryptoFeatureEngineer


class HorizonModelTrainer:
    """
    Trains and saves horizon-specific (Scalp, Intraday, Swing) XGBoost models
    using deep historical OHLCV datasets (10,000 to 25,000+ candles) and
    Exponential Time-Decay Sample Weighting so recent market regimes carry higher weight.
    """

    # Optimal candle counts & target thresholds per horizon
    HORIZON_TRAINING_SPECS: Dict[str, Dict[str, Any]] = {
        "scalp": {
            "timeframe": "5m",
            "forward_bars": 4,
            "min_atr_move": 0.30,
            "target_candles": 25000,  # ~3 months of 5m candles
        },
        "intraday": {
            "timeframe": "1h",
            "forward_bars": 4,
            "min_atr_move": 0.45,
            "target_candles": 18000,  # ~2 years of 1h candles
        },
        "swing": {
            "timeframe": "4h",
            "forward_bars": 6,
            "min_atr_move": 0.60,
            "target_candles": 10000,  # ~4.5 years of 4h candles
        },
    }

    def __init__(self, config_path: str = "config/assets_config.yaml"):
        self.config_loader = ConfigLoader(config_path)
        self.feature_cols = CryptoFeatureEngineer.ML_FEATURE_COLUMNS

    def get_model_dir(self, symbol: str) -> Path:
        assets_cfg = self.config_loader.full_config.get("assets", {})
        rel_path = assets_cfg.get(symbol, {}).get("model_registry_path", f"models/{symbol}/")
        model_dir = PROJECT_ROOT / rel_path
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    def get_model_path(self, symbol: str, horizon: str) -> Path:
        return self.get_model_dir(symbol) / f"xgb_{horizon}_bundle.joblib"

    def train_horizon_model(
        self,
        symbol: str,
        horizon: str,
        raw_df: pd.DataFrame,
        retrain_records: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Trains both the Directional Classifier and PPV Regressor on chronological data
        with time-decay sample weights (recent market structure weighted higher).
        """
        if horizon not in self.HORIZON_TRAINING_SPECS:
            raise ValueError(f"Unknown horizon '{horizon}'")

        spec = self.HORIZON_TRAINING_SPECS[horizon]
        feat_df = CryptoFeatureEngineer.compute_features(raw_df, drop_na=True)
        labeled_df = CryptoFeatureEngineer.add_training_targets(
            feat_df,
            forward_bars=spec["forward_bars"],
            min_atr_move=spec["min_atr_move"],
        )

        labeled_df = labeled_df.dropna(
            subset=self.feature_cols + ["target_forward_atr_change", "target_up_label"]
        ).reset_index(drop=True)

        if len(labeled_df) < 60:
            raise ValueError(f"Insufficient training rows ({len(labeled_df)}) for {symbol} {horizon}")

        X = labeled_df[self.feature_cols].astype("float64")
        y_cls = labeled_df["target_up_label"].astype(int)
        y_reg = labeled_df["target_forward_atr_change"].astype("float64")

        # Incorporate real closed trade outcomes from the segregated database if available
        if retrain_records and len(retrain_records) > 0:
            db_rows, db_cls_labels, db_reg_labels = [], [], []
            for rec in retrain_records:
                f_json = rec.get("feature_vector_json", {})
                if all(k in f_json for k in self.feature_cols):
                    db_rows.append([float(f_json[k]) for k in self.feature_cols])
                    is_win = bool(rec.get("is_win", False))
                    direction = rec.get("direction", "LONG")
                    up_label = 1 if (direction == "LONG" and is_win) or (direction == "SHORT" and not is_win) else 0
                    db_cls_labels.append(up_label)
                    mfe = float(rec.get("mfe_atr_ratio") or 1.0)
                    db_reg_labels.append(mfe if up_label == 1 else -mfe)

            if db_rows:
                X_db = pd.DataFrame(db_rows, columns=self.feature_cols)
                X = pd.concat([X, X_db], ignore_index=True)
                y_cls = pd.concat([y_cls, pd.Series(db_cls_labels)], ignore_index=True)
                y_reg = pd.concat([y_reg, pd.Series(db_reg_labels)], ignore_index=True)

        # Chronological Walk-Forward Split (85% Train, 15% Out-of-Sample Validation)
        split_idx = int(len(X) * 0.85)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_cls_train, y_cls_val = y_cls.iloc[:split_idx], y_cls.iloc[split_idx:]
        y_reg_train, y_reg_val = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]

        # Time-Decay Sample Weights: Oldest candle gets weight 0.40, newest candle gets 1.00
        time_weights = np.linspace(0.40, 1.00, len(X_train))

        pos_count = int(y_cls_train.sum())
        neg_count = len(y_cls_train) - pos_count
        scale_pos_weight = float(neg_count / max(pos_count, 1))

        # 1. Train Directional Classifier
        clf = XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
        )
        clf.fit(X_train, y_cls_train, sample_weight=time_weights)

        # 2. Train Expected Price Variation (PPV) ATR Regressor
        reg = XGBRegressor(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1,
        )
        reg.fit(X_train, y_reg_train, sample_weight=time_weights)

        # Validation Metrics
        val_preds = clf.predict(X_val)
        val_probs = clf.predict_proba(X_val)[:, 1]
        val_reg_preds = reg.predict(X_val)

        acc = float(accuracy_score(y_cls_val, val_preds))
        try:
            auc = float(roc_auc_score(y_cls_val, val_probs))
        except ValueError:
            auc = 0.50
        mae_atr = float(mean_absolute_error(y_reg_val, val_reg_preds))

        model_bundle = {
            "symbol": symbol,
            "horizon": horizon,
            "timeframe": spec["timeframe"],
            "feature_columns": self.feature_cols,
            "classifier": clf,
            "regressor": reg,
            "metrics": {
                "total_candles_used": len(labeled_df),
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "val_accuracy": round(acc, 4),
                "val_roc_auc": round(auc, 4),
                "val_ppv_mae_atr": round(mae_atr, 4),
            },
        }

        save_path = self.get_model_path(symbol, horizon)
        joblib.dump(model_bundle, save_path)
        return model_bundle["metrics"]

    async def train_all_horizons_for_symbol(
        self, symbol: str = "BTCUSDT", custom_limit: Optional[int] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Downloads bulk historical candles (25k for 5m, 18k for 1h, 10k for 4h)
        and trains all 3 horizon models.
        """
        fetcher = MultiTimeframeDataFetcher(max_memory_candles=70000)
        results: Dict[str, Dict[str, Any]] = {}
        try:
            for horizon, spec in self.HORIZON_TRAINING_SPECS.items():
                tf = spec["timeframe"]
                target_count = custom_limit if custom_limit is not None else spec["target_candles"]
                df = await fetcher.fetch_historical_bulk(
                    symbol, tf, total_candles=target_count
                )
                metrics = self.train_horizon_model(symbol, horizon, df)
                results[horizon] = metrics
        finally:
            await fetcher.close_exchange()
        return results


async def _demo_train():
    trainer = HorizonModelTrainer()
    print("Training BTCUSDT Models with Deep Historical Datasets (Scalp 25k, Intraday 18k, Swing 10k)...")
    metrics_map = await trainer.train_all_horizons_for_symbol("BTCUSDT")
    for hz, m in metrics_map.items():
        model_file = trainer.get_model_path("BTCUSDT", hz)
        print(
            f" -> [{hz.upper():<8}] Saved: {model_file.name} | "
            f"Total Candles: {m['total_candles_used']} (Train: {m['train_samples']}, Val: {m['val_samples']}) | "
            f"Val Acc: {m['val_accuracy']*100:.1f}% | AUC: {m['val_roc_auc']:.3f} | PPV MAE: {m['val_ppv_mae_atr']:.3f} ATR"
        )


if __name__ == "__main__":
    asyncio.run(_demo_train())