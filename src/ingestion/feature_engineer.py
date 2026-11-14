import sys
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class CryptoFeatureEngineer:
    """
    Vectorized Technical Indicator & Scale-Invariant ML Feature Engineering Engine.
    Transforms raw OHLCV DataFrames into normalized, stationary features suitable
    for XGBoost/LightGBM, SHAP explainability, and Pattern/Breakout detection
    across any crypto asset (BTCUSDT, ETHUSDT, SOLUSDT) and timeframe.
    """

    # List of scale-invariant feature columns fed directly into ML models & saved in DB
    ML_FEATURE_COLUMNS: List[str] = [
        "ret_1",
        "ret_3",
        "ret_6",
        "ret_12",
        "log_ret_1",
        "natr_14_pct",
        "rsi_14",
        "stoch_k",
        "stoch_d",
        "macd_norm",
        "macd_signal_norm",
        "macd_hist_norm",
        "ema_9_dist_atr",
        "ema_21_dist_atr",
        "ema_50_dist_atr",
        "ema_200_dist_atr",
        "ema_9_21_cross_atr",
        "bb_width_pct",
        "bb_percent_b",
        "vol_ratio_20",
        "vwap_dist_atr",
        "body_to_range_ratio",
        "upper_wick_ratio",
        "lower_wick_ratio",
        "adx_14",
    ]

    @staticmethod
    def compute_features(df: pd.DataFrame, drop_na: bool = False) -> pd.DataFrame:
        """
        Computes raw technical indicators (EMA, ATR, Bollinger, RSI, MACD, ADX)
        and scale-invariant ML features on an OHLCV DataFrame.
        """
        if df.empty or len(df) < 30:
            return df.copy()

        data = df.copy()
        close = data["close"]
        high = data["high"]
        low = data["low"]
        open_p = data["open"]
        volume = data["volume"]

        eps = 1e-9  # Prevents division by zero

        # -----------------------------------------------------------------
        # 1. MULTI-HORIZON RETURNS & LOG RETURNS
        # -----------------------------------------------------------------
        data["ret_1"] = close.pct_change(1) * 100.0
        data["ret_3"] = close.pct_change(3) * 100.0
        data["ret_6"] = close.pct_change(6) * 100.0
        data["ret_12"] = close.pct_change(12) * 100.0
        data["log_ret_1"] = np.log(close / close.shift(1).replace(0, np.nan)) * 100.0

        # -----------------------------------------------------------------
        # 2. VOLATILITY: TRUE RANGE, ATR(14) & NORMALIZED ATR (%)
        # -----------------------------------------------------------------
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Wilder's Smoothing for ATR(14)
        data["atr_14"] = true_range.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        atr_safe = data["atr_14"].replace(0, eps).bfill().fillna(close * 0.005)
        data["natr_14_pct"] = (atr_safe / close) * 100.0

        # -----------------------------------------------------------------
        # 3. TREND: EMAs (9, 21, 50, 200) & ATR-NORMALIZED DISTANCES
        # -----------------------------------------------------------------
        for span in [9, 21, 50, 200]:
            ema_col = f"ema_{span}"
            data[ema_col] = close.ewm(span=span, adjust=False).mean()
            data[f"ema_{span}_dist_atr"] = (close - data[ema_col]) / atr_safe

        data["ema_9_21_cross_atr"] = (data["ema_9"] - data["ema_21"]) / atr_safe

        # -----------------------------------------------------------------
        # 4. MOMENTUM: RSI(14), STOCHASTIC(14,3) & ATR-NORMALIZED MACD
        # -----------------------------------------------------------------
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, eps)
        data["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

        # Stochastic Oscillator (14, 3, 3)
        low_14 = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        data["stoch_k"] = ((close - low_14) / (high_14 - low_14 + eps)) * 100.0
        data["stoch_d"] = data["stoch_k"].rolling(3).mean()

        # MACD (12, 26, 9) Normalized by ATR so it is scale-invariant across coins
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        data["macd_raw"] = ema_12 - ema_26
        data["macd_signal_raw"] = data["macd_raw"].ewm(span=9, adjust=False).mean()
        data["macd_hist_raw"] = data["macd_raw"] - data["macd_signal_raw"]

        data["macd_norm"] = data["macd_raw"] / atr_safe
        data["macd_signal_norm"] = data["macd_signal_raw"] / atr_safe
        data["macd_hist_norm"] = data["macd_hist_raw"] / atr_safe

        # -----------------------------------------------------------------
        # 5. BOLLINGER BANDS (20, 2.0) & SQUEEZE METRICS
        # -----------------------------------------------------------------
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std(ddof=0)
        data["bb_upper"] = bb_mid + (2.0 * bb_std)
        data["bb_lower"] = bb_mid - (2.0 * bb_std)
        data["bb_mid"] = bb_mid

        data["bb_width_pct"] = ((data["bb_upper"] - data["bb_lower"]) / (bb_mid + eps)) * 100.0
        data["bb_percent_b"] = (close - data["bb_lower"]) / (
            (data["bb_upper"] - data["bb_lower"]) + eps
        )

        # -----------------------------------------------------------------
        # 6. VOLUME SPIKE RATIO & ROLLING VWAP DISTANCE
        # -----------------------------------------------------------------
        vol_ma_20 = volume.rolling(20).mean()
        data["vol_ratio_20"] = volume / (vol_ma_20 + eps)

        typical_price = (high + low + close) / 3.0
        rolling_tp_vol = (typical_price * volume).rolling(20).sum()
        rolling_vol_sum = volume.rolling(20).sum()
        data["vwap_20"] = rolling_tp_vol / (rolling_vol_sum + eps)
        data["vwap_dist_atr"] = (close - data["vwap_20"]) / atr_safe

        # -----------------------------------------------------------------
        # 7. CANDLE MICROSTRUCTURE (WICKS, BODY & LIQUIDITY SWEEP GEOMETRY)
        # -----------------------------------------------------------------
        candle_range = (high - low) + eps
        body = close - open_p
        upper_wick = high - np.maximum(open_p, close)
        lower_wick = np.minimum(open_p, close) - low

        data["body_to_range_ratio"] = body / candle_range        # -1.0 (full bear) to +1.0 (full bull)
        data["upper_wick_ratio"] = upper_wick / candle_range     # 0.0 to 1.0 (rejection from top)
        data["lower_wick_ratio"] = lower_wick / candle_range     # 0.0 to 1.0 (demand/sweep from bottom)

        # -----------------------------------------------------------------
        # 8. TREND STRENGTH: ADX(14) & MARKET REGIME CLASSIFICATION
        # -----------------------------------------------------------------
        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        plus_di = 100.0 * (
            pd.Series(plus_dm, index=data.index).ewm(alpha=1 / 14, adjust=False).mean() / atr_safe
        )
        minus_di = 100.0 * (
            pd.Series(minus_dm, index=data.index).ewm(alpha=1 / 14, adjust=False).mean() / atr_safe
        )
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di + eps)) * 100.0
        data["adx_14"] = dx.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean().fillna(20.0)

        # Classify Market Regime for Database Memory Lookup
        conditions = [
            (data["adx_14"] >= 25.0) & (data["ema_21"] > data["ema_50"]),
            (data["adx_14"] >= 25.0) & (data["ema_21"] < data["ema_50"]),
            (data["natr_14_pct"] > data["natr_14_pct"].rolling(50, min_periods=10).quantile(0.85)),
        ]
        choices = ["Trending_Bull", "Trending_Bear", "High_Vol"]
        data["market_regime"] = np.select(conditions, choices, default="Ranging")

        if drop_na:
            data = data.dropna(subset=CryptoFeatureEngineer.ML_FEATURE_COLUMNS).reset_index(drop=True)

        return data

    @classmethod
    def extract_latest_feature_snapshot(cls, featured_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extracts the latest closed candle's ML feature vector as a clean JSON-serializable
        dictionary to store in `feature_vector_json` for weekly retraining and SHAP analysis.
        """
        if featured_df.empty:
            return {}

        latest = featured_df.iloc[-1]
        snapshot: Dict[str, Any] = {}
        for col in cls.ML_FEATURE_COLUMNS:
            val = latest.get(col, 0.0)
            if pd.isna(val) or np.isinf(val):
                snapshot[col] = 0.0
            else:
                snapshot[col] = round(float(val), 6)
        return snapshot

    @staticmethod
    def add_training_targets(
        featured_df: pd.DataFrame, forward_bars: int = 4, min_atr_move: float = 0.5
    ) -> pd.DataFrame:
        """
        Generates supervised ML training labels (target_direction and future MAE/MFE)
        for XGBoost/LightGBM model training without look-ahead bias.
        """
        df = featured_df.copy()
        future_close = df["close"].shift(-forward_bars)
        atr = df["atr_14"].replace(0, 1e-9)

        # Forward price variation in ATR multiples
        df["target_forward_atr_change"] = (future_close - df["close"]) / atr
        # 1 = Profitable Bullish Move (> min_atr_move), 0 = Otherwise
        df["target_up_label"] = (df["target_forward_atr_change"] > min_atr_move).astype(int)
        return df


if __name__ == "__main__":
    # Quick standalone verification using cached BTCUSDT data (or synthetic data if cache empty)
    cache_path = PROJECT_ROOT / "data" / "cache" / "BTCUSDT_15m.parquet"
    if cache_path.exists():
        raw_df = pd.read_parquet(cache_path)
        print(f"Loaded {len(raw_df)} candles from {cache_path.name}")
    else:
        idx = pd.date_range("2026-01-01", periods=250, freq="15min", tz="UTC")
        prices = 90000 + np.cumsum(np.random.randn(250) * 120)
        raw_df = pd.DataFrame({
            "timestamp": idx,
            "open": prices - 20,
            "high": prices + 80,
            "low": prices - 80,
            "close": prices,
            "volume": np.random.uniform(100, 1000, size=250),
        })

    feat_df = CryptoFeatureEngineer.compute_features(raw_df, drop_na=True)
    snapshot = CryptoFeatureEngineer.extract_latest_feature_snapshot(feat_df)
    latest_row = feat_df.iloc[-1]

    print(f"Computed {len(CryptoFeatureEngineer.ML_FEATURE_COLUMNS)} Scale-Invariant ML Features!")
    print(f"Latest Market Regime : {latest_row['market_regime']} (ADX: {latest_row['adx_14']:.2f})")
    print(f"Latest ATR(14)       : {latest_row['atr_14']:.2f} (NATR%: {latest_row['natr_14_pct']:.3f}%)")
    print("Sample Feature JSON  :", dict(list(snapshot.items())[:6]))