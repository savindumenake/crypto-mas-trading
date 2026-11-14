import sys
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.feature_engineer import CryptoFeatureEngineer


class PatternBreakoutAgent:
    """
    Multi-Timeframe Price Action, Breakout & Liquidity Sweep Analyzing Agent.
    Evaluates execution timeframe candles alongside higher-timeframe (HTF) trend filter
    to detect:
      1. Volatility Squeeze Breakouts (Bollinger Squeeze + Volume expansion)
      2. Clean Structural Breakouts (Support/Resistance break with strong body & volume)
      3. Liquidity Sweeps / Stop-Hunt Reversals (Fakeout wick rejection)
      4. EMA Trend Pullback Continuations
    Outputs a structured dictionary with a Confluence Sub-Score (0.0 to 25.0 points).
    """

    def __init__(self, lookback_bars: int = 20):
        self.lookback_bars = lookback_bars

    def analyze(
        self,
        exec_df: pd.DataFrame,
        htf_df: Optional[pd.DataFrame] = None,
        max_points: float = 25.0,
    ) -> Dict[str, Any]:
        """
        Analyzes the latest closed candle on the execution timeframe and checks alignment
        with the Higher Timeframe (HTF) filter.
        """
        if exec_df.empty or len(exec_df) < self.lookback_bars + 5:
            return self._neutral_response("INSUFFICIENT_DATA")

        # Ensure technical features are computed
        if "atr_14" not in exec_df.columns:
            exec_df = CryptoFeatureEngineer.compute_features(exec_df)
        if htf_df is not None and not htf_df.empty and "atr_14" not in htf_df.columns:
            htf_df = CryptoFeatureEngineer.compute_features(htf_df)

        # Latest closed candle and prior N-bar structure window (excluding current candle)
        latest = exec_df.iloc[-1]
        prior_window = exec_df.iloc[-(self.lookback_bars + 1) : -1]

        close_p = float(latest["close"])
        open_p = float(latest["open"])
        high_p = float(latest["high"])
        low_p = float(latest["low"])
        atr = float(latest.get("atr_14", close_p * 0.005))
        vol_ratio = float(latest.get("vol_ratio_20", 1.0))
        body_ratio = float(latest.get("body_to_range_ratio", 0.0))
        upper_wick = float(latest.get("upper_wick_ratio", 0.0))
        lower_wick = float(latest.get("lower_wick_ratio", 0.0))
        market_regime = str(latest.get("market_regime", "Ranging"))

        # Structural Support & Resistance from prior N closed candles
        resistance_level = float(prior_window["high"].max())
        support_level = float(prior_window["low"].min())

        # Check Bollinger Band Squeeze state (Was BB width in the lowest 35% recently?)
        recent_bb_widths = prior_window["bb_width_pct"].dropna()
        is_squeeze_setup = False
        if len(recent_bb_widths) >= 10:
            squeeze_threshold = float(np.percentile(recent_bb_widths, 35))
            is_squeeze_setup = float(prior_window["bb_width_pct"].iloc[-1]) <= squeeze_threshold

        # -----------------------------------------------------------------
        # 1. DETECT PATTERN & DIRECTION ON EXECUTION TIMEFRAME
        # -----------------------------------------------------------------
        direction = "NEUTRAL"
        pattern_type = "No_Clear_Pattern"
        base_pattern_score = 0.0  # Out of 15.0 points

        # Case A: Bullish Breakout (Close breaks above prior N-bar Resistance)
        if close_p > resistance_level:
            breakout_margin_atr = (close_p - resistance_level) / max(atr, 1e-9)
            if abs(body_ratio) >= 0.40 and breakout_margin_atr >= 0.05:
                direction = "LONG"
                if is_squeeze_setup:
                    pattern_type = "Volatility_Squeeze_Breakout_Bull"
                    base_pattern_score = 15.0
                else:
                    pattern_type = "Resistance_Structure_Breakout"
                    base_pattern_score = 13.0

        # Case B: Bearish Breakdown (Close breaks below prior N-bar Support)
        elif close_p < support_level:
            breakdown_margin_atr = (support_level - close_p) / max(atr, 1e-9)
            if abs(body_ratio) >= 0.40 and breakdown_margin_atr >= 0.05:
                direction = "SHORT"
                if is_squeeze_setup:
                    pattern_type = "Volatility_Squeeze_Breakout_Bear"
                    base_pattern_score = 15.0
                else:
                    pattern_type = "Support_Structure_Breakdown"
                    base_pattern_score = 13.0

        # Case C: Bullish Liquidity Sweep / Stop-Hunt Reversal (Wick pierces Support, closes above)
        elif low_p < support_level and close_p > support_level and lower_wick >= 0.42:
            direction = "LONG"
            pattern_type = "Bullish_Liquidity_Sweep_Reversal"
            base_pattern_score = 13.5

        # Case D: Bearish Liquidity Sweep / Stop-Hunt Reversal (Wick pierces Resistance, closes below)
        elif high_p > resistance_level and close_p < resistance_level and upper_wick >= 0.42:
            direction = "SHORT"
            pattern_type = "Bearish_Liquidity_Sweep_Reversal"
            base_pattern_score = 13.5

        # Case E: EMA(21) Trend Pullback Continuation (During strong trending regimes)
        else:
            ema_21 = float(latest.get("ema_21", close_p))
            ema_50 = float(latest.get("ema_50", close_p))
            dist_to_ema21_atr = abs(close_p - ema_21) / max(atr, 1e-9)

            if (
                ema_21 > ema_50
                and close_p > ema_21
                and dist_to_ema21_atr <= 0.8
                and body_ratio >= 0.35
            ):
                direction = "LONG"
                pattern_type = "Bullish_EMA21_Trend_Pullback"
                base_pattern_score = 10.5
            elif (
                ema_21 < ema_50
                and close_p < ema_21
                and dist_to_ema21_atr <= 0.8
                and body_ratio <= -0.35
            ):
                direction = "SHORT"
                pattern_type = "Bearish_EMA21_Trend_Pullback"
                base_pattern_score = 10.5

        if direction == "NEUTRAL":
            return self._neutral_response(
                "NO_VALID_SETUP",
                support=support_level,
                resistance=resistance_level,
                regime=market_regime,
            )

        # -----------------------------------------------------------------
        # 2. VOLUME CONFIRMATION SCORE (UP TO 5.0 POINTS)
        # -----------------------------------------------------------------
        if vol_ratio >= 1.8:
            volume_score = 5.0
        elif vol_ratio >= 1.3:
            volume_score = 3.5
        elif vol_ratio >= 1.0:
            volume_score = 2.0
        else:
            # Low volume breakout penalty
            volume_score = 0.5

        # -----------------------------------------------------------------
        # 3. HIGHER-TIMEFRAME (HTF) TREND ALIGNMENT (UP TO 5.0 POINTS)
        # -----------------------------------------------------------------
        htf_aligned = True
        htf_veto = False
        htf_score = 3.5  # Default neutral HTF score if no HTF DataFrame provided
        htf_trend = "UNKNOWN"

        if htf_df is not None and len(htf_df) >= 25:
            htf_latest = htf_df.iloc[-1]
            htf_close = float(htf_latest["close"])
            htf_ema21 = float(htf_latest.get("ema_21", htf_close))
            htf_ema50 = float(htf_latest.get("ema_50", htf_close))
            htf_adx = float(htf_latest.get("adx_14", 20.0))

            if htf_ema21 > htf_ema50 and htf_close >= htf_ema21:
                htf_trend = "BULLISH"
            elif htf_ema21 < htf_ema50 and htf_close <= htf_ema21:
                htf_trend = "BEARISH"
            else:
                htf_trend = "NEUTRAL_TRANSITION"

            if (direction == "LONG" and htf_trend == "BULLISH") or (
                direction == "SHORT" and htf_trend == "BEARISH"
            ):
                htf_aligned = True
                htf_score = 5.0
            elif htf_trend == "NEUTRAL_TRANSITION":
                htf_aligned = True
                htf_score = 3.0
            else:
                # Counter-trend trade against a strong HTF trend (ADX > 30) triggers Veto warning
                htf_aligned = False
                htf_score = 0.0
                if htf_adx >= 30.0:
                    htf_veto = True

        total_score = min(max_points, round(base_pattern_score + volume_score + htf_score, 2))

        return {
            "direction": direction,
            "pattern_type": pattern_type,
            "market_regime": market_regime,
            "pattern_confluence_score": total_score,  # Out of 25.0 max points
            "volume_ratio": round(vol_ratio, 2),
            "is_squeeze_setup": bool(is_squeeze_setup),
            "htf_trend": htf_trend,
            "htf_aligned": htf_aligned,
            "htf_counter_trend_veto": htf_veto,
            "key_levels": {
                "support_level": round(support_level, 4),
                "resistance_level": round(resistance_level, 4),
                "latest_close": round(close_p, 4),
                "atr_14": round(atr, 4),
            },
        }

    @staticmethod
    def _neutral_response(
        reason: str,
        support: float = 0.0,
        resistance: float = 0.0,
        regime: str = "Ranging",
    ) -> Dict[str, Any]:
        return {
            "direction": "NEUTRAL",
            "pattern_type": "No_Clear_Pattern",
            "market_regime": regime,
            "pattern_confluence_score": 0.0,
            "volume_ratio": 0.0,
            "is_squeeze_setup": False,
            "htf_trend": "UNKNOWN",
            "htf_aligned": False,
            "htf_counter_trend_veto": False,
            "reason": reason,
            "key_levels": {
                "support_level": round(support, 4),
                "resistance_level": round(resistance, 4),
            },
        }


if __name__ == "__main__":
    # Standalone verification on synthetic Bullish Breakout data
    idx = pd.date_range("2026-01-01", periods=60, freq="15min", tz="UTC")
    np.random.seed(42)
    base_prices = np.linspace(90000, 90200, 59) + np.random.normal(0, 35, 59)
    # Create a clean breakout candle on bar 60 with a volume spike
    closes = np.append(base_prices, 90450.0)
    opens = np.append(base_prices - 10, 90210.0)
    highs = np.append(base_prices + 40, 90470.0)
    lows = np.append(base_prices - 40, 90200.0)
    vols = np.append(np.random.uniform(200, 300, 59), 650.0)

    sample_df = pd.DataFrame({
        "timestamp": idx,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": vols,
    })

    agent = PatternBreakoutAgent(lookback_bars=20)
    result = agent.analyze(sample_df, htf_df=sample_df)
    print("Pattern & Breakout Agent Output:")
    for k, v in result.items():
        print(f"  {k:<26}: {v}")