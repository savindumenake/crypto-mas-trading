import sys
from pathlib import Path
from typing import Any, Dict, Optional
import joblib
import numpy as np
import pandas as pd
import shap

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import ConfigLoader
from src.ingestion.feature_engineer import CryptoFeatureEngineer
from src.agents.ml_trainer import HorizonModelTrainer


class MLPredictionAndShapAgent:
    """
    Real-Time Machine Learning Inference & SHAP Explainability Agent.
    Features hot-reloading on model file updates, exact TreeSHAP attributions,
    and a 45-point ML Confluence scoring engine.
    """

    def __init__(self, config_path: str = "config/assets_config.yaml"):
        self.config_loader = ConfigLoader(config_path)
        self.trainer = HorizonModelTrainer(config_path)
        self.feature_cols = CryptoFeatureEngineer.ML_FEATURE_COLUMNS

        # Cache loaded models, SHAP explainers, and file modification timestamps
        self._model_cache: Dict[str, Dict[str, Any]] = {}
        self._explainer_cache: Dict[str, shap.TreeExplainer] = {}
        self._mtime_cache: Dict[str, float] = {}

    def _get_bundle_and_explainer(
        self, symbol: str, horizon: str
    ) -> tuple[Dict[str, Any], shap.TreeExplainer]:
        cache_key = f"{symbol}_{horizon}"
        model_path = self.trainer.get_model_path(symbol, horizon)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model bundle not found at {model_path}. Run ml_trainer.py first."
            )

        current_mtime = model_path.stat().st_mtime

        # Hot-reload automatically if model isn't cached OR if retrained file is newer
        if (
            cache_key not in self._model_cache
            or self._mtime_cache.get(cache_key) != current_mtime
        ):
            bundle = joblib.load(model_path)
            self._model_cache[cache_key] = bundle
            self._explainer_cache[cache_key] = shap.TreeExplainer(bundle["classifier"])
            self._mtime_cache[cache_key] = current_mtime

        return self._model_cache[cache_key], self._explainer_cache[cache_key]

    def predict_and_explain(
        self,
        symbol: str,
        horizon: str,
        df: pd.DataFrame,
        target_direction: Optional[str] = None,
        max_points: float = 45.0,
    ) -> Dict[str, Any]:
        """
        Runs live inference and SHAP attribution on the latest closed candle.
        """
        if df.empty or len(df) < 35:
            return self._empty_response(symbol, horizon, "INSUFFICIENT_CANDLES")

        bundle, explainer = self._get_bundle_and_explainer(symbol, horizon)
        model_features = bundle.get("feature_columns", self.feature_cols)

        # Compute technical features if not already present
        feat_df = (
            df
            if all(c in df.columns for c in model_features)
            else CryptoFeatureEngineer.compute_features(df)
        )

        latest_row = (
            feat_df.iloc[[-1]][model_features]
            .astype("float64")
            .replace([np.inf, -np.inf], 0.0)
            .fillna(0.0)
        )

        clf = bundle["classifier"]
        reg = bundle["regressor"]

        # 1. Directional Probability & Expected Price Variation (PPV in ATRs)
        probs = clf.predict_proba(latest_row)[0]
        prob_down = float(probs[0])
        prob_up = float(probs[1])
        predicted_ppv_atr = float(reg.predict(latest_row)[0])

        ml_direction = "LONG" if prob_up >= 0.50 else "SHORT"
        eval_direction = (
            target_direction if target_direction in ("LONG", "SHORT") else ml_direction
        )
        directional_prob = prob_up if eval_direction == "LONG" else prob_down

        # 2. Exact TreeSHAP Feature Attributions
        raw_shap = explainer.shap_values(latest_row)
        if isinstance(raw_shap, list):
            shap_vec = np.array(raw_shap[1])[0]
        elif getattr(raw_shap, "ndim", 1) == 3:
            shap_vec = raw_shap[0, :, 1]
        else:
            shap_vec = np.array(raw_shap)[0]

        exp_val = explainer.expected_value
        if isinstance(exp_val, (list, np.ndarray)):
            base_val = float(exp_val[1]) if len(exp_val) > 1 else float(exp_val[0])
        else:
            base_val = float(exp_val)

        # Positive SHAP pushes toward LONG (Up); Negative SHAP pushes toward SHORT (Down)
        signed_shap = shap_vec if eval_direction == "LONG" else -shap_vec

        shap_pairs = [
            (col, float(raw_s), float(aligned_s), float(latest_row[col].iloc[0]))
            for col, raw_s, aligned_s in zip(model_features, shap_vec, signed_shap)
        ]
        shap_pairs_sorted = sorted(shap_pairs, key=lambda x: abs(x[1]), reverse=True)

        top_5_shap_dict: Dict[str, float] = {
            col: round(raw_s, 4) for col, raw_s, _, _ in shap_pairs_sorted[:5]
        }
        supporting_drivers = [
            f"{col}={val:.2f} (SHAP {raw_s:+.3f})"
            for col, raw_s, aligned_s, val in shap_pairs_sorted[:5]
            if aligned_s > 0
        ]
        opposing_drivers = [
            f"{col}={val:.2f} (SHAP {raw_s:+.3f})"
            for col, raw_s, aligned_s, val in shap_pairs_sorted[:5]
            if aligned_s < 0
        ]

        # 3. Compute 45-Point ML & SHAP Confluence Score
        prob_edge = max(0.0, directional_prob - 0.48)
        prob_score = min(25.0, (prob_edge / 0.24) * 25.0)

        signed_ppv = (
            predicted_ppv_atr if eval_direction == "LONG" else -predicted_ppv_atr
        )
        min_req_atr = self.trainer.HORIZON_TRAINING_SPECS[horizon]["min_atr_move"]
        if signed_ppv > 0:
            ppv_score = min(10.0, (signed_ppv / max(min_req_atr * 1.5, 0.2)) * 10.0)
        else:
            ppv_score = 0.0

        top5_aligned_sum = sum(aligned_s for _, _, aligned_s, _ in shap_pairs_sorted[:5])
        top5_abs_sum = (
            sum(abs(raw_s) for _, raw_s, _, _ in shap_pairs_sorted[:5]) + 1e-9
        )
        synergy_ratio = max(0.0, top5_aligned_sum / top5_abs_sum)
        shap_synergy_score = round(synergy_ratio * 10.0, 2)

        total_ml_score = round(
            min(max_points, prob_score + ppv_score + shap_synergy_score), 2
        )

        return {
            "symbol": symbol,
            "horizon": horizon,
            "ml_dominant_direction": ml_direction,
            "evaluated_direction": eval_direction,
            "up_probability": round(prob_up, 4),
            "down_probability": round(prob_down, 4),
            "directional_confidence": round(directional_prob, 4),
            "predicted_ppv_atr": round(predicted_ppv_atr, 4),
            "shap_base_value": round(base_val, 4),
            "ml_confluence_score": total_ml_score,
            "score_breakdown": {
                "probability_score": round(prob_score, 2),
                "ppv_magnitude_score": round(ppv_score, 2),
                "shap_synergy_score": round(shap_synergy_score, 2),
            },
            "shap_top_features_json": top_5_shap_dict,
            "supporting_drivers": supporting_drivers[:3],
            "opposing_drivers": opposing_drivers[:3],
            "latest_feature_vector": CryptoFeatureEngineer.extract_latest_feature_snapshot(
                feat_df
            ),
        }

    @staticmethod
    def _empty_response(symbol: str, horizon: str, reason: str) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "horizon": horizon,
            "ml_dominant_direction": "NEUTRAL",
            "evaluated_direction": "NEUTRAL",
            "up_probability": 0.50,
            "down_probability": 0.50,
            "directional_confidence": 0.50,
            "predicted_ppv_atr": 0.0,
            "shap_base_value": 0.0,
            "ml_confluence_score": 0.0,
            "score_breakdown": {
                "probability_score": 0.0,
                "ppv_magnitude_score": 0.0,
                "shap_synergy_score": 0.0,
            },
            "shap_top_features_json": {},
            "supporting_drivers": [],
            "opposing_drivers": [],
            "latest_feature_vector": {},
            "reason": reason,
        }


if __name__ == "__main__":
    agent = MLPredictionAndShapAgent()
    for hz, tf in [("scalp", "5m"), ("intraday", "1h"), ("swing", "4h")]:
        cache_file = PROJECT_ROOT / "data" / "cache" / f"BTCUSDT_{tf}.parquet"
        if cache_file.exists():
            df_cached = pd.read_parquet(cache_file).tail(250).reset_index(drop=True)
            res = agent.predict_and_explain("BTCUSDT", hz, df_cached)
            print(f"\n[{hz.upper()} ({tf}) ML + SHAP Inference]:")
            print(
                f"  Dominant Direction : {res['ml_dominant_direction']} (P_Up: {res['up_probability']*100:.1f}%, P_Down: {res['down_probability']*100:.1f}%)"
            )
            print(f"  Predicted Move PPV : {res['predicted_ppv_atr']:+.3f} ATR")
            print(
                f"  ML Confluence Score: {res['ml_confluence_score']} / 45.0 | Breakdown: {res['score_breakdown']}"
            )
            print(f"  Top SHAP Features  : {res['shap_top_features_json']}")
            print(f"  Supporting Drivers : {res['supporting_drivers']}")