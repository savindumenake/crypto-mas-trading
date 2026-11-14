import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.ml_trainer import HorizonModelTrainer
from src.agents.ml_prediction_agent import MLPredictionAndShapAgent


def test_ml_trainer_and_shap_agent_pipeline(tmp_path, monkeypatch):
    """
    Tests end-to-end ML training (Classifier + PPV Regressor), hot-reload caching,
    TreeSHAP attribution, and 45-point confluence scoring in an isolated temp directory.
    """
    trainer = HorizonModelTrainer()
    # Redirect model storage to pytest tmp_path so production models are untouched
    monkeypatch.setattr(trainer, "get_model_dir", lambda sym: tmp_path)

    # Generate 150 synthetic 5m candles
    idx = pd.date_range("2026-01-01", periods=150, freq="5min", tz="UTC")
    np.random.seed(42)
    prices = 90000 + np.cumsum(np.random.normal(5, 60, 150))

    df = pd.DataFrame({
        "timestamp": idx,
        "open": prices - 15,
        "high": prices + 50,
        "low": prices - 50,
        "close": prices,
        "volume": np.random.uniform(150, 600, 150),
    })

    # 1. Train temporary scalp bundle
    metrics = trainer.train_horizon_model("BTCUSDT", "scalp", df)
    assert metrics["train_samples"] > 60
    assert (tmp_path / "xgb_scalp_bundle.joblib").exists()

    # 2. Run SHAP Agent inference using the temporary model
    agent = MLPredictionAndShapAgent()
    agent.trainer = trainer
    result = agent.predict_and_explain("BTCUSDT", "scalp", df, target_direction="LONG")

    assert result["evaluated_direction"] == "LONG"
    assert abs((result["up_probability"] + result["down_probability"]) - 1.0) < 1e-3
    assert isinstance(result["shap_base_value"], float)
    assert 0.0 <= result["ml_confluence_score"] <= 45.0
    assert len(result["shap_top_features_json"]) == 5
    assert "probability_score" in result["score_breakdown"]
    assert "ppv_magnitude_score" in result["score_breakdown"]
    assert "shap_synergy_score" in result["score_breakdown"]