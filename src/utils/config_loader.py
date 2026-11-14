from pathlib import Path
from typing import Any, Dict
import yaml


class ConfigLoader:
    """Loads and validates the multi-asset YAML configuration."""

    def __init__(self, config_path: str = "config/assets_config.yaml"):
        self.project_root = Path(__file__).resolve().parents[2]
        self.config_file = self.project_root / config_path
        self._config: Dict[str, Any] = self._load_yaml()

    def _load_yaml(self) -> Dict[str, Any]:
        if not self.config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
        with open(self.config_file, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    @property
    def full_config(self) -> Dict[str, Any]:
        return self._config

    def get_active_assets(self) -> Dict[str, Dict[str, Any]]:
        """Returns only assets where status == 'active' (e.g., BTCUSDT in v1.0)."""
        assets = self._config.get("assets", {})
        return {
            symbol: details
            for symbol, details in assets.items()
            if details.get("status") == "active"
        }

    def get_database_paths(self) -> Dict[str, Path]:
        """Returns absolute paths for scalp, intraday, and swing databases."""
        db_cfg = self._config.get("databases", {})
        return {
            horizon: self.project_root / rel_path
            for horizon, rel_path in db_cfg.items()
        }

    def get_horizon_config(self, horizon: str) -> Dict[str, Any]:
        """Returns settings for 'scalp', 'intraday', or 'swing'."""
        horizons = self._config.get("horizons", {})
        if horizon not in horizons:
            raise ValueError(f"Invalid horizon '{horizon}'. Choose from: {list(horizons.keys())}")
        return horizons[horizon]

    def get_confluence_and_risk_config(self) -> Dict[str, Any]:
        """Returns the 100-point confluence weights and risk thresholds."""
        return self._config.get("confluence_and_risk", {})


if __name__ == "__main__":
    loader = ConfigLoader()
    print("Active Assets:", list(loader.get_active_assets().keys()))
    print("Database Paths:", {k: str(v) for k, v in loader.get_database_paths().items()})
    
    # Horizons 3ම (scalp, intraday, swing) පරීක්ෂා කිරීම:
    for horizon in ["scalp", "intraday", "swing"]:
        cfg = loader.get_horizon_config(horizon)
        print(f"{horizon.capitalize()} Timeframes: {cfg['execution_timeframes']} (Filter: {cfg['trend_filter_timeframe']})")