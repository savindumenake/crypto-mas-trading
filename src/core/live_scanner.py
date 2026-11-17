import asyncio
import importlib
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import ConfigLoader
from src.ingestion.feature_engineer import CryptoFeatureEngineer
from src.core.langgraph_orchestrator import LangGraphTradingOrchestrator
from src.execution.paper_executor import PaperTradeLifecycleManager
from src.notifications.telegram_notifier import TelegramAlertNotifier


def _resolve_fetcher_instance(config_path: str = "config/assets_config.yaml") -> Any:
    """
    Dynamically discovers the data ingestion class inside `src/ingestion/`
    regardless of module naming (`data_fetcher.py`, `binance_fetcher.py`, etc.).
    """
    ingestion_dir = PROJECT_ROOT / "src" / "ingestion"
    for py_file in sorted(ingestion_dir.glob("*.py")):
        if py_file.name in ("__init__.py", "feature_engineer.py"):
            continue
        mod_name = f"src.ingestion.{py_file.stem}"
        try:
            mod = importlib.import_module(mod_name)
            for _, obj in inspect.getmembers(mod, inspect.isclass):
                if obj.__module__ == mod_name and any(
                    hasattr(obj, m)
                    for m in ("fetch_closed_ohlcv", "fetch_ohlcv", "get_historical_data")
                ):
                    try:
                        return obj(config_path)
                    except TypeError:
                        return obj()
        except Exception:
            continue
    return None


class LiveMultiHorizonScanner:
    """
    Production Multi-Asset & Multi-Horizon Scanner Engine:
      1. Iterates across configured symbols (BTCUSDT, ETHUSDT, SOLUSDT) and segregated
         horizons (`scalp`: 5m/1h, `intraday`: 15m/4h, `swing`: 1h/4h).
      2. Fetches fresh boundary-aligned candles (or falls back to local Parquet cache / 1h->4h resample).
      3. Updates active paper trades (`OPEN` / `TP1_HIT`) and dispatches Telegram lifecycle alerts.
      4. Runs the 6-node LangGraphTradingOrchestrator on each (symbol, horizon) pair,
         opens qualifying setups, and writes a complete UI state snapshot for the SaaS Web Platform.
    """

    HORIZON_TIMEFRAMES = {
        "scalp": {"entry_tf": "5m", "htf": "1h"},
        "intraday": {"entry_tf": "15m", "htf": "4h"},
        "swing": {"entry_tf": "1h", "htf": "4h"},
    }

    DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    def __init__(
        self,
        config_path: str = "config/assets_config.yaml",
        account_balance_per_horizon_usdt: float = 1000.0,
    ):
        self.config_loader = ConfigLoader(config_path)
        self.fetcher = _resolve_fetcher_instance(config_path)
        self.orchestrator = LangGraphTradingOrchestrator(config_path)
        self.lifecycle = PaperTradeLifecycleManager(
            config_path=config_path,
            default_horizon_capital_usdt=account_balance_per_horizon_usdt,
        )
        self.notifier = TelegramAlertNotifier()
        self.account_balance_usdt = float(account_balance_per_horizon_usdt)
        self.state_cache_path = PROJECT_ROOT / "data" / "cache" / "latest_scanner_state.json"
        self.state_cache_path.parent.mkdir(parents=True, exist_ok=True)

    def _resolve_configured_symbols(self) -> List[str]:
        """Resolves target symbols across any ConfigLoader implementation or local cache."""
        # 1. Check helper methods on ConfigLoader
        for method_name in ("get_symbols", "get_enabled_symbols", "get_active_symbols", "list_symbols"):
            if hasattr(self.config_loader, method_name):
                res = getattr(self.config_loader, method_name)()
                if isinstance(res, list) and res:
                    return [str(s).replace("/", "").upper() for s in res]

        # 2. Check underlying dictionary attributes on ConfigLoader
        raw_cfg = getattr(self.config_loader, "config", None) or getattr(self.config_loader, "data", None)
        if isinstance(raw_cfg, dict):
            assets = raw_cfg.get("assets") or raw_cfg.get("symbols") or raw_cfg.get("pairs")
            if isinstance(assets, list) and assets:
                extracted = []
                for item in assets:
                    if isinstance(item, str):
                        extracted.append(item.replace("/", "").upper())
                    elif isinstance(item, dict) and item.get("enabled", True):
                        sym = item.get("symbol") or item.get("name")
                        if sym:
                            extracted.append(str(sym).replace("/", "").upper())
                if extracted:
                    return extracted
            elif isinstance(assets, dict) and assets:
                return [
                    str(k).replace("/", "").upper()
                    for k, v in assets.items()
                    if not isinstance(v, dict) or v.get("enabled", True)
                ]

        # 3. Discover available symbols from `data/cache/*_5m.parquet`
        cache_dir = PROJECT_ROOT / "data" / "cache"
        cached_syms = sorted(
            {p.name.split("_")[0].upper() for p in cache_dir.glob("*_5m.parquet")}
        )
        if cached_syms:
            return cached_syms

        return list(self.DEFAULT_SYMBOLS)

    @staticmethod
    def _resample_1h_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
        """Synthesizes 4h OHLCV candles from 1h candles if 4h cache is not present on disk."""
        df = df_1h.copy()
        ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
        df = df.set_index(ts_col).sort_index()
        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        df_4h = df.resample("4h").agg(agg_dict).dropna().reset_index()
        return df_4h

    async def _load_or_fetch_candles(
        self, symbol: str, timeframe: str, limit: int = 260, fetch_live: bool = True
    ) -> pd.DataFrame:
        """Fetches fresh closed candles from Binance or loads local Parquet cache cleanly."""
        cache_file = PROJECT_ROOT / "data" / "cache" / f"{symbol}_{timeframe}.parquet"
        if fetch_live and self.fetcher is not None:
            try:
                for fn_name in ("fetch_closed_ohlcv", "fetch_ohlcv"):
                    if hasattr(self.fetcher, fn_name):
                        fn = getattr(self.fetcher, fn_name)
                        res = fn(symbol, timeframe, limit=limit)
                        df = await res if inspect.iscoroutine(res) else res
                        if df is not None and len(df) >= 120:
                            df.to_parquet(cache_file, index=False)
                            return df
            except Exception:
                pass

        if cache_file.exists():
            df_cached = pd.read_parquet(cache_file)
            if len(df_cached) >= 50:
                return df_cached.tail(limit).reset_index(drop=True)

        # Fallback for 4h timeframe: resample from 1h cached Parquet
        if timeframe == "4h":
            cache_1h = PROJECT_ROOT / "data" / "cache" / f"{symbol}_1h.parquet"
            if cache_1h.exists():
                df_1h = pd.read_parquet(cache_1h)
                df_4h = self._resample_1h_to_4h(df_1h)
                if len(df_4h) >= 40:
                    df_4h.to_parquet(cache_file, index=False)
                    return df_4h.tail(limit).reset_index(drop=True)

        raise RuntimeError(f"Insufficient candle data for {symbol} ({timeframe}).")

    async def _close_fetcher(self) -> None:
        if self.fetcher is not None:
            for close_name in ("close", "close_exchange"):
                if hasattr(self.fetcher, close_name):
                    fn = getattr(self.fetcher, close_name)
                    res = fn()
                    if inspect.iscoroutine(res):
                        await res
                    break

    async def scan_single_pair_horizon(
        self,
        symbol: str,
        horizon: str,
        fetch_live: bool = True,
        use_llm_debate: bool = False,
        send_telegram: bool = True,
    ) -> Dict[str, Any]:
        """Executes a complete lifecycle + LangGraph evaluation pass for one (symbol, horizon)."""
        tf_spec = self.HORIZON_TIMEFRAMES[horizon]
        entry_tf = tf_spec["entry_tf"]
        htf = tf_spec["htf"]

        df_entry = await self._load_or_fetch_candles(symbol, entry_tf, limit=260, fetch_live=fetch_live)
        df_htf = await self._load_or_fetch_candles(symbol, htf, limit=220, fetch_live=fetch_live)

        feat_entry = CryptoFeatureEngineer.compute_features(df_entry, drop_na=True)
        feat_htf = CryptoFeatureEngineer.compute_features(df_htf, drop_na=True)

        trainer = self.orchestrator.engine.ml_agent.trainer
        if not trainer.get_model_path(symbol, horizon).exists():
            trainer.train_horizon_model(symbol, horizon, feat_entry)

        latest_bar = feat_entry.iloc[-1]
        c_high = float(latest_bar["high"])
        c_low = float(latest_bar["low"])
        c_close = float(latest_bar["close"])

        # 1. Update any existing active trades (OPEN / TP1_HIT) on this (symbol, horizon)
        lifecycle_events = self.lifecycle.update_open_trades(
            horizon=horizon,
            symbol=symbol,
            candle_high=c_high,
            candle_low=c_low,
            candle_close=c_close,
        )

        if send_telegram and lifecycle_events:
            h_summary = self.lifecycle.get_horizon_performance_summary(
                horizon=horizon, initial_capital_usdt=self.account_balance_usdt
            )
            for ev in lifecycle_events:
                if ev["event_type"] != "MAE_MFE_UPDATED":
                    msg = self.notifier.format_lifecycle_event_alert(ev, h_summary)
                    self.notifier.send_message(msg)

        # 2. Fetch live orderbook/funding microstructure (if live mode enabled)
        microstructure: Optional[Dict[str, float]] = None
        if fetch_live:
            try:
                microstructure = await self.orchestrator.engine.sentiment_agent.fetch_live_microstructure(symbol)
            except Exception:
                microstructure = None

        # 3. Run 6-Node LangGraph Orchestrator
        h_equity = self.lifecycle.get_horizon_performance_summary(
            horizon=horizon, initial_capital_usdt=self.account_balance_usdt
        )["current_equity_usdt"]

        decision = self.orchestrator.run_graph(
            symbol=symbol,
            horizon=horizon,
            df=feat_entry,
            htf_df=feat_htf,
            microstructure=microstructure,
            use_llm_debate=use_llm_debate,
            account_balance_usdt=h_equity,
            auto_log_to_db=False,
        )

        # Attach last 60 OHLCV candles for the SaaS Web UI Candlestick Chart
        chart_slice = feat_entry.tail(60)
        chart_candles: List[Dict[str, Any]] = []
        for idx, (_, row) in enumerate(chart_slice.iterrows()):
            ts_val = row.get("timestamp") or row.get("open_time")
            try:
                if hasattr(ts_val, "timestamp"):
                    unix_sec = int(ts_val.timestamp())
                else:
                    unix_sec = int(pd.Timestamp(ts_val).timestamp())
            except Exception:
                unix_sec = int(datetime.now(timezone.utc).timestamp()) - ((60 - idx) * 300)

            chart_candles.append(
                {
                    "time": unix_sec,
                    "open": round(float(row["open"]), 4),
                    "high": round(float(row["high"]), 4),
                    "low": round(float(row["low"]), 4),
                    "close": round(float(row["close"]), 4),
                    "volume": round(float(row.get("volume", 0.0)), 2),
                }
            )

        decision["timeframe"] = entry_tf
        decision["htf_timeframe"] = htf
        decision["latest_price"] = round(c_close, 4)
        decision["chart_candles"] = chart_candles
        decision["lifecycle_events"] = lifecycle_events

        # 4. If signal qualifies for execution, open paper trade & dispatch Telegram alert
        open_result = {"opened": False, "reason": "NO_EXECUTION"}
        if decision.get("should_execute"):
            open_result = self.lifecycle.open_paper_trade(decision)
            if open_result.get("opened") and send_telegram:
                alert_html = self.notifier.format_new_trade_alert(decision, open_result)
                self.notifier.send_message(alert_html)

        decision["execution_status"] = open_result
        return decision

    async def run_full_scan_cycle(
        self,
        symbols: Optional[List[str]] = None,
        horizons: Optional[List[str]] = None,
        fetch_live: bool = True,
        use_llm_debate: bool = False,
        send_telegram: bool = True,
    ) -> Dict[str, Any]:
        """
        Scans all configured symbols across all 3 horizons (`scalp`, `intraday`, `swing`),
        persists the full state snapshot to `data/cache/latest_scanner_state.json`,
        and returns the complete SaaS platform payload.
        """
        target_symbols = symbols or self._resolve_configured_symbols()
        target_horizons = horizons or ["scalp", "intraday", "swing"]

        signals_by_horizon: Dict[str, Dict[str, Any]] = {h: {} for h in target_horizons}
        flat_signals: List[Dict[str, Any]] = []

        try:
            for sym in target_symbols:
                for hor in target_horizons:
                    res = await self.scan_single_pair_horizon(
                        symbol=sym,
                        horizon=hor,
                        fetch_live=fetch_live,
                        use_llm_debate=use_llm_debate,
                        send_telegram=send_telegram,
                    )
                    signals_by_horizon[hor][sym] = res
                    flat_signals.append(res)
        finally:
            await self._close_fetcher()
            await self.orchestrator.engine.sentiment_agent.close_exchange()

        dashboard_perf = self.lifecycle.get_multi_horizon_dashboard_payload(
            initial_capital_per_horizon_usdt=self.account_balance_usdt
        )
        active_by_horizon = {
            h: self.lifecycle.get_active_trades(h) for h in ("scalp", "intraday", "swing")
        }

        full_state = {
            "scan_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "symbols_scanned": target_symbols,
            "horizons_scanned": target_horizons,
            "isolated_performance": dashboard_perf,
            "active_trades_by_horizon": active_by_horizon,
            "signals_by_horizon": signals_by_horizon,
            "all_signals": flat_signals,
        }

        with open(self.state_cache_path, "w", encoding="utf-8") as f:
            json.dump(full_state, f, indent=2, default=str)

        return full_state


async def _demo_scanner():
    scanner = LiveMultiHorizonScanner(account_balance_per_horizon_usdt=1000.0)
    print("Running Multi-Horizon Scan across BTCUSDT, ETHUSDT, SOLUSDT (Scalp, Intraday, Swing)...")
    state = await scanner.run_full_scan_cycle(
        fetch_live=False,
        use_llm_debate=False,
        send_telegram=True,
    )
    print(f"\nScan Complete! State cached at: {scanner.state_cache_path}")
    print("-" * 82)
    for sig in state["all_signals"]:
        print(
            f" [{sig['horizon'].upper():8s} | {sig['timeframe']:3s}] {sig['symbol']:8s} -> "
            f"{sig['direction']:11s} | Score: {sig['total_confluence_score']:5.2f}/100 | "
            f"Price: ${sig['latest_price']:,.2f} | Tier: {sig['signal_tier']}"
        )
    print("-" * 82)


if __name__ == "__main__":
    asyncio.run(_demo_scanner())