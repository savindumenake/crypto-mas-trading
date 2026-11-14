import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional
import ccxt.async_support as ccxt
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import ConfigLoader

# Type alias for async callback triggered immediately when a candle closes
CandleCloseCallback = Callable[[str, str, pd.DataFrame], Coroutine[Any, Any, None]]


class MultiTimeframeDataFetcher:
    """
    Automated Multi-Timeframe Data Engine with:
    1. Local Parquet Delta-Caching (fetches only missing candles after initial load).
    2. Bulk Historical Pagination (downloads 10k-65k+ candles for deep ML training).
    3. In-Memory Rolling Window Store (instant 0ms DataFrame access for agents).
    4. Boundary-Aligned Real-Time Auto-Streamer (triggers immediately on candle close).
    """

    TIMEFRAME_SECONDS: Dict[str, int] = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "1h": 3600,
        "4h": 14400,
        "1d": 86400,
        "1w": 604800,
    }

    def __init__(self, config_path: str = "config/assets_config.yaml", max_memory_candles: int = 1000):
        self.config_loader = ConfigLoader(config_path)
        sys_cfg = self.config_loader.full_config.get("system", {})
        self.exchange_id = sys_cfg.get("default_exchange", "binance")
        self.exchange: Optional[ccxt.Exchange] = None

        self.cache_dir = PROJECT_ROOT / "data" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_memory_candles = max_memory_candles

        # In-Memory live rolling store: self.memory_store["BTCUSDT"]["5m"] -> pd.DataFrame
        self.memory_store: Dict[str, Dict[str, pd.DataFrame]] = {}
        self._streaming_active = False

    async def initialize_exchange(self) -> None:
        """Initializes the async CCXT exchange instance with rate-limiting enabled."""
        if self.exchange is None:
            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange = exchange_class(
                {
                    "enableRateLimit": True,
                    "options": {
                        "defaultType": "future",  # USDT-M Perpetual Futures
                    },
                }
            )

    async def close_exchange(self) -> None:
        """Gracefully stops streaming and closes the async CCXT session."""
        self._streaming_active = False
        if self.exchange is not None:
            await self.exchange.close()
            self.exchange = None

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        if "/" in symbol:
            return symbol
        if symbol.endswith("USDT"):
            return f"{symbol[:-4]}/USDT"
        return symbol

    def _get_cache_file(self, symbol: str, timeframe: str) -> Path:
        clean_sym = symbol.replace("/", "").replace(":", "")
        return self.cache_dir / f"{clean_sym}_{timeframe}.parquet"

    # -----------------------------------------------------------------
    # 1. PARQUET CACHE + DELTA SYNC (NO REDUNDANT DOWNLOADS)
    # -----------------------------------------------------------------
    async def sync_timeframe_with_cache(
        self,
        symbol: str,
        timeframe: str,
        initial_limit: int = 500,
        max_retries: int = 3,
    ) -> pd.DataFrame:
        """
        Loads historical data from local Parquet cache if available, then fetches
        ONLY the newly closed candles from Binance since the last cached timestamp.
        """
        await self.initialize_exchange()
        ccxt_symbol = self._normalize_symbol(symbol)
        cache_file = self._get_cache_file(symbol, timeframe)

        cached_df = pd.DataFrame()
        since_ms: Optional[int] = None

        if symbol in self.memory_store and timeframe in self.memory_store[symbol]:
            cached_df = self.memory_store[symbol][timeframe]
        elif cache_file.exists():
            try:
                cached_df = pd.read_parquet(cache_file)
            except Exception:
                cached_df = pd.DataFrame()

        if not cached_df.empty:
            last_ts = cached_df["timestamp"].iloc[-1]
            since_ms = int(last_ts.timestamp() * 1000)

        for attempt in range(1, max_retries + 1):
            try:
                if since_ms is not None:
                    raw_ohlcv = await self.exchange.fetch_ohlcv(
                        ccxt_symbol, timeframe=timeframe, since=since_ms, limit=500
                    )
                else:
                    raw_ohlcv = await self.exchange.fetch_ohlcv(
                        ccxt_symbol, timeframe=timeframe, limit=initial_limit + 1
                    )
                break
            except Exception as exc:
                if attempt == max_retries:
                    raise RuntimeError(f"Delta sync failed for {symbol} ({timeframe}): {exc}") from exc
                await asyncio.sleep(1.0 * attempt)

        if raw_ohlcv:
            new_df = pd.DataFrame(
                raw_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], unit="ms", utc=True)
            for col in ["open", "high", "low", "close", "volume"]:
                new_df[col] = new_df[col].astype("float64")

            now_utc = pd.Timestamp.now(tz="UTC")
            tf_sec = self.TIMEFRAME_SECONDS.get(timeframe, 60)
            new_df = new_df[
                (new_df["timestamp"] + pd.Timedelta(seconds=tf_sec)) <= now_utc
            ]

            if not cached_df.empty and not new_df.empty:
                combined_df = pd.concat([cached_df, new_df], ignore_index=True)
            elif not new_df.empty:
                combined_df = new_df
            else:
                combined_df = cached_df

            if not combined_df.empty:
                keep_limit = max(self.max_memory_candles, len(cached_df))
                combined_df = (
                    combined_df.drop_duplicates(subset=["timestamp"], keep="last")
                    .sort_values("timestamp")
                    .tail(keep_limit)
                    .reset_index(drop=True)
                )
                self.memory_store.setdefault(symbol, {})[timeframe] = combined_df
                combined_df.to_parquet(cache_file, index=False)
                return combined_df

        return cached_df

    async def warm_up_all_timeframes(
        self, symbol: str, initial_limit: int = 500
    ) -> Dict[str, pd.DataFrame]:
        """Concurrently warms up cache and memory for all configured timeframes."""
        horizons_cfg = self.config_loader.full_config.get("horizons", {})
        unique_tfs: List[str] = []
        for h_cfg in horizons_cfg.values():
            for tf in h_cfg.get("execution_timeframes", []) + [h_cfg.get("trend_filter_timeframe")]:
                if tf and tf not in unique_tfs:
                    unique_tfs.append(tf)

        tasks = [
            self.sync_timeframe_with_cache(symbol, tf, initial_limit=initial_limit)
            for tf in unique_tfs
        ]
        results = await asyncio.gather(*tasks)
        return dict(zip(unique_tfs, results))

    # -----------------------------------------------------------------
    # 1B. BULK HISTORICAL PAGINATION (FOR DEEP ML TRAINING: 10K - 65K CANDLES)
    # -----------------------------------------------------------------
    async def fetch_historical_bulk(
        self,
        symbol: str,
        timeframe: str,
        total_candles: int = 15000,
        batch_size: int = 1000,
    ) -> pd.DataFrame:
        """
        Paginates forward in batches of 1000 candles to download large historical
        datasets (e.g., 25,000 5m candles, 18,000 1h candles, 10,000 4h candles)
        and saves them into the local Parquet cache.
        """
        await self.initialize_exchange()
        ccxt_symbol = self._normalize_symbol(symbol)
        cache_file = self._get_cache_file(symbol, timeframe)
        tf_sec = self.TIMEFRAME_SECONDS.get(timeframe, 60)

        if cache_file.exists():
            try:
                existing_df = pd.read_parquet(cache_file)
                if len(existing_df) >= int(total_candles * 0.95):
                    return await self.sync_timeframe_with_cache(symbol, timeframe)
            except Exception:
                pass

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ms = now_ms - (total_candles * tf_sec * 1000)
        current_since = start_ms
        all_batches: List[pd.DataFrame] = []

        print(f"   [Bulk Download] Fetching ~{total_candles} historical '{timeframe}' candles for {symbol}...")

        while current_since < now_ms:
            raw_ohlcv = await self.exchange.fetch_ohlcv(
                ccxt_symbol, timeframe=timeframe, since=current_since, limit=batch_size
            )
            if not raw_ohlcv:
                break

            batch_df = pd.DataFrame(
                raw_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            all_batches.append(batch_df)

            last_fetched_ms = int(raw_ohlcv[-1][0])
            next_since = last_fetched_ms + (tf_sec * 1000)
            if next_since <= current_since or len(raw_ohlcv) < (batch_size // 2):
                break
            current_since = next_since

        if not all_batches:
            return pd.DataFrame()

        full_df = pd.concat(all_batches, ignore_index=True)
        full_df["timestamp"] = pd.to_datetime(full_df["timestamp"], unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            full_df[col] = full_df[col].astype("float64")

        now_utc = pd.Timestamp.now(tz="UTC")
        full_df = full_df[(full_df["timestamp"] + pd.Timedelta(seconds=tf_sec)) <= now_utc]

        full_df = (
            full_df.drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp")
            .tail(total_candles)
            .reset_index(drop=True)
        )

        self.memory_store.setdefault(symbol, {})[timeframe] = full_df
        full_df.to_parquet(cache_file, index=False)
        return full_df

    # -----------------------------------------------------------------
    # 2. AUTOMATED REAL-TIME CANDLE-CLOSE STREAMER (ZERO MANUAL WORK)
    # -----------------------------------------------------------------
    @classmethod
    def get_closed_timeframes_at_timestamp(
        cls, unix_ts: int, watched_timeframes: List[str]
    ) -> List[str]:
        closed_tfs = []
        minute_boundary = (unix_ts // 60) * 60
        for tf in watched_timeframes:
            tf_sec = cls.TIMEFRAME_SECONDS.get(tf)
            if tf_sec and minute_boundary % tf_sec == 0:
                closed_tfs.append(tf)
        return closed_tfs

    async def start_auto_stream(
        self,
        symbols: Optional[List[str]] = None,
        on_candle_close: Optional[CandleCloseCallback] = None,
        max_cycles: Optional[int] = None,
    ) -> None:
        if symbols is None:
            symbols = list(self.config_loader.get_active_assets().keys())

        for sym in symbols:
            await self.warm_up_all_timeframes(sym)

        watched_tfs = list(self.memory_store[symbols[0]].keys()) if symbols else ["1m", "5m", "15m", "1h", "4h", "1d"]
        self._streaming_active = True
        cycle_count = 0

        while self._streaming_active:
            now = datetime.now(timezone.utc).timestamp()
            next_minute_boundary = ((int(now) // 60) + 1) * 60
            sleep_duration = max(0.05, (next_minute_boundary + 0.15) - now)
            await asyncio.sleep(sleep_duration)

            trigger_ts = int(datetime.now(timezone.utc).timestamp())
            closed_tfs = self.get_closed_timeframes_at_timestamp(trigger_ts, watched_tfs)

            if closed_tfs:
                sync_tasks = []
                task_meta = []
                for sym in symbols:
                    for tf in closed_tfs:
                        sync_tasks.append(self.sync_timeframe_with_cache(sym, tf))
                        task_meta.append((sym, tf))

                updated_dfs = await asyncio.gather(*sync_tasks)

                if on_candle_close is not None:
                    cb_tasks = [
                        on_candle_close(sym, tf, df)
                        for (sym, tf), df in zip(task_meta, updated_dfs)
                        if not df.empty
                    ]
                    if cb_tasks:
                        await asyncio.gather(*cb_tasks)

            cycle_count += 1
            if max_cycles is not None and cycle_count >= max_cycles:
                break