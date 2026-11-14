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
    2. In-Memory Rolling Window Store (instant 0ms DataFrame access for agents).
    3. Boundary-Aligned Real-Time Auto-Streamer (triggers immediately on 1m, 5m, 15m, 1h, 4h, 1d candle close).
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

        # Step A: Check in-memory store first, otherwise load from Parquet disk cache
        if symbol in self.memory_store and timeframe in self.memory_store[symbol]:
            cached_df = self.memory_store[symbol][timeframe]
        elif cache_file.exists():
            try:
                cached_df = pd.read_parquet(cache_file)
            except Exception:
                cached_df = pd.DataFrame()

        # Step B: Determine 'since' timestamp to fetch ONLY new missing candles
        if not cached_df.empty:
            last_ts = cached_df["timestamp"].iloc[-1]
            # Fetch starting from the last known closed candle timestamp
            since_ms = int(last_ts.timestamp() * 1000)

        # Step C: Fetch either delta candles (since_ms) or initial_limit candles
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

            # Drop the currently forming (unclosed) candle by comparing with current UTC time
            now_utc = pd.Timestamp.now(tz="UTC")
            tf_sec = self.TIMEFRAME_SECONDS.get(timeframe, 60)
            # A candle is closed only if (candle_open_time + timeframe_duration) <= current_time
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
                combined_df = (
                    combined_df.drop_duplicates(subset=["timestamp"], keep="last")
                    .sort_values("timestamp")
                    .tail(self.max_memory_candles)
                    .reset_index(drop=True)
                )
                # Update both In-Memory store and Parquet file
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
    # 2. AUTOMATED REAL-TIME CANDLE-CLOSE STREAMER (ZERO MANUAL WORK)
    # -----------------------------------------------------------------
    @classmethod
    def get_closed_timeframes_at_timestamp(
        cls, unix_ts: int, watched_timeframes: List[str]
    ) -> List[str]:
        """
        Determines which timeframes just closed at the exact minute boundary.
        Example: At 12:15:00 -> ['1m', '5m', '15m'] all closed simultaneously!
        """
        closed_tfs = []
        # Round down to the nearest minute boundary
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
        """
        Runs continuously in the background. Wakes up 150ms after every minute boundary,
        identifies which timeframes (1m, 5m, 15m, 1h, 4h, 1d) just closed, delta-syncs
        only those candles, and immediately triggers the agent callback for live execution.
        """
        if symbols is None:
            symbols = list(self.config_loader.get_active_assets().keys())

        # Step 1: Initial Warm-Up for all active symbols
        for sym in symbols:
            await self.warm_up_all_timeframes(sym)

        watched_tfs = list(self.memory_store[symbols[0]].keys()) if symbols else ["1m", "5m", "15m", "1h", "4h", "1d"]
        self._streaming_active = True
        cycle_count = 0

        while self._streaming_active:
            now = datetime.now(timezone.utc).timestamp()
            # Calculate exact sleep time until the next minute boundary + 0.15s buffer
            next_minute_boundary = ((int(now) // 60) + 1) * 60
            sleep_duration = max(0.05, (next_minute_boundary + 0.15) - now)
            await asyncio.sleep(sleep_duration)

            trigger_ts = int(datetime.now(timezone.utc).timestamp())
            closed_tfs = self.get_closed_timeframes_at_timestamp(trigger_ts, watched_tfs)

            if closed_tfs:
                # Fetch all newly closed timeframes across all symbols in parallel
                sync_tasks = []
                task_meta = []
                for sym in symbols:
                    for tf in closed_tfs:
                        sync_tasks.append(self.sync_timeframe_with_cache(sym, tf))
                        task_meta.append((sym, tf))

                updated_dfs = await asyncio.gather(*sync_tasks)

                # Trigger immediate agent analysis callback for each newly closed candle
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


async def _sample_agent_callback(symbol: str, timeframe: str, df: pd.DataFrame) -> None:
    """Example callback triggered automatically within milliseconds of candle close."""
    latest = df.iloc[-1]
    print(
        f"[AUTO-TRIGGER] {symbol} [{timeframe:>3}] Candle Closed! "
        f"Time: {latest['timestamp']} | Close: {latest['close']} | Vol: {latest['volume']}"
    )


async def _demo_run():
    fetcher = MultiTimeframeDataFetcher()
    try:
        print("1. Running Initial Warm-up & Parquet Cache Sync for BTCUSDT...")
        data_map = await fetcher.warm_up_all_timeframes("BTCUSDT", initial_limit=300)
        for tf, df in data_map.items():
            cache_path = fetcher._get_cache_file("BTCUSDT", tf)
            print(
                f" -> [{tf:>3}] Cached {len(df)} candles | Latest: {df['close'].iloc[-1]} "
                f"| Parquet Saved: {cache_path.name}"
            )

        print("\n2. Testing Boundary-Aligned Auto-Trigger (Waiting for next 1m candle close)...")
        await fetcher.start_auto_stream(
            symbols=["BTCUSDT"],
            on_candle_close=_sample_agent_callback,
            max_cycles=1,  # Runs 1 automatic minute boundary cycle for demo verification
        )
    finally:
        await fetcher.close_exchange()


if __name__ == "__main__":
    asyncio.run(_demo_run())