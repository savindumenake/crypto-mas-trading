import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import requests
from dotenv import load_dotenv

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env", override=True)


class TelegramAlertNotifier:
    """
    Dispatches formatted institutional trade alerts and lifecycle updates to Telegram.
    Routes through Cloudflare Worker Relay by default to bypass ISP TLS/SNI blocks.
    """

    HORIZON_BADGES = {
        "scalp": "⚡ SCALP (5m)",
        "intraday": "📈 INTRADAY (15m)",
        "swing": "🌊 SWING (1h/4h)",
    }

    DEFAULT_RELAY_URL = "https://cryptomastrading.savidumenake2002.workers.dev"

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        timeout_sec: int = 15,
    ):
        load_dotenv(PROJECT_ROOT / ".env", override=True)
        self.bot_token = (bot_token or os.getenv("TELEGRAM_BOT_TOKEN") or "").strip().strip('"').strip("'")
        self.chat_id = (chat_id or os.getenv("TELEGRAM_CHAT_ID") or "").strip().strip('"').strip("'")
        self.base_url = (
            os.getenv("TELEGRAM_API_BASE_URL") or self.DEFAULT_RELAY_URL
        ).strip().rstrip("/")
        self.timeout_sec = timeout_sec

    @property
    def is_configured(self) -> bool:
        return bool(
            self.bot_token
            and ":" in self.bot_token
            and self.chat_id
            and "your_" not in self.bot_token.lower()
            and "your_" not in self.chat_id.lower()
        )

    def send_message(self, text: str, disable_web_preview: bool = True) -> Dict[str, Any]:
        """Sends an HTML-formatted message via Cloudflare Relay / Telegram Bot API."""
        if not self.is_configured:
            return {
                "sent": False,
                "mode": "DRY_RUN_CONSOLE",
                "reason": "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not configured in .env",
                "message_preview": text,
            }

        endpoints = [self.base_url]
        if "api.telegram.org" not in self.base_url:
            endpoints.append("https://api.telegram.org")

        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": disable_web_preview,
        }
        payload_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        errors = []
        for base in endpoints:
            url = f"{base}/bot{self.bot_token}/sendMessage"
            # 1. Try Python requests
            try:
                resp = requests.post(
                    url,
                    data=payload_bytes,
                    headers={
                        "Content-Type": "application/json; charset=utf-8",
                        "User-Agent": "NeuralPulseQuantBot/1.0",
                    },
                    timeout=self.timeout_sec,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("ok"):
                        return {"sent": True, "mode": f"TELEGRAM_LIVE ({base})", "response": data}
                errors.append(f"HTTP {resp.status_code} from {base}: {resp.text[:160]}")
            except Exception as exc:
                errors.append(f"Requests ({base}): {exc}")

            # 2. Try curl -4 fallback on this endpoint
            try:
                cmd = [
                    "curl",
                    "-4",
                    "-sS",
                    "--ssl-no-revoke",
                    "--max-time",
                    str(self.timeout_sec),
                    "-X",
                    "POST",
                    url,
                    "-H",
                    "Content-Type: application/json; charset=utf-8",
                    "--data-binary",
                    "@-",
                ]
                proc = subprocess.run(cmd, input=payload_bytes, capture_output=True, check=False)
                out_str = proc.stdout.decode("utf-8", errors="replace").strip()
                err_str = proc.stderr.decode("utf-8", errors="replace").strip()
                if proc.returncode == 0 and out_str:
                    data = json.loads(out_str)
                    if data.get("ok"):
                        return {"sent": True, "mode": f"TELEGRAM_CURL ({base})", "response": data}
                    errors.append(f"Telegram API ({base}): {data.get('description') or out_str}")
                else:
                    errors.append(f"Curl ({base}): {err_str or out_str}")
            except Exception as c_exc:
                errors.append(f"CurlExc ({base}): {c_exc}")

        return {
            "sent": False,
            "mode": "TELEGRAM_DELIVERY_FAILED",
            "reason": " | ".join(errors),
            "message_preview": text,
        }

    def format_new_trade_alert(
        self, decision: Dict[str, Any], open_result: Dict[str, Any]
    ) -> str:
        horizon = str(decision.get("horizon", "scalp")).lower()
        badge = self.HORIZON_BADGES.get(horizon, horizon.upper())
        symbol = decision.get("symbol", "UNKNOWN")
        direction = decision.get("evaluated_bias") or decision.get("direction", "LONG")
        dir_emoji = "🟢 LONG" if direction == "LONG" else "🔴 SHORT"
        tier = decision.get("signal_tier", "STANDARD")
        score = decision.get("total_confluence_score", 0.0)
        comp = decision.get("component_scores", {})
        rp = decision.get("risk_plan", {})

        reports = decision.get("agent_reports", {})
        pat_rep = reports.get("pattern", {})
        ml_rep = reports.get("ml_shap", {})
        deb_rep = reports.get("debate", {})
        llm_rev = deb_rep.get("llm_executive_review", {})

        shap_top = ml_rep.get("shap_top_features_json", {})
        if isinstance(shap_top, dict) and shap_top:
            shap_str = ", ".join(f"{k} ({v:+.3f})" for k, v in list(shap_top.items())[:3])
        else:
            shap_str = "Balanced Technical Drivers"

        synthesis = llm_rev.get("judge_synthesis", "Quantitative confluence verified.")
        trade_id = open_result.get("trade_id", "N/A")

        lines = [
            f"<b>{badge} | NEW TRADE OPENED (#{trade_id})</b>",
            f"<b>Asset:</b> {symbol} | <b>Side:</b> {dir_emoji}",
            f"<b>Confluence:</b> {score:.2f} / 100.0 (<code>{tier}</code>)",
            f"<b>Breakdown:</b> ML {comp.get('ml_shap_score_45', 0):.1f}/45 | Pat {comp.get('pattern_structure_score_25', 0):.1f}/25 | Sent {comp.get('sentiment_microstructure_score_15', 0):.1f}/15 | Deb {comp.get('debate_memory_score_15', 0):.1f}/15",
            "────────────────────────",
            f"<b>Entry Price:</b> ${rp.get('entry_price', 0):,.2f} (ATR: ${rp.get('atr_value', 0):,.2f})",
            f"<b>Stop-Loss:</b> ${rp.get('stop_loss', 0):,.2f} ({rp.get('sl_atr_mult', 1.5)}x ATR)",
            f"<b>TP1 (50% + BE):</b> ${rp.get('tp1_price', 0):,.2f} (R:R 1:{rp.get('rr_ratio_tp1', 1.2)})",
            f"<b>TP2 (Runner):</b> ${rp.get('tp2_price', 0):,.2f} (R:R 1:{rp.get('rr_ratio_tp2', 2.0)})",
            f"<b>Position Size:</b> {rp.get('position_size_coins', 0)} (Risk: ${rp.get('risk_amount_usdt', 0):.2f})",
            "────────────────────────",
            f"<b>Pattern:</b> {pat_rep.get('pattern_type', 'N/A')} ({pat_rep.get('market_regime', 'N/A')})",
            f"<b>Top SHAP Drivers:</b> {shap_str}",
            f"<b>Judge Note:</b> <i>{synthesis}</i>",
        ]
        return "\n".join(lines)

    def format_lifecycle_event_alert(
        self, event: Dict[str, Any], horizon_summary: Optional[Dict[str, Any]] = None
    ) -> str:
        horizon = str(event.get("horizon", "scalp")).lower()
        badge = self.HORIZON_BADGES.get(horizon, horizon.upper())
        ev_type = event.get("event_type", "UPDATE")
        trade_id = event.get("trade_id", "N/A")
        symbol = event.get("symbol", "UNKNOWN")
        direction = event.get("direction", "LONG")

        if ev_type == "TP1_PARTIAL_AND_BREAKEVEN_SET":
            header = f"🎯 <b>{badge} | TP1 HIT & BREAKEVEN LOCKED (#{trade_id})</b>"
            outcome_line = f"50% Partial Profit Locked! Stop-Loss moved to Entry (<b>${event.get('stop_loss', 0):,.2f}</b>) — Risk-Free Runner Active."
        elif ev_type == "TP2_FULL_TARGET_HIT":
            header = f"🚀 <b>{badge} | TP2 FULL TARGET HIT (#{trade_id})</b>"
            outcome_line = f"Closed at <b>${event.get('exit_price', 0):,.2f}</b> | Realized PnL: <b>{event.get('realized_pnl_r', 0):+.2f}R (${event.get('realized_pnl_usdt', 0):+,.2f})</b>"
        elif ev_type == "BREAKEVEN_STOP_AFTER_TP1":
            header = f"🛡️ <b>{badge} | RUNNER STOPPED AT BREAKEVEN (#{trade_id})</b>"
            outcome_line = f"Runner exited at Entry (<b>${event.get('exit_price', 0):,.2f}</b>) | Net Locked Profit: <b>{event.get('realized_pnl_r', 0):+.2f}R (${event.get('realized_pnl_usdt', 0):+,.2f})</b>"
        else:
            header = f"🛑 <b>{badge} | STOP-LOSS HIT (#{trade_id})</b>"
            outcome_line = f"Closed at <b>${event.get('exit_price', 0):,.2f}</b> | Realized PnL: <b>{event.get('realized_pnl_r', 0):+.2f}R (${event.get('realized_pnl_usdt', 0):+,.2f})</b>"

        lines = [
            header,
            f"<b>Asset:</b> {symbol} ({direction}) | <b>Status:</b> {event.get('previous_status')} → {event.get('new_status')}",
            outcome_line,
            f"<b>Excursion Stats:</b> MAE {event.get('mae_atr', 0):.2f} ATR | MFE {event.get('mfe_atr', 0):.2f} ATR",
        ]

        if horizon_summary:
            lines.append("────────────────────────")
            lines.append(
                f"<b>{horizon.upper()} Book Equity:</b> ${horizon_summary.get('current_equity_usdt', 1000):,.2f} "
                f"(Growth: <b>{horizon_summary.get('growth_rate_pct', 0):+.2f}%</b> | WinRate: {horizon_summary.get('win_rate_pct', 0)}%)"
            )
        return "\n".join(lines)

    def format_multi_horizon_digest(self, dashboard_payload: Dict[str, Any]) -> str:
        lines = ["📊 <b>MULTI-HORIZON ISOLATED PERFORMANCE DIGEST</b>", "────────────────────────"]
        horizons = dashboard_payload.get("horizons", {})
        for h_key in ("scalp", "intraday", "swing"):
            st = horizons.get(h_key, {})
            badge = self.HORIZON_BADGES.get(h_key, h_key.upper())
            lines.append(
                f"<b>{badge}</b>\n"
                f"  • Equity: <b>${st.get('current_equity_usdt', 1000):,.2f}</b> ({st.get('growth_rate_pct', 0):+.2f}%)\n"
                f"  • Net PnL: <b>${st.get('net_pnl_usdt', 0):+,.2f}</b> ({st.get('net_realized_r', 0):+.2f}R)\n"
                f"  • Win Rate: {st.get('win_rate_pct', 0)}% ({st.get('closed_trades_count', 0)} closed, {st.get('active_trades_count', 0)} active)"
            )
        rollup = dashboard_payload.get("portfolio_Rollup", {})
        lines.append("────────────────────────")
        lines.append(
            f"<b>Total Combined Equity:</b> ${rollup.get('total_current_equity_usdt', 3000):,.2f} "
            f"({rollup.get('portfolio_growth_rate_pct', 0):+.2f}%)"
        )
        return "\n".join(lines)


def _demo_notifier():
    notifier = TelegramAlertNotifier()
    sample_decision = {
        "symbol": "BTCUSDT",
        "horizon": "scalp",
        "direction": "LONG",
        "evaluated_bias": "LONG",
        "signal_tier": "A_PLUS_INSTITUTIONAL",
        "total_confluence_score": 84.5,
        "component_scores": {
            "ml_shap_score_45": 36.5,
            "pattern_structure_score_25": 23.0,
            "sentiment_microstructure_score_15": 13.0,
            "debate_memory_score_15": 12.0,
        },
        "risk_plan": {
            "entry_price": 91250.0,
            "atr_value": 115.0,
            "stop_loss": 91077.5,
            "tp1_price": 91457.0,
            "tp2_price": 91618.0,
            "sl_atr_mult": 1.5,
            "rr_ratio_tp1": 1.2,
            "rr_ratio_tp2": 2.13,
            "position_size_coins": 0.058,
            "risk_amount_usdt": 10.0,
        },
        "agent_reports": {
            "pattern": {"pattern_type": "Volatility_Squeeze_Breakout_Bull", "market_regime": "Trending_Bull"},
            "ml_shap": {"shap_top_features_json": {"vol_ratio_20": 0.192, "rsi_14": 0.145}},
            "debate": {"llm_executive_review": {"judge_synthesis": "Strong volume expansion confirms 5m squeeze breakout."}},
        },
    }
    msg = notifier.format_new_trade_alert(sample_decision, {"trade_id": 101})
    res = notifier.send_message(msg)
    print(f"Dispatch Mode: {res['mode']} (Sent: {res['sent']})")
    if not res["sent"]:
        print(f"Reason: {res.get('reason')}")
    print("\n" + msg)


if __name__ == "__main__":
    _demo_notifier()