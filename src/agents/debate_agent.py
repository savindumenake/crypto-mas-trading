import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env", override=True)

from src.database.db_manager import SegregatedDatabaseManager
from src.agents.sentiment_agent import CryptoSentimentAndMicrostructureAgent


class BullBearDebateAgent:
    """
    Hybrid Quantitative + LLM Adversarial Bull vs. Bear Researcher Debate Agent.
    1. Deterministic Engine (<1ms): Evaluates Pattern, TreeSHAP, Derivatives Sentiment,
       and 15 Sentiment-Matched Historical Trades to compute a base 15.0-pt score.
    2. Selective LLM API Wrapper (Gemini / OpenAI): Uses x-goog-api-key header and
       automatic model discovery so both AIzaSy and AQ.Ab8 scoped keys work seamlessly.
    """

    def __init__(self, config_path: str = "config/assets_config.yaml"):
        self.db_manager = SegregatedDatabaseManager(config_path)
        self.sentiment_agent = CryptoSentimentAndMicrostructureAgent(config_path)
        self.gemini_api_key = (
            os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
        )
        self.openai_api_key = (
            os.getenv("OPENAI_API_KEY", "").strip().strip('"').strip("'")
        )
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
        self._resolved_gemini_models: List[str] = []

    @staticmethod
    def _extract_json_dict(raw_text: str) -> Dict[str, Any]:
        """Strips markdown code fences and parses JSON safely."""
        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.IGNORECASE
        )
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]
        return json.loads(cleaned)

    def _get_candidate_gemini_models(self, timeout_sec: float = 5.0) -> List[str]:
        """Discovers available generateContent models for the user's API key."""
        if self._resolved_gemini_models:
            return self._resolved_gemini_models

        candidates = []
        try:
            req = urllib.request.Request(
                "https://generativelanguage.googleapis.com/v1beta/models",
                headers={"x-goog-api-key": self.gemini_api_key},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                available = [
                    m["name"].replace("models/", "")
                    for m in data.get("models", [])
                    if "generateContent" in m.get("supportedGenerationMethods", [])
                ]
                # Prioritize flash models for low-latency trading debate
                preferred = [
                    m for m in available if "flash" in m and "lite" not in m and "tts" not in m and "image" not in m
                ]
                candidates.extend(preferred if preferred else available[:3])
        except Exception:
            pass

        for fallback in [
            self.gemini_model,
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-flash-latest",
        ]:
            if fallback and fallback not in candidates:
                candidates.append(fallback)

        self._resolved_gemini_models = candidates
        return candidates

    # -----------------------------------------------------------------
    # OPTIONAL LLM API WRAPPER (GEMINI / OPENAI JSON SYNTHESIS)
    # -----------------------------------------------------------------
    def _invoke_llm_debate_wrapper(
        self,
        symbol: str,
        horizon: str,
        target_direction: str,
        quant_score: float,
        bull_args: List[str],
        bear_args: List[str],
        hist_match: Dict[str, Any],
        timeout_sec: float = 10.0,
    ) -> Dict[str, Any]:
        prompt = (
            f"You are the Chief Quantitative Risk Judge for an institutional crypto trading desk.\n"
            f"Evaluate this {horizon.upper()} {target_direction} setup on {symbol}.\n"
            f"- Quantitative Base Debate Score: {quant_score}/15.0\n"
            f"- Historical 15-Trade Memory: {hist_match.get('matched_trades_count')} trades, "
            f"Win Rate={hist_match.get('sentiment_matched_win_rate', 0.5)*100:.1f}%\n"
            f"- Bull Researcher Evidence: {bull_args}\n"
            f"- Bear Researcher Counter-Evidence: {bear_args}\n\n"
            f"Respond ONLY with valid JSON in this exact schema:\n"
            f'{{"llm_verdict": "APPROVE_{target_direction}" or "REJECT_{target_direction}", '
            f'"score_adjustment": float between -2.0 and +2.0, '
            f'"bull_executive_thesis": "1 concise sentence", '
            f'"bear_executive_risk": "1 concise sentence", '
            f'"judge_synthesis": "1 concise sentence explaining final decision"}}'
        )

        # 1. Try Google Gemini API using x-goog-api-key header and auto-discovered models
        if self.gemini_api_key:
            candidate_models = self._get_candidate_gemini_models()
            last_err = ""
            for model_name in candidate_models:
                try:
                    url = (
                        f"https://generativelanguage.googleapis.com/v1beta/models/"
                        f"{model_name}:generateContent"
                    )
                    payload = {
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": 0.1,
                            "responseMimeType": "application/json",
                        },
                    }
                    req = urllib.request.Request(
                        url,
                        data=json.dumps(payload).encode("utf-8"),
                        headers={
                            "Content-Type": "application/json",
                            "x-goog-api-key": self.gemini_api_key,
                        },
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                        body = json.loads(resp.read().decode("utf-8"))
                        text_out = body["candidates"][0]["content"]["parts"][0]["text"]
                        parsed = self._extract_json_dict(text_out)
                        return {
                            "llm_used": True,
                            "llm_provider": f"gemini:{model_name}",
                            "llm_verdict": str(
                                parsed.get("llm_verdict", f"APPROVE_{target_direction}")
                            ),
                            "score_adjustment": float(
                                max(
                                    -2.0,
                                    min(2.0, float(parsed.get("score_adjustment", 0.0))),
                                )
                            ),
                            "bull_executive_thesis": str(
                                parsed.get("bull_executive_thesis", "")
                            ),
                            "bear_executive_risk": str(
                                parsed.get("bear_executive_risk", "")
                            ),
                            "judge_synthesis": str(parsed.get("judge_synthesis", "")),
                        }
                except urllib.error.HTTPError as http_err:
                    try:
                        err_body = http_err.read().decode("utf-8")
                        err_msg = (
                            json.loads(err_body)
                            .get("error", {})
                            .get("message", "")[:90]
                        )
                    except Exception:
                        err_msg = str(http_err.reason)
                    last_err = f"HTTP_{http_err.code}:{model_name} ({err_msg})"
                except Exception as exc:
                    last_err = f"{type(exc).__name__}:{model_name}"

            return self._default_llm_fallback(
                target_direction,
                bull_args,
                bear_args,
                reason=f"GEMINI_FALLBACK ({last_err})",
            )

        # 2. Try OpenAI API if OPENAI_API_KEY is available
        if self.openai_api_key:
            try:
                url = "https://api.openai.com/v1/chat/completions"
                payload = {
                    "model": self.openai_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"},
                }
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.openai_api_key}",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    text_out = body["choices"][0]["message"]["content"]
                    parsed = self._extract_json_dict(text_out)
                    return {
                        "llm_used": True,
                        "llm_provider": f"openai:{self.openai_model}",
                        "llm_verdict": str(
                            parsed.get("llm_verdict", f"APPROVE_{target_direction}")
                        ),
                        "score_adjustment": float(
                            max(
                                -2.0,
                                min(2.0, float(parsed.get("score_adjustment", 0.0))),
                            )
                        ),
                        "bull_executive_thesis": str(
                            parsed.get("bull_executive_thesis", "")
                        ),
                        "bear_executive_risk": str(
                            parsed.get("bear_executive_risk", "")
                        ),
                        "judge_synthesis": str(parsed.get("judge_synthesis", "")),
                    }
            except Exception as exc:
                return self._default_llm_fallback(
                    target_direction,
                    bull_args,
                    bear_args,
                    reason=f"OPENAI_FALLBACK ({type(exc).__name__})",
                )

        # 3. Deterministic Executive Synthesis when no external LLM API key is set
        return self._default_llm_fallback(
            target_direction,
            bull_args,
            bear_args,
            reason="QUANT_DETERMINISTIC_SYNTHESIS",
        )

    @staticmethod
    def _default_llm_fallback(
        target_direction: str,
        bull_args: List[str],
        bear_args: List[str],
        reason: str,
    ) -> Dict[str, Any]:
        top_bull = (
            bull_args[0]
            if bull_args
            else "Quantitative indicators support upside structure."
        )
        top_bear = (
            bear_args[0]
            if bear_args
            else "No major structural counter-signals detected."
        )
        return {
            "llm_used": False,
            "llm_provider": reason,
            "llm_verdict": f"QUANT_{target_direction}",
            "score_adjustment": 0.0,
            "bull_executive_thesis": top_bull,
            "bear_executive_risk": top_bear,
            "judge_synthesis": f"Evaluated {len(bull_args)} Bull vs {len(bear_args)} Bear quantitative proofs for {target_direction}.",
        }

    # -----------------------------------------------------------------
    # MAIN DEBATE ORCHESTRATOR (QUANT + SELECTIVE LLM WRAPPER)
    # -----------------------------------------------------------------
    def conduct_debate(
        self,
        symbol: str,
        horizon: str,
        target_direction: str,
        pattern_report: Dict[str, Any],
        ml_report: Dict[str, Any],
        sentiment_report: Dict[str, Any],
        use_llm: bool = True,
        min_quant_score_for_llm: float = 7.0,
        max_points: float = 15.0,
    ) -> Dict[str, Any]:
        if target_direction not in ("LONG", "SHORT"):
            target_direction = ml_report.get("evaluated_direction", "LONG")

        pattern_type = pattern_report.get("pattern_type", "No_Clear_Pattern")
        market_regime = pattern_report.get("market_regime", "Ranging")
        cur_sent_score = float(sentiment_report.get("sentiment_confluence_score", 7.5))
        cur_fund_regime = str(
            sentiment_report.get("funding_regime", "NEUTRAL_POSITIONING")
        )

        # 1. Query 15 Sentiment-Matched Historical Trades from Horizon DB
        hist_match = self.sentiment_agent.get_sentiment_matched_historical_trades(
            horizon=horizon,
            symbol=symbol,
            pattern_type=pattern_type,
            market_regime=market_regime,
            current_sentiment_score=cur_sent_score,
            current_funding_regime=cur_fund_regime,
            top_k=15,
        )
        matched_count = int(hist_match["matched_trades_count"])
        hist_win_rate = float(hist_match["sentiment_matched_win_rate"])

        # 2. Build Bull vs. Bear Arguments & Evidence Weights
        pro_arguments: List[str] = []
        counter_arguments: List[str] = []
        pro_evidence_pts = 0.0
        counter_evidence_pts = 0.0

        pat_dir = pattern_report.get("direction", "NEUTRAL")
        vol_ratio = float(pattern_report.get("volume_ratio", 1.0))
        htf_aligned = bool(pattern_report.get("htf_aligned", False))
        htf_veto = bool(pattern_report.get("htf_counter_trend_veto", False))

        if pat_dir == target_direction:
            pro_evidence_pts += 3.0
            pro_arguments.append(
                f"Confirmed '{pattern_type}' in {market_regime} regime (Vol Ratio: {vol_ratio:.2f}x)."
            )
        else:
            counter_evidence_pts += 2.5
            counter_arguments.append(
                f"Price action shows '{pattern_type}' ({pat_dir}) which does not confirm {target_direction}."
            )

        if htf_aligned:
            pro_evidence_pts += 2.0
            pro_arguments.append(
                f"Higher-Timeframe trend ({pattern_report.get('htf_trend')}) aligns with {target_direction}."
            )
        elif htf_veto:
            counter_evidence_pts += 4.0
            counter_arguments.append(
                "CRITICAL: Trade opposes a strong Higher-Timeframe trend (ADX > 30)."
            )

        dir_conf = float(ml_report.get("directional_confidence", 0.50))
        ppv_atr = float(ml_report.get("predicted_ppv_atr", 0.0))
        signed_ppv = ppv_atr if target_direction == "LONG" else -ppv_atr
        sup_drivers = ml_report.get("supporting_drivers", [])
        opp_drivers = ml_report.get("opposing_drivers", [])

        if dir_conf >= 0.58:
            pro_evidence_pts += 3.0
            pro_arguments.append(
                f"XGBoost favours {target_direction} ({dir_conf*100:.1f}% prob) driven by {', '.join(sup_drivers[:2])}."
            )
        elif dir_conf < 0.50:
            counter_evidence_pts += 3.0
            counter_arguments.append(
                f"XGBoost opposes {target_direction} (only {dir_conf*100:.1f}% prob)."
            )

        if signed_ppv >= 0.25:
            pro_evidence_pts += 2.0
            pro_arguments.append(
                f"PPV Regressor projects +{signed_ppv:.2f} ATR favorable excursion."
            )
        elif signed_ppv < 0.0:
            counter_evidence_pts += 2.0
            counter_arguments.append(
                f"PPV Regressor warns of adverse excursion ({signed_ppv:+.2f} ATR) against {target_direction}."
            )

        if opp_drivers:
            counter_evidence_pts += min(2.0, len(opp_drivers) * 0.6)
            counter_arguments.append(
                f"Opposing SHAP factors present: {', '.join(opp_drivers[:2])}."
            )

        ob_imb = float(sentiment_report.get("orderbook_imbalance", 0.0))
        aligned_ob = ob_imb if target_direction == "LONG" else -ob_imb
        cooldown_state = sentiment_report.get("cooldown_state", {})

        if aligned_ob >= 0.15:
            pro_evidence_pts += 2.0
            pro_arguments.append(
                f"Orderbook imbalance ({ob_imb:+.2f}) shows strong institutional absorption for {target_direction}."
            )
        elif aligned_ob <= -0.15:
            counter_evidence_pts += 2.0
            counter_arguments.append(
                f"Orderbook pressure ({ob_imb:+.2f}) opposes {target_direction} entry."
            )

        if "WARNING" in cur_fund_regime:
            counter_evidence_pts += 2.5
            counter_arguments.append(
                f"Derivatives risk: {cur_fund_regime} increases liquidation squeeze risk."
            )
        elif "FUEL" in cur_fund_regime or "HEALTHY" in cur_fund_regime:
            pro_evidence_pts += 1.5
            pro_arguments.append(
                f"Funding regime ({cur_fund_regime}) supports continuation."
            )

        if cooldown_state.get("in_cooldown_window"):
            mins_ago = cooldown_state.get("minutes_since_shock")
            counter_evidence_pts += 3.5
            counter_arguments.append(
                f"Market is in post-shock cooldown window ({mins_ago}m since volatility shock)."
            )

        if matched_count >= 3:
            if hist_win_rate >= 0.60:
                pro_evidence_pts += 3.5
                pro_arguments.append(
                    f"Historical Memory ({matched_count} sentiment-matched trades) shows strong {hist_win_rate*100:.1f}% Win Rate."
                )
            elif hist_win_rate < 0.45:
                counter_evidence_pts += 3.5
                counter_arguments.append(
                    f"Historical Memory warning: Last {matched_count} similar setups had a low {hist_win_rate*100:.1f}% Win Rate."
                )
            else:
                pro_arguments.append(
                    f"Historical Memory ({matched_count} matched trades) shows neutral {hist_win_rate*100:.1f}% Win Rate."
                )
        else:
            pro_arguments.append(
                "Cold-start memory (<3 historical matches); relying on primary quantitative edge."
            )

        # 3. Quantitative Base Scoring (0.0 to 15.0 points)
        total_ev = pro_evidence_pts + counter_evidence_pts + 1e-9
        debate_win_ratio = pro_evidence_pts / total_ev
        debate_dominance_score = round(debate_win_ratio * 9.0, 2)

        if matched_count >= 3:
            mem_edge = max(0.0, min(1.0, (hist_win_rate - 0.35) / 0.40))
            historical_memory_score = round(mem_edge * 6.0, 2)
        else:
            historical_memory_score = 3.50

        quant_base_score = round(
            min(max_points, debate_dominance_score + historical_memory_score), 2
        )

        if target_direction == "LONG":
            bull_args, bear_args = pro_arguments, counter_arguments
            bull_score, bear_score = (
                round(pro_evidence_pts, 2),
                round(counter_evidence_pts, 2),
            )
        else:
            bull_args, bear_args = counter_arguments, pro_arguments
            bull_score, bear_score = (
                round(counter_evidence_pts, 2),
                round(pro_evidence_pts, 2),
            )

        # 4. Selective LLM Executive Wrapper
        if use_llm and quant_base_score >= min_quant_score_for_llm:
            llm_review = self._invoke_llm_debate_wrapper(
                symbol=symbol,
                horizon=horizon,
                target_direction=target_direction,
                quant_score=quant_base_score,
                bull_args=bull_args,
                bear_args=bear_args,
                hist_match=hist_match,
            )
        else:
            llm_review = self._default_llm_fallback(
                target_direction,
                bull_args,
                bear_args,
                reason="SKIPPED_LOW_QUANT_SCORE_OR_DISABLED",
            )

        final_debate_score = round(
            max(
                0.0,
                min(max_points, quant_base_score + llm_review["score_adjustment"]),
            ),
            2,
        )

        verdict = (
            f"APPROVE_{target_direction}"
            if final_debate_score >= 8.0 and not htf_veto
            else f"CAUTION_{target_direction}"
        )

        return {
            "symbol": symbol,
            "horizon": horizon,
            "evaluated_direction": target_direction,
            "debate_confluence_score": final_debate_score,
            "judge_verdict": verdict,
            "score_breakdown": {
                "quant_base_score": quant_base_score,
                "debate_dominance_score": debate_dominance_score,
                "historical_memory_score": historical_memory_score,
                "llm_score_adjustment": llm_review["score_adjustment"],
                "bull_conviction_pts": bull_score,
                "bear_conviction_pts": bear_score,
            },
            "llm_executive_review": llm_review,
            "historical_memory_report": hist_match,
            "bull_researcher_arguments": bull_args,
            "bear_researcher_arguments": bear_args,
        }


if __name__ == "__main__":
    sample_pattern = {
        "direction": "LONG",
        "pattern_type": "Volatility_Squeeze_Breakout_Bull",
        "market_regime": "Trending_Bull",
        "volume_ratio": 2.35,
        "htf_trend": "BULLISH",
        "htf_aligned": True,
        "htf_counter_trend_veto": False,
    }
    sample_ml = {
        "evaluated_direction": "LONG",
        "directional_confidence": 0.64,
        "predicted_ppv_atr": 0.42,
        "supporting_drivers": [
            "rsi_14=28.5 (SHAP +0.14)",
            "vol_ratio_20=2.35 (SHAP +0.11)",
        ],
        "opposing_drivers": ["vwap_dist_atr=1.2 (SHAP -0.03)"],
    }
    sample_sentiment = {
        "sentiment_confluence_score": 11.73,
        "funding_regime": "HEALTHY_BULL_FUNDING",
        "orderbook_imbalance": 0.27,
        "cooldown_state": {"in_cooldown_window": False},
    }

    debate_agent = BullBearDebateAgent()
    res = debate_agent.conduct_debate(
        symbol="BTCUSDT",
        horizon="scalp",
        target_direction="LONG",
        pattern_report=sample_pattern,
        ml_report=sample_ml,
        sentiment_report=sample_sentiment,
        use_llm=True,
    )

    print("Hybrid Quantitative + LLM Bull vs. Bear Debate Output:")
    print(f"  Judge Verdict           : {res['judge_verdict']}")
    print(f"  Debate Confluence Score : {res['debate_confluence_score']} / 15.0")
    print(f"  Score Breakdown         : {res['score_breakdown']}")
    print(f"  LLM Executive Review    : {res['llm_executive_review']}")
    print(f"  Historical Memory Match : {res['historical_memory_report']}")
    print("  Bull Arguments          :")
    for arg in res["bull_researcher_arguments"]:
        print(f"    + {arg}")
    print("  Bear Arguments          :")
    for arg in res["bear_researcher_arguments"]:
        print(f"    - {arg}")