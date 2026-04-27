"""Decision-making agent using OpenAI compatible SDK.

Supports various open source models via Groq, OpenRouter, or Local Ollama.
"""

import json
import logging
from datetime import datetime

from openai import OpenAI

from src.config_loader import CONFIG
from src.indicators.local_indicators import compute_all, last_n, latest

logger = logging.getLogger(__name__)


class TradingAgent:
    """High-level trading agent that delegates reasoning to an LLM."""

    def __init__(self, mt5_api=None):
        self.mt5_api    = mt5_api
        base_model = CONFIG.get("llm_model") or "llama-3.3-70b-versatile"
        all_models = [
            base_model, 
            "llama-3.3-70b-versatile", 
        ]
        self.fallback_models = []
        for m in all_models:
            if m not in self.fallback_models:
                self.fallback_models.append(m)
        self.current_model_idx = 0
        self.model_name = self.fallback_models[self.current_model_idx]
        
        self.openrouter_models = [
            "meta-llama/llama-3.3-70b-instruct",
            "anthropic/claude-3.5-sonnet-20241022",
            "google/gemini-2.0-flash-001"
        ]

        # --- Multi-Groq-key rotation -----------------------------------------
        # Build a list of Groq clients from all available API keys.
        # When one key hits the daily rate limit, we rotate to the next account
        # before falling back to OpenRouter.
        groq_base = CONFIG.get("llm_base_url") or "https://api.groq.com/openai/v1"
        groq_keys = [CONFIG["llm_api_key"]]
        for extra in ("llm_api_key_2", "llm_api_key_3", "llm_api_key_4", "llm_api_key_5", "llm_api_key_6"):
            k = CONFIG.get(extra)
            if k:
                groq_keys.append(k)
        self.groq_clients = [
            OpenAI(api_key=k, base_url=groq_base) for k in groq_keys
        ]
        self.groq_client_idx = 0
        self.client = self.groq_clients[0]
        logger.info("Groq key pool: %d account(s) loaded", len(self.groq_clients))

        self.fallback_client = None
        if CONFIG.get("fallback_llm_api_key"):
            self.fallback_client = OpenAI(
                api_key=CONFIG["fallback_llm_api_key"],
                base_url=CONFIG.get("fallback_llm_base_url") or "https://openrouter.ai/api/v1"
            )
        self.active_client = self.client
        # Store initial state for reset at each cycle
        self._initial_models = list(self.fallback_models)
        self._initial_model_idx = 0
        self.enable_tools = CONFIG.get("enable_tool_calling", True)

    def decide_trade(self, assets, context):
        """Decide for multiple assets in one call."""
        return self._decide(context, assets=assets)

    # ------------------------------------------------------------------

    def _decide(self, context: str, assets):
        """Send context to the LLM and return structured trade decisions."""
        # Reset model state at the start of each decision cycle so a broken
        # model from a previous cycle (e.g. 404 on a stale name) doesn't persist.
        self.active_client = self.client
        self.fallback_models = list(self._initial_models)
        self.current_model_idx = self._initial_model_idx
        self.model_name = self.fallback_models[self.current_model_idx]

        system_prompt = (
            "You are a rigorous QUANTITATIVE TRADER and interdisciplinary MATHEMATICIAN-ENGINEER "
            "optimizing risk-adjusted returns for CFD and spot positions on Exness MetaTrader 5 "
            "under real execution, margin, and swap constraints.\n"
            "You will receive market + account context for SEVERAL assets, including:\n"
            f"- assets = {json.dumps(list(assets))}\n"
            "- per-asset intraday (5m) and higher-timeframe (4h) metrics\n"
            "- Active Trades with Exit Plans\n"
            "- Recent Trading History\n"
            "- Risk management limits (hard-enforced by the system, not just guidelines)\n\n"
            "Always use the 'current time' provided in the user message to evaluate any "
            "time-based conditions, such as cooldown expirations or timed exit plans.\n\n"
            "Your goal: make decisive, first-principles decisions per asset that minimize "
            "churn while capturing edge.\n\n"
            "Core policy (low-churn, position-aware)\n"
            "1) Respect prior plans: If an active trade has an exit_plan with explicit "
            "invalidation, DO NOT close or flip early unless that invalidation has occurred.\n"
            "2) Hysteresis: Only flip direction if BOTH higher-timeframe structure AND "
            "intraday structure confirm with a decisive break beyond ~0.5×ATR and momentum "
            "alignment. Otherwise prefer HOLD or adjust TP/SL.\n"
            "3) Cooldown: After opening/flipping, impose at least 3 bars cooldown before "
            "another direction change unless hard invalidation occurs. Encode in exit_plan.\n"
            "4) Swap cost is a tilt not a trigger: Don't open/close/flip solely due to swap "
            "unless expected swap cost meaningfully exceeds expected edge (>~0.25×ATR).\n"
            "5) Overbought/oversold ≠ reversal: RSI extremes need structure + momentum "
            "confirmation. Prefer tightening stops or partial profits over instant flips.\n"
            "6) Prefer adjustments over exits: Tighten stop, trail TP, reduce size first. "
            "Flip only on hard invalidation + fresh confluence.\n\n"
            "Decision discipline (per asset)\n"
            "- Choose one: buy / sell / hold / adjust.\n"
            "  • 'adjust' modifies tp_price and/or sl_price on an existing open position "
            "WITHOUT opening a new trade. Use this to pull TP closer once the trade is in "
            "your favor, or to tighten SL. Requires at least one of tp_price or sl_price to be set.\n"
            "  • Do NOT use 'adjust' if there is no open position for the asset.\n"
            "- You control allocation_usd (the system will cap it per risk limits). "
            "Ignored for 'adjust' and 'hold'.\n"
            "- order_type: 'market' (default) or 'limit'. "
            "For limit orders set limit_price. For market orders limit_price is null.\n"
            "- TP/SL sanity:\n"
            "  BUY:  tp_price > current_price, sl_price < current_price\n"
            "  SELL: tp_price < current_price, sl_price > current_price\n"
            "  If sensible TP/SL cannot be set, use null. A mandatory SL will be auto-applied.\n"
            "- **ATR-anchored targets (CRITICAL)**: The system is running on a SHORT "
            "intraday loop but applies 1-hour trend protection. The market will NOT "
            "reach 3–5% directional targets before momentum reverses. You MUST size "
            "TP and SL as multiples of the 1h ATR (or intraday ATR if 1h is unavailable).\n"
            f"  • SL distance: {CONFIG.get('sl_atr_mult_min', 1.2)}× to {CONFIG.get('sl_atr_mult_max', 2.0)}× ATR from entry.\n"
            f"  • TP distance: {CONFIG.get('tp_atr_mult_min', 1.5)}× to {CONFIG.get('tp_atr_mult_max', 3.0)}× ATR from entry.\n"
            f"  • Reward:Risk must be between {CONFIG.get('min_rr', 1.5)} and {CONFIG.get('max_rr', 3.5)} (system enforces this).\n"
            f"  Example: BTCUSD price=75600, atr=120. For a BUY:\n"
            f"    sl_price ≈ 75600 − {CONFIG.get('sl_atr_mult_min', 1.2)}×120 = {75600 - float(str(CONFIG.get('sl_atr_mult_min', 1.2))) * 120:.0f} ; "
            f"tp_price ≈ 75600 + {CONFIG.get('tp_atr_mult_min', 1.5)}×120 = {75600 + float(str(CONFIG.get('tp_atr_mult_min', 1.5))) * 120:.0f}.\n"
            "  The system will clamp your values into these bands — picking round "
            "numbers far outside the ATR bands is wasted. Do not repeat static round-number "
            "TPs (e.g. 4700, 1.165) across different entries; they become unreachable.\n"
            "- Avoid fixed round-number targets unrelated to current ATR; they are the #1 "
            "reason profitable trades reverse before TP.\n"
            "- exit_plan must include at least ONE explicit invalidation trigger.\n\n"
            "Leverage policy\n"
            "- System enforces a hard leverage cap. Stay within limits.\n"
            "- In high volatility (elevated ATR), reduce or avoid leverage.\n"
            "- Treat allocation_usd as notional exposure.\n\n"
            "Tool usage\n"
            "- Use fetch_indicator when an additional datapoint could sharpen your thesis.\n"
            "- Parameters: indicator, asset (MT5 symbol e.g. BTCUSD, XAUUSD), interval, period.\n"
            "- Indicators computed locally from MT5 candles — works for all CFD symbols.\n"
            "- Summarize tool findings in reasoning; NEVER paste raw output into the JSON.\n\n"
            "Output contract — CRITICAL\n"
            "- Output ONLY a strict JSON object with NO markdown, NO code fences, NO extra keys.\n"
            "- You MUST output valid JSON format.\n"
            "- Exactly two top-level properties:\n"
            "  'reasoning': long-form string with step-by-step analysis\n"
            "  'trade_decisions': array with one entry per asset in the order provided\n"
            "- Each trade_decisions entry MUST have ALL these keys:\n"
            "  asset, action (buy/sell/hold/adjust), allocation_usd (number), "
            "order_type (market/limit), limit_price (number or null), "
            "tp_price (number or null), sl_price (number or null), "
            "exit_plan (string), rationale (string)\n"
            "- Do NOT emit markdown, commentary, or partial JSON."
        )

        # --- Tool definition ------------------------------------------------
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "fetch_indicator",
                    "description": (
                        "Fetch technical indicators computed locally from MT5 candle data. "
                        "Works for ALL Exness MT5 symbols: BTCUSD, ETHUSD, XAUUSD, EURUSD, USOIL, etc. "
                        "Returns latest values and recent series for the requested indicator."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "indicator": {
                                "type": "string",
                                "description": "Indicator name (e.g., 'ema', 'sma', 'rsi', 'macd', 'bbands', 'atr', 'all'). DO NOT include numbers in the name (e.g., use 'rsi' NOT 'rsi14'). Use 'period' below instead."
                            },
                            "asset": {
                                "type": "string",
                                "description": "MT5 symbol, e.g. BTCUSD, XAUUSD, EURUSD"
                            },
                            "interval": {
                                "type": "string",
                                "description": "Candle interval: '1m', '5m', '15m', '1h', '4h', '1d'"
                            },
                            "period": {
                                "type": "integer",
                                "description": "Indicator period (optional, uses default if omitted)"
                            }
                        },
                        "required": ["indicator", "asset", "interval"]
                    }
                }
            }
        ]

        # --- Helpers --------------------------------------------------------

        def _log(msgs_count: int, role: str, snippet: str):
            with open("llm_requests.log", "a", encoding="utf-8") as f:
                f.write(f"\n\n=== {datetime.now()} ===\n")
                f.write(f"Model: {self.model_name}  Messages: {msgs_count}\n")
                f.write(f"Last role: {role}  Content (truncated): {snippet[:400]}\n")

        def _handle_tool_call(func_name: str, func_args: dict) -> str:
            """Execute the fetch_indicator tool and return a JSON string."""
            if func_name != "fetch_indicator":
                return json.dumps({"error": f"Unknown tool: {func_name}"})
            try:
                asset     = func_args["asset"]
                interval  = func_args["interval"]
                indicator = func_args["indicator"]

                candles = self.mt5_api.get_candles(asset, interval, 100)
                all_ind = compute_all(candles)

                if indicator == "all":
                    result = {
                        k: {"latest": latest(v) if isinstance(v, list) else v,
                            "series": last_n(v, 10) if isinstance(v, list) else v}
                        for k, v in all_ind.items()
                    }
                elif indicator == "macd":
                    result = {
                        "macd":      {"latest": latest(all_ind.get("macd", [])),
                                      "series": last_n(all_ind.get("macd", []), 10)},
                        "signal":    {"latest": latest(all_ind.get("macd_signal", [])),
                                      "series": last_n(all_ind.get("macd_signal", []), 10)},
                        "histogram": {"latest": latest(all_ind.get("macd_histogram", [])),
                                      "series": last_n(all_ind.get("macd_histogram", []), 10)},
                    }
                elif indicator == "bbands":
                    result = {
                        "upper":  {"latest": latest(all_ind.get("bbands_upper", [])),
                                   "series": last_n(all_ind.get("bbands_upper", []), 10)},
                        "middle": {"latest": latest(all_ind.get("bbands_middle", [])),
                                   "series": last_n(all_ind.get("bbands_middle", []), 10)},
                        "lower":  {"latest": latest(all_ind.get("bbands_lower", [])),
                                   "series": last_n(all_ind.get("bbands_lower", []), 10)},
                    }
                elif indicator in ("ema", "sma"):
                    period = int(func_args.get("period", 20))
                    from src.indicators.local_indicators import ema as _ema, sma as _sma
                    closes = [c["close"] for c in candles]
                    series = _ema(closes, period) if indicator == "ema" else _sma(closes, period)
                    result = {"latest": latest(series), "series": last_n(series, 10), "period": period}
                elif indicator == "rsi":
                    period = int(func_args.get("period", 14))
                    from src.indicators.local_indicators import rsi as _rsi
                    series = _rsi(candles, period)
                    result = {"latest": latest(series), "series": last_n(series, 10), "period": period}
                elif indicator == "atr":
                    period = int(func_args.get("period", 14))
                    from src.indicators.local_indicators import atr as _atr
                    series = _atr(candles, period)
                    result = {"latest": latest(series), "series": last_n(series, 10), "period": period}
                else:
                    key_map = {"adx": "adx", "obv": "obv", "vwap": "vwap", "stoch_rsi": "stoch_rsi"}
                    mapped = key_map.get(indicator, indicator)
                    series = all_ind.get(mapped, [])
                    result = {
                        "latest": latest(series) if isinstance(series, list) else series,
                        "series": last_n(series, 10) if isinstance(series, list) else series,
                    }
                return json.dumps(result, default=str)
            except Exception as ex:
                logger.error("Tool call error: %s", ex)
                return json.dumps({"error": str(ex)})

        def _sanitize_output(raw_content: str) -> dict:
            """Use the same model to normalise malformed output."""
            try:
                sanitize_msgs = [
                    {"role": "system", "content": (
                        "You are a strict JSON normalizer. Return ONLY a raw JSON object — "
                        "no markdown, no code fences, no explanation. The object must have "
                        "exactly two keys: 'reasoning' (string) and 'trade_decisions' (array). "
                        "Each trade_decisions item must have: asset, action (buy/sell/hold/adjust), "
                        "allocation_usd (number), order_type ('market' or 'limit'), "
                        "limit_price (number or null), tp_price (number or null), "
                        "sl_price (number or null), exit_plan (string), rationale (string). "
                        f"Valid assets: {json.dumps(list(assets))}. "
                        "Extract and fix JSON from the user message. Do not add extra fields."
                    )},
                    {"role": "user", "content": raw_content}
                ]
                resp = self.active_client.chat.completions.create(
                    model=self.model_name,
                    messages=sanitize_msgs,
                    temperature=0.0
                )
                cleaned = resp.choices[0].message.content
                if cleaned:
                    cleaned = cleaned.strip()
                if not cleaned:
                    logger.error("Sanitize returned empty response")
                    return {"reasoning": "", "trade_decisions": []}
                # Try _parse_response_text logic inline to handle fences
                parsed = _parse_response_text(cleaned)
                if parsed and parsed.get("trade_decisions"):
                    return parsed
                # Fallback: raw json.loads
                result = json.loads(cleaned)
                if isinstance(result, dict) and "trade_decisions" in result:
                    return result
                return {"reasoning": "", "trade_decisions": []}
            except Exception as se:
                err_str = str(se).lower()
                is_api_err = "429" in err_str or "rate limit" in err_str or "404" in err_str or "connection" in err_str or "502" in err_str or "503" in err_str
                if is_api_err:
                    logger.warning("Sanitize API call failed with %s, raising to trigger failover", se)
                    raise se
                logger.error("Sanitize failed: %s", se)
                return {"reasoning": "", "trade_decisions": []}

        def _parse_response_text(raw_text: str) -> dict | None:
            """Try to parse JSON from LLM text output. Returns None on failure."""
            cleaned = raw_text.strip()
            # Strip markdown fences if present (handles ```json, ```JSON, ``` etc.)
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                # Remove the opening fence line (e.g. ```json)
                lines = lines[1:]
                # Remove trailing fence if present
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                cleaned = "\n".join(lines).strip()
            # Find first { in case there is preamble text before the JSON
            brace_idx = cleaned.find("{")
            if brace_idx > 0:
                cleaned = cleaned[brace_idx:]
            # Find last } to trim any trailing prose
            last_brace = cleaned.rfind("}")
            if last_brace != -1 and last_brace < len(cleaned) - 1:
                cleaned = cleaned[:last_brace + 1]
            try:
                parsed = json.loads(cleaned)
                if not isinstance(parsed, dict):
                    return None
                reasoning  = parsed.get("reasoning", "") or ""
                decisions  = parsed.get("trade_decisions")
                if not isinstance(decisions, list):
                    return None
                normalised = []
                for item in decisions:
                    if isinstance(item, dict):
                        item.setdefault("allocation_usd", 0.0)
                        item.setdefault("order_type", "market")
                        item.setdefault("limit_price", None)
                        item.setdefault("tp_price", None)
                        item.setdefault("sl_price", None)
                        item.setdefault("exit_plan", "")
                        item.setdefault("rationale", "")
                        normalised.append(item)
                return {"reasoning": reasoning, "trade_decisions": normalised}
            except (json.JSONDecodeError, ValueError, TypeError):
                return None

        # --- Tool-use conversation loop ------------------------------------
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context}
        ]

        for iteration in range(10):
            try:
                if iteration == 0:
                    _log(2, "user", context[:400])
                
                # Make the API call
                req_kwargs = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": 0.2
                }
                if self.enable_tools:
                    req_kwargs["tools"] = tools
                    req_kwargs["tool_choice"] = "auto"
                
                response = self.active_client.chat.completions.create(**req_kwargs)
                
                msg = response.choices[0].message

                # Extract token usage if available
                usage = response.usage
                if usage:
                    with open("llm_requests.log", "a", encoding="utf-8") as f:
                        f.write(
                            f"Usage: input={usage.prompt_tokens} "
                            f"output={usage.completion_tokens}\n"
                        )
                
                # If there are tool calls, execute them and append results
                if msg.tool_calls and self.enable_tools:
                    # Append the model's tool calls to the messages array so it has context
                    messages.append(msg)

                    for tool_call in msg.tool_calls:
                        func_name = tool_call.function.name
                        try:
                            # Safely parse JSON arguments
                            args = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError:
                            args = {}
                            
                        result_str = _handle_tool_call(func_name, args)
                        
                        logger.info(
                            "Tool '%s' called for asset=%s interval=%s",
                            func_name, args.get("asset"), args.get("interval"),
                        )
                        
                        # Append the tool's result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": func_name,
                            "content": result_str
                        })
                    
                    # Continue the loop so model can generate final response
                    continue

                # No tool calls — parse text response
                raw_text = msg.content or ""
                
                if not raw_text.strip():
                    logger.error("Empty text response from LLM (iteration %d)", iteration)
                    break

                with open("llm_requests.log", "a", encoding="utf-8") as f:
                    f.write(f"Raw response (truncated): {raw_text[:400]}\n")

                parsed = _parse_response_text(raw_text)
                if parsed is not None:
                    reasoning = parsed.get("reasoning", "")
                    if reasoning:
                        logger.info("LLM reasoning: %s", reasoning[:400])
                    return parsed

                # JSON parsing failed — try sanitizer
                logger.warning("JSON parse failed, attempting sanitize (iteration %d)", iteration)
                sanitized = _sanitize_output(raw_text)
                if sanitized.get("trade_decisions"):
                    return sanitized
                # Sanitizer also failed — try next fallback model before giving up
                logger.warning("Sanitize failed on model %s, trying fallback model", self.model_name)
                self.current_model_idx += 1
                if self.current_model_idx >= len(self.fallback_models):
                    if self.fallback_client and self.active_client in self.groq_clients:
                        logger.warning("Primary API models exhausted for JSON formatting. Failing over to OpenRouter.")
                        self.active_client = self.fallback_client
                        self.fallback_models = self.openrouter_models
                        self.current_model_idx = 0
                        self.model_name = self.fallback_models[self.current_model_idx]
                        continue
                    logger.error("All fallback models exhausted after JSON failures.")
                    self.current_model_idx = 0
                    break
                self.model_name = self.fallback_models[self.current_model_idx]
                logger.info("Switched to fallback model after JSON failure: %s", self.model_name)
                continue  # retry with new model

            except Exception as api_err:
                err_str = str(api_err).lower()
                is_rate_limit = "429" in err_str or "rate limit" in err_str or "rate_limit" in err_str or "413" in err_str
                is_decommissioned = "model_decommissioned" in err_str or "decommissioned" in err_str
                is_not_found = "404" in err_str or "not found" in err_str or "no endpoints" in err_str
                is_conn_error = "connection" in err_str or "timeout" in err_str or "502" in err_str or "503" in err_str
                
                if is_rate_limit or is_decommissioned or is_not_found or is_conn_error:
                    reason = "Rate limit" if is_rate_limit else ("Decommissioned model" if is_decommissioned else ("Model not found" if is_not_found else "Connection Error"))
                    logger.warning("%s hit for model %s: %s", reason, self.model_name, api_err)
                    
                    # If it's a rate limit on Groq, try rotating API keys first before switching models
                    if is_rate_limit and hasattr(self, "groq_clients") and self.active_client in self.groq_clients:
                        current_idx = self.groq_clients.index(self.active_client)
                        if current_idx + 1 < len(self.groq_clients):
                            self.active_client = self.groq_clients[current_idx + 1]
                            self.client = self.active_client  # Update primary client so it persists across cycles
                            logger.info("Rotated to next Groq API key (index %d). Retrying model %s.", current_idx + 1, self.model_name)
                            with open("llm_requests.log", "a", encoding="utf-8") as f:
                                f.write(f"Rate limited. Rotated API key to index {current_idx + 1}\\n")
                            continue # Retry same model with new key
                        else:
                            logger.warning("Exhausted all Groq API keys.")
                    
                    # Try next model
                    self.current_model_idx += 1
                    if self.current_model_idx >= len(self.fallback_models):
                        if self.fallback_client and self.active_client in self.groq_clients:
                            logger.warning("Primary API exhausted. Failing over to OpenRouter Fallback API.")
                            self.active_client = self.fallback_client
                            self.fallback_models = self.openrouter_models
                            self.current_model_idx = 0
                            self.model_name = self.fallback_models[self.current_model_idx]
                            with open("llm_requests.log", "a", encoding="utf-8") as f:
                                f.write("Switched to FALLBACK API Provider (OpenRouter) with Premium Models.\\n")
                            continue
                        logger.error("All models and fallback providers exhausted. Cannot proceed.")
                        self.current_model_idx = 0  # Reset for future runs
                        self.active_client = self.client # reset active to main on loop restart
                        break
                    
                    self.model_name = self.fallback_models[self.current_model_idx]
                    logger.info("Gracefully falling back to model: %s", self.model_name)
                    
                    with open("llm_requests.log", "a", encoding="utf-8") as f:
                        f.write(f"Switched to model: {self.model_name}\\n")
                    
                    continue # Retry on this new model
                
                logger.error("OpenAI API error (iteration %d): %s", iteration, api_err)
                with open("llm_requests.log", "a", encoding="utf-8") as f:
                    f.write(f"API Error: {api_err}\n")
                break

        # Exhausted loop — return hold for all assets
        logger.error("Decision loop exhausted — holding all assets")
        return {
            "reasoning": "decision loop cap",
            "trade_decisions": [{
                "asset":          a,
                "action":         "hold",
                "allocation_usd": 0.0,
                "order_type":     "market",
                "limit_price":    None,
                "tp_price":       None,
                "sl_price":       None,
                "exit_plan":      "",
                "rationale":      "decision loop cap",
            } for a in assets],
        }
