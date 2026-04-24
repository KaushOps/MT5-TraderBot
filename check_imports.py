"""
Import validation script — run this to confirm all modules load correctly.
Does NOT connect to MT5 or Claude. Safe to run anytime.

Usage:
    python check_imports.py
"""
import os, sys

# Inject env overrides so config_loader does not fail on missing vars
os.environ.setdefault("MT5_LOGIN",    "279576065")
os.environ.setdefault("MT5_PASSWORD", "placeholder")
os.environ.setdefault("MT5_SERVER",   "Exness-MT5Trial8")
os.environ.setdefault("GEMINI_API_KEY", "placeholder")

sys.path.insert(0, ".")

errors = []

def check(label, fn):
    try:
        fn()
        print(f"  OK  {label}")
    except Exception as e:
        print(f"  FAIL {label}: {e}")
        errors.append((label, e))

print("\n=== MT5 AI Trading Agent — Import Check ===\n")

check("config_loader",   lambda: __import__("src.config_loader", fromlist=["CONFIG"]))
check("risk_manager",    lambda: __import__("src.risk_manager",  fromlist=["RiskManager"]))
check("local_indicators",lambda: __import__("src.indicators.local_indicators", fromlist=["compute_all"]))
check("formatting",      lambda: __import__("src.utils.formatting",   fromlist=["format_number"]))
check("prompt_utils",    lambda: __import__("src.utils.prompt_utils",  fromlist=["json_default"]))
check("mt5_api",         lambda: __import__("src.trading.mt5_api",    fromlist=["MT5API"]))
check("decision_maker",  lambda: __import__("src.agent.decision_maker", fromlist=["TradingAgent"]))
check("MetaTrader5",     lambda: __import__("MetaTrader5"))
check("google-genai",    lambda: __import__("google.genai"))
check("aiohttp",         lambda: __import__("aiohttp"))

# Quick indicator smoke-test
from src.indicators.local_indicators import compute_all, latest
candles = [{"open": 100+i, "high": 101+i, "low": 99+i, "close": 100.5+i, "volume": 1000+i} for i in range(60)]
out = compute_all(candles)
ema20  = round(latest(out["ema20"]),  4)
rsi14  = round(latest(out["rsi14"]),  2)
atr14  = round(latest(out["atr14"]),  6)
print(f"\n  Indicator smoke-test:")
print(f"    EMA20  = {ema20}  (expected ~129.5)")
print(f"    RSI14  = {rsi14}  (expected ~100.0 for rising series)")
print(f"    ATR14  = {atr14}")

print()
if errors:
    print(f"=== {len(errors)} ERROR(S) — fix before running the bot ===")
    sys.exit(1)
else:
    print("=== ALL CHECKS PASSED — Bot is ready to run ===")
    print()
    print("Next steps:")
    print("  1. Open MetaTrader 5 and log in to your Exness Zero account (Exness-MT5Trial8)")
    print("  2. Run:  python src/main.py")
