#!/usr/bin/env python3
"""
AlphaStrike V3 — Paper Trading Monitor
Runs nightly, fetches latest HL L2 orderbook + 5m candles,
evaluates V3 signals, logs simulated fills, tracks P&L.

Usage:
    uv run python scripts/paper_trade_v3.py [--days 1]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

# ── config ────────────────────────────────────────────────────────────────────
COINS = ["BTC", "ETH", "SOL"]
INTERVAL = "5m"
LEVERAGE = 5
STOP_LOSS_PCT = 0.02
TAKE_PROFIT_PCT = 0.04
IMBALANCE_LONG_THRESH = 0.60   # bid weight > 60% → bid-heavy
IMBALANCE_SHORT_THRESH = 0.40  # bid weight < 40% → ask-heavy
RSI_OVERSOLD = 45
RSI_OVERBOUGHT = 55
VOLUME_RATIO_MIN = 1.0
TOP5_LEVELS = 5

STATE_FILE = Path(__file__).parent.parent / "data" / "paper_trade_state.json"
LOG_FILE   = Path(__file__).parent.parent / "data" / "paper_trade_log.jsonl"

HL_INFO_URL = "https://api.hyperliquid.xyz/info"

# ── HL helpers ─────────────────────────────────────────────────────────────────

def fetch_candles(coin: str, days: int = 1) -> list[dict]:
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86_400_000
    resp = requests.post(
        HL_INFO_URL,
        json={"type": "candleSnapshot", "req": {"coin": coin, "interval": INTERVAL, "startTime": start_ms, "endTime": end_ms}},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_orderbook_imbalance(coin: str) -> float:
    """Return top-5 bid-weight in [0,1]. 0.5 on failure."""
    try:
        resp = requests.post(
            HL_INFO_URL,
            json={"type": "l2Book", "coin": coin},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        bids = data.get("levels", [[], []])[0][:TOP5_LEVELS]
        asks = data.get("levels", [[], []])[1][:TOP5_LEVELS]
        bid_sz = sum(float(b["sz"]) for b in bids)
        ask_sz = sum(float(a["sz"]) for a in asks)
        total = bid_sz + ask_sz
        return bid_sz / total if total > 0 else 0.5
    except Exception:
        return 0.5


# ── indicators ─────────────────────────────────────────────────────────────────

def _ema(values: list[float], period: int) -> list[float]:
    k = 2 / (period + 1)
    result = [values[0]]
    for v in values[1:]:
        result.append(v * k + result[-1] * (1 - k))
    return result


def compute_rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [max(d, 0) for d in deltas]
    losses = [max(-d, 0) for d in deltas]
    avg_g = sum(gains[-period:]) / period
    avg_l = sum(losses[-period:]) / period
    if avg_l == 0:
        return 100.0
    rs = avg_g / avg_l
    return 100 - 100 / (1 + rs)


def compute_macd_histogram(closes: list[float]) -> float:
    if len(closes) < 26:
        return 0.0
    ema12 = _ema(closes, 12)[-1]
    ema26 = _ema(closes, 26)[-1]
    macd_line = ema12 - ema26
    # signal = 9-period EMA of macd — approximate with last 9
    # For simplicity: compute macd series over last 35 candles
    macd_series = [_ema(closes[max(0,i-25):i+1], 12)[-1] - _ema(closes[max(0,i-25):i+1], 26)[-1]
                   for i in range(25, len(closes))]
    if len(macd_series) < 9:
        return 0.0
    signal = _ema(macd_series, 9)[-1]
    return macd_line - signal


def compute_volume_ratio(volumes: list[float], period: int = 20) -> float:
    if len(volumes) < period + 1:
        return 1.0
    avg = sum(volumes[-period-1:-1]) / period
    return volumes[-1] / avg if avg > 0 else 1.0


# ── signal evaluation ──────────────────────────────────────────────────────────

def evaluate_signal(coin: str, candles: list[dict], imbalance: float) -> dict[str, Any]:
    closes  = [float(c["c"]) for c in candles]
    volumes = [float(c["v"]) for c in candles]

    rsi    = compute_rsi(closes)
    macd_h = compute_macd_histogram(closes)
    vol_r  = compute_volume_ratio(volumes)
    price  = closes[-1]

    # V2 base signal
    long_v2  = rsi < RSI_OVERSOLD  and macd_h > 0 and vol_r >= VOLUME_RATIO_MIN
    short_v2 = rsi > RSI_OVERBOUGHT and macd_h < 0 and vol_r >= VOLUME_RATIO_MIN

    # V3 imbalance gate
    imb_long_ok  = imbalance > IMBALANCE_LONG_THRESH
    imb_short_ok = imbalance < IMBALANCE_SHORT_THRESH

    long_v3  = long_v2  and imb_long_ok
    short_v3 = short_v2 and imb_short_ok

    direction = "LONG" if long_v3 else ("SHORT" if short_v3 else "FLAT")

    return {
        "coin": coin,
        "price": price,
        "rsi": round(rsi, 2),
        "macd_histogram": round(macd_h, 6),
        "volume_ratio": round(vol_r, 3),
        "orderbook_imbalance": round(imbalance, 4),
        "v2_signal": "LONG" if long_v2 else ("SHORT" if short_v2 else "FLAT"),
        "v3_signal": direction,
        "imbalance_gate_long": imb_long_ok,
        "imbalance_gate_short": imb_short_ok,
    }


# ── state / P&L tracking ────────────────────────────────────────────────────────

def load_state() -> dict:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"positions": {}, "closed_trades": [], "equity": 1000.0}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def log_event(event: dict) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a") as f:
        f.write(json.dumps(event) + "\n")


def update_positions(state: dict, signal: dict) -> list[str]:
    """Open/close simulated positions. Returns list of action strings."""
    actions = []
    coin = signal["coin"]
    price = signal["price"]
    direction = signal["v3_signal"]
    pos = state["positions"].get(coin)

    # Check if existing position needs closing
    if pos:
        entry = pos["entry_price"]
        side  = pos["side"]
        pnl_pct = ((price - entry) / entry) * (1 if side == "LONG" else -1) * LEVERAGE

        should_close = False
        reason = ""
        if pnl_pct <= -STOP_LOSS_PCT:
            should_close, reason = True, "STOP_LOSS"
        elif pnl_pct >= TAKE_PROFIT_PCT:
            should_close, reason = True, "TAKE_PROFIT"
        elif direction != side and direction != "FLAT":
            should_close, reason = True, "SIGNAL_FLIP"

        if should_close:
            pnl_usd = pos["size_usd"] * pnl_pct
            state["equity"] += pnl_usd
            trade = {
                "coin": coin, "side": side, "entry": entry, "exit": price,
                "pnl_pct": round(pnl_pct * 100, 2), "pnl_usd": round(pnl_usd, 2),
                "reason": reason, "closed_at": datetime.now(timezone.utc).isoformat(),
            }
            state["closed_trades"].append(trade)
            del state["positions"][coin]
            actions.append(f"CLOSE {side} {coin} @ {price:.2f} | PnL: {pnl_usd:+.2f} USDC ({pnl_pct*100:+.1f}%) [{reason}]")

    # Open new position if signal and no open position
    if direction != "FLAT" and coin not in state["positions"]:
        size_usd = state["equity"] * 0.10  # 10% of equity per position
        state["positions"][coin] = {
            "side": direction, "entry_price": price,
            "size_usd": size_usd,
            "opened_at": datetime.now(timezone.utc).isoformat(),
        }
        actions.append(f"OPEN {direction} {coin} @ {price:.2f} | Size: {size_usd:.2f} USDC × {LEVERAGE}x")

    return actions


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=1, help="Candle history window in days")
    args = parser.parse_args()

    run_ts = datetime.now(timezone.utc).isoformat()
    print(f"\n{'='*60}")
    print(f"AlphaStrike V3 Paper Trade Run — {run_ts}")
    print(f"{'='*60}")

    state = load_state()
    all_actions: list[str] = []
    signals: list[dict] = []

    for coin in COINS:
        try:
            print(f"\n[{coin}] Fetching {args.days}d of {INTERVAL} candles...")
            candles = fetch_candles(coin, days=args.days)
            if len(candles) < 30:
                print(f"[{coin}] Not enough candles ({len(candles)}), skipping")
                continue

            print(f"[{coin}] Fetching live orderbook...")
            imbalance = fetch_orderbook_imbalance(coin)

            sig = evaluate_signal(coin, candles, imbalance)
            signals.append(sig)

            print(f"[{coin}] RSI={sig['rsi']} | MACD_H={sig['macd_histogram']} | "
                  f"VolR={sig['volume_ratio']} | OBI={sig['orderbook_imbalance']:.3f}")
            print(f"[{coin}] V2={sig['v2_signal']} → V3={sig['v3_signal']} "
                  f"(gate_L={sig['imbalance_gate_long']}, gate_S={sig['imbalance_gate_short']})")

            actions = update_positions(state, sig)
            all_actions.extend(actions)
            for a in actions:
                print(f"  ▶ {a}")

        except Exception as e:
            print(f"[{coin}] ERROR: {e}")

    # Open positions summary
    print(f"\n{'─'*60}")
    print(f"Equity: {state['equity']:.2f} USDC | Open: {len(state['positions'])} | Closed: {len(state['closed_trades'])}")
    for coin, pos in state["positions"].items():
        print(f"  {pos['side']} {coin} @ {pos['entry_price']:.2f} | since {pos['opened_at'][:10]}")

    # Closed trades summary
    if state["closed_trades"]:
        wins  = [t for t in state["closed_trades"] if t["pnl_usd"] > 0]
        total_pnl = sum(t["pnl_usd"] for t in state["closed_trades"])
        print(f"  Win rate: {len(wins)}/{len(state['closed_trades'])} | Total realised PnL: {total_pnl:+.2f} USDC")

    save_state(state)

    # Log full run
    log_event({
        "ts": run_ts,
        "equity": state["equity"],
        "signals": signals,
        "actions": all_actions,
        "open_positions": state["positions"],
    })

    print(f"{'='*60}\nState saved → {STATE_FILE}\nLog → {LOG_FILE}\n")


if __name__ == "__main__":
    main()
