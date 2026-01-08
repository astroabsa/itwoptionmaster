from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any, List

import numpy as np
import pandas as pd


# -----------------------------
# Data prep
# -----------------------------
def flatten_chain(oc_resp: Dict) -> Tuple[pd.DataFrame, float]:
    data = oc_resp.get("data", {})
    spot = float(data.get("last_price", np.nan))
    oc = data.get("oc", {}) or {}

    rows = []
    for strike_str, sides in oc.items():
        try:
            strike = float(strike_str)
        except ValueError:
            continue

        ce = (sides or {}).get("ce") or {}
        pe = (sides or {}).get("pe") or {}

        def g(x, k): return x.get(k)

        rows.append({
            "strike": strike,

            "ce_oi": g(ce, "oi") or 0,
            "ce_oi_chg": g(ce, "oi_change") or 0,
            "ce_vol": g(ce, "volume") or 0,
            "ce_iv": g(ce, "implied_volatility") or 0,

            "pe_oi": g(pe, "oi") or 0,
            "pe_oi_chg": g(pe, "oi_change") or 0,
            "pe_vol": g(pe, "volume") or 0,
            "pe_iv": g(pe, "implied_volatility") or 0,
        })

    df = pd.DataFrame(rows).sort_values("strike")
    for c in df.columns:
        if c != "strike":
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df.reset_index(drop=True), spot


def atm_strike(spot: float, step: int) -> float:
    return round(spot / step) * step if not np.isnan(spot) else np.nan


def pcr(df: pd.DataFrame) -> float:
    ce = float(df["ce_oi"].sum())
    pe = float(df["pe_oi"].sum())
    return float(pe / ce) if ce else np.nan


# -----------------------------
# Zones
# -----------------------------
@dataclass
class Zone:
    lo: float
    hi: float
    center: float
    score: float


def _robust_z(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").fillna(0.0)
    med = x.median()
    iqr = x.quantile(0.75) - x.quantile(0.25)
    if iqr == 0 or pd.isna(iqr):
        return x * 0.0
    return (x - med) / iqr


def support_resistance_zones(df: pd.DataFrame, spot: float, band: int = 800) -> Dict[str, Optional[Zone]]:
    if np.isnan(spot):
        return {"support": None, "resistance": None}

    win = df[(df["strike"] >= spot - band) & (df["strike"] <= spot + band)].copy()
    if win.empty:
        return {"support": None, "resistance": None}

    # relative strength scoring (robust-normalized) - focuses on OI, ΔOI, Volume
    win["z_pe"] = (
        1.0 * _robust_z(win["pe_oi"]) +
        1.2 * _robust_z(win["pe_oi_chg"]) +
        0.6 * _robust_z(win["pe_vol"])
    )
    win["z_ce"] = (
        1.0 * _robust_z(win["ce_oi"]) +
        1.2 * _robust_z(win["ce_oi_chg"]) +
        0.6 * _robust_z(win["ce_vol"])
    )

    sup = win[win["strike"] <= spot].nlargest(10, "z_pe")
    res = win[win["strike"] >= spot].nlargest(10, "z_ce")

    def make_zone(x: pd.DataFrame, col: str) -> Optional[Zone]:
        if x.empty:
            return None
        lo, hi = float(x["strike"].min()), float(x["strike"].max())
        weights = x[col].clip(lower=0.1).astype(float)
        center = float(np.average(x["strike"].astype(float), weights=weights))
        score = float(x[col].sum())
        return Zone(lo=lo, hi=hi, center=center, score=score)

    return {"support": make_zone(sup, "z_pe"), "resistance": make_zone(res, "z_ce")}


# -----------------------------
# Bias
# -----------------------------
def buildup_summary(df: pd.DataFrame, spot: float, band: int = 500) -> Dict[str, Any]:
    if np.isnan(spot):
        return {"bias": "Unknown", "score": 0.0, "net_oi": 0.0, "net_vol": 0.0}

    win = df[(df["strike"] >= spot - band) & (df["strike"] <= spot + band)]
    if win.empty:
        return {"bias": "Unknown", "score": 0.0, "net_oi": 0.0, "net_vol": 0.0}

    net_oi = float(win["pe_oi_chg"].sum() - win["ce_oi_chg"].sum())
    net_vol = float(win["pe_vol"].sum() - win["ce_vol"].sum())

    score = float(np.tanh(net_oi / 2e5) + 0.5 * np.tanh(net_vol / 2e5))

    if score > 0.6:
        bias = "Bullish"
    elif score < -0.6:
        bias = "Bearish"
    else:
        bias = "Sideways"

    return {"bias": bias, "score": round(score, 2), "net_oi": net_oi, "net_vol": net_vol}


# -----------------------------
# Breakout / breakdown / range heuristic
# -----------------------------
def breakout_breakdown_probability(
    spot: float,
    support: Optional[Zone],
    resistance: Optional[Zone],
    bias_score: float,
    step: int
) -> Dict[str, float]:
    if np.isnan(spot) or support is None or resistance is None:
        return {"breakout": 0.33, "breakdown": 0.33, "range": 0.34}

    # distances to zones
    dist_to_res = max(0.0, resistance.lo - spot) if spot <= resistance.lo else max(0.0, spot - resistance.hi)
    dist_to_sup = max(0.0, spot - support.hi) if spot >= support.hi else max(0.0, support.lo - spot)

    d_res = dist_to_res / max(step, 1)
    d_sup = dist_to_sup / max(step, 1)

    # wall weakness (bigger score => stronger wall => less likely to break)
    res_weak = 1.0 / (1.0 + max(resistance.score, 0.0))
    sup_weak = 1.0 / (1.0 + max(support.score, 0.0))

    breakout_logit = (
        1.6 * res_weak
        - 0.8 * d_res
        + 0.9 * max(bias_score, 0.0)
    )
    breakdown_logit = (
        1.6 * sup_weak
        - 0.8 * d_sup
        + 0.9 * max(-bias_score, 0.0)
    )

    midpoint = (support.center + resistance.center) / 2.0
    d_mid = abs(spot - midpoint) / max(step, 1)
    range_logit = (
        0.7 * np.log1p(max(support.score, 0.0)) +
        0.7 * np.log1p(max(resistance.score, 0.0)) -
        0.5 * d_mid
    )

    logits = np.array([breakout_logit, breakdown_logit, range_logit], dtype=float)
    exps = np.exp(logits - logits.max())
    probs = exps / exps.sum()
    return {"breakout": float(probs[0]), "breakdown": float(probs[1]), "range": float(probs[2])}


# -----------------------------
# Wall shift metrics
# -----------------------------
def wall_shift(prev_zones: Dict[str, Optional[Zone]], now_zones: Dict[str, Optional[Zone]]) -> Dict[str, float]:
    out = {}
    for key in ["support", "resistance"]:
        p = prev_zones.get(key)
        n = now_zones.get(key)

        if p is None or n is None:
            out[f"{key}_center_shift"] = np.nan
            out[f"{key}_strength_change_pct"] = np.nan
            continue

        out[f"{key}_center_shift"] = float(n.center - p.center)
        denom = max(abs(p.score), 1e-9)
        out[f"{key}_strength_change_pct"] = float((n.score - p.score) / denom * 100.0)

    return out


# -----------------------------
# Event detection (wall flip / big moves)
# -----------------------------
def detect_wall_events(prev: Optional[Dict[str, Any]], now: Dict[str, Any], step: int) -> List[str]:
    events: List[str] = []
    if not prev:
        return events

    big_move = 3 * step
    big_strength = 30.0  # %

    ds = now["support_center"] - prev["support_center"]
    dr = now["resistance_center"] - prev["resistance_center"]

    if pd.notna(ds) and abs(ds) >= big_move:
        events.append(f"Support wall moved {'UP' if ds > 0 else 'DOWN'} by {abs(ds):.0f} pts")
    if pd.notna(dr) and abs(dr) >= big_move:
        events.append(f"Resistance wall moved {'UP' if dr > 0 else 'DOWN'} by {abs(dr):.0f} pts")

    if prev.get("support_score", 0) > 0:
        s_pct = (now["support_score"] - prev["support_score"]) / prev["support_score"] * 100.0
        if abs(s_pct) >= big_strength:
            events.append(f"Support strength {'UP' if s_pct > 0 else 'DOWN'} {abs(s_pct):.0f}%")

    if prev.get("resistance_score", 0) > 0:
        r_pct = (now["resistance_score"] - prev["resistance_score"]) / prev["resistance_score"] * 100.0
        if abs(r_pct) >= big_strength:
            events.append(f"Resistance strength {'UP' if r_pct > 0 else 'DOWN'} {abs(r_pct):.0f}%")

    # “flip-like” crossings
    if prev["spot"] < prev["resistance_center"] and now["spot"] > now["resistance_center"]:
        events.append("Spot crossed ABOVE resistance center (potential breakout attempt)")
    if prev["spot"] > prev["support_center"] and now["spot"] < now["support_center"]:
        events.append("Spot crossed BELOW support center (potential breakdown attempt)")

    return events


# -----------------------------
# Buy-only recommendation (rule based, no selling)
# -----------------------------
def _nearest_strike(spot: float, step: int) -> float:
    return round(spot / step) * step


def _clamp_strike_to_chain(df: pd.DataFrame, strike: float) -> float:
    strikes = df["strike"].dropna().unique()
    if len(strikes) == 0:
        return strike
    return float(strikes[np.argmin(np.abs(strikes - strike))])


def option_buy_recommendation(
    df: pd.DataFrame,
    spot: float,
    step: int,
    support: Optional[Zone],
    resistance: Optional[Zone],
    probs: Dict[str, float],
    bias: Dict[str, Any],
) -> Dict[str, Any]:
    if np.isnan(spot):
        return {"action": "NO_TRADE", "reason": "Spot unavailable"}

    p_break = probs.get("breakout", 0.0)
    p_down = probs.get("breakdown", 0.0)
    p_range = probs.get("range", 0.0)
    bias_score = float(bias.get("score", 0.0))
    bias_label = str(bias.get("bias", "Unknown"))

    # Don’t force buys in deep chop
    if p_range >= 0.70 and max(p_break, p_down) < 0.25:
        return {
            "action": "NO_TRADE",
            "reason": f"Range conditions dominant ({p_range*100:.0f}% range). Wait for wall shift / momentum.",
            "bias": bias_label,
        }

    direction = None
    if (p_break > p_down) and (bias_score >= -0.1):
        direction = "CALL"
    elif (p_down > p_break) and (bias_score <= 0.1):
        direction = "PUT"
    else:
        return {
            "action": "NO_TRADE",
            "reason": f"Signals mixed (break {p_break*100:.0f}%, down {p_down*100:.0f}%, bias {bias_label}).",
            "bias": bias_label,
        }

    atm = _nearest_strike(spot, step)

    if direction == "CALL":
        near_res = False
        if resistance:
            near_res = abs(resistance.lo - spot) <= 1.5 * step
        strike = atm if near_res else (atm - step)  # 1 step ITM CALL
        strike = _clamp_strike_to_chain(df, strike)

        row = df.loc[df["strike"] == strike].head(1)
        ce_chg = float(row["ce_oi_chg"].iloc[0]) if not row.empty else 0.0
        ce_vol = float(row["ce_vol"].iloc[0]) if not row.empty else 0.0
        confirm = (np.tanh(ce_chg / 1e5) + 0.5 * np.tanh(ce_vol / 1e5))

        conf = 0.5 * (p_break - p_down) + 0.3 * max(bias_score, 0) + 0.2 * confirm
        conf = float(np.clip(conf, 0.0, 1.0))

        return {
            "action": "BUY_CALL",
            "strike": strike,
            "style": "ATM" if strike == atm else "1-step ITM",
            "confidence": conf,
            "reason": f"Breakout favored (break {p_break*100:.0f}%, down {p_down*100:.0f}%), bias {bias_label}.",
        }

    # PUT
    near_sup = False
    if support:
        near_sup = abs(spot - support.hi) <= 1.5 * step

    strike = atm if near_sup else (atm + step)  # 1 step ITM PUT
    strike = _clamp_strike_to_chain(df, strike)

    row = df.loc[df["strike"] == strike].head(1)
    pe_chg = float(row["pe_oi_chg"].iloc[0]) if not row.empty else 0.0
    pe_vol = float(row["pe_vol"].iloc[0]) if not row.empty else 0.0
    confirm = (np.tanh(pe_chg / 1e5) + 0.5 * np.tanh(pe_vol / 1e5))

    conf = 0.5 * (p_down - p_break) + 0.3 * max(-bias_score, 0) + 0.2 * confirm
    conf = float(np.clip(conf, 0.0, 1.0))

    return {
        "action": "BUY_PUT",
        "strike": strike,
        "style": "ATM" if strike == atm else "1-step ITM",
        "confidence": conf,
        "reason": f"Breakdown favored (down {p_down*100:.0f}%, break {p_break*100:.0f}%), bias {bias_label}.",
    }
