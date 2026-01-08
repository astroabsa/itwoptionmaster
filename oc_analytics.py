from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd


# ---------- basic utilities ----------

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
            "ce_oi": g(ce, "oi"),
            "ce_oi_chg": g(ce, "oi_change"),
            "ce_vol": g(ce, "volume"),
            "ce_iv": g(ce, "implied_volatility"),
            "pe_oi": g(pe, "oi"),
            "pe_oi_chg": g(pe, "oi_change"),
            "pe_vol": g(pe, "volume"),
            "pe_iv": g(pe, "implied_volatility"),
        })

    df = pd.DataFrame(rows).sort_values("strike")
    for c in df.columns:
        if c != "strike":
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    return df.reset_index(drop=True), spot


def atm_strike(spot: float, step: int) -> float:
    return round(spot / step) * step if not np.isnan(spot) else np.nan


def pcr(df: pd.DataFrame) -> float:
    ce = df["ce_oi"].sum()
    pe = df["pe_oi"].sum()
    return float(pe / ce) if ce else np.nan


# ---------- zones ----------

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


def _infer_step(df: pd.DataFrame) -> int:
    vals = np.sort(df["strike"].dropna().unique())
    if len(vals) < 2:
        return 50
    diffs = np.diff(vals)
    diffs = diffs[diffs > 0]
    return int(pd.Series(diffs).mode().iloc[0]) if len(diffs) else 50


def support_resistance_zones(
    df: pd.DataFrame,
    spot: float,
    band: int = 800
) -> Dict[str, Optional[Zone]]:
    """
    Builds ONE best support zone (below spot) and ONE best resistance zone (above spot).
    Score uses OI + ΔOI + Volume (robust-normalized).
    """
    if np.isnan(spot):
        return {"support": None, "resistance": None}

    win = df[(df["strike"] >= spot - band) & (df["strike"] <= spot + band)].copy()
    if win.empty:
        return {"support": None, "resistance": None}

    # robust z-scores to avoid scale issues
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

    # choose top strikes first
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

    return {
        "support": make_zone(sup, "z_pe"),
        "resistance": make_zone(res, "z_ce"),
    }


# ---------- bias ----------

def buildup_summary(df: pd.DataFrame, spot: float, band: int = 500) -> Dict:
    """
    Simple direction score around ATM range:
    net ΔOI (puts - calls) + volume imbalance.
    """
    if np.isnan(spot):
        return {"bias": "Unknown", "score": 0.0}

    win = df[(df["strike"] >= spot - band) & (df["strike"] <= spot + band)]
    if win.empty:
        return {"bias": "Unknown", "score": 0.0}

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


# ---------- breakout / breakdown probability ----------

def breakout_breakdown_probability(
    spot: float,
    support: Optional[Zone],
    resistance: Optional[Zone],
    bias_score: float,
    step: int
) -> Dict[str, float]:
    """
    Explainable heuristic (NOT a statistical probability):
    - Breakout more likely when resistance is weak, spot is near resistance, bias_score is positive.
    - Breakdown more likely when support is weak, spot is near support, bias_score is negative.
    We return 3 values: breakout / breakdown / range.
    """
    if np.isnan(spot) or support is None or resistance is None:
        return {"breakout": 0.33, "breakdown": 0.33, "range": 0.34}

    # distance to nearest edge of zone
    dist_to_res = max(0.0, resistance.lo - spot) if spot <= resistance.lo else max(0.0, spot - resistance.hi)
    dist_to_sup = max(0.0, spot - support.hi) if spot >= support.hi else max(0.0, support.lo - spot)

    # normalize distances by step to keep scale stable
    d_res = dist_to_res / max(step, 1)
    d_sup = dist_to_sup / max(step, 1)

    # strengths: larger score = stronger wall
    # convert to "weakness" (smaller -> weaker wall)
    res_weak = 1.0 / (1.0 + max(resistance.score, 0.0))
    sup_weak = 1.0 / (1.0 + max(support.score, 0.0))

    # heuristic scores (logits)
    breakout_logit = (
        1.6 * res_weak      # weaker resistance => more breakout
        - 0.8 * d_res       # farther from resistance => less breakout
        + 0.9 * max(bias_score, 0.0)  # bullish bias helps breakout
    )

    breakdown_logit = (
        1.6 * sup_weak      # weaker support => more breakdown
        - 0.8 * d_sup       # farther from support => less breakdown
        + 0.9 * max(-bias_score, 0.0) # bearish bias helps breakdown
    )

    # range logit increases when both walls are strong or spot is in middle
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


def wall_shift(prev_zones: Dict[str, Optional[Zone]], now_zones: Dict[str, Optional[Zone]]) -> Dict[str, float]:
    """
    Compare yesterday vs today (or previous snapshot vs current):
    + center shift (points)
    + strength % change
    """
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
