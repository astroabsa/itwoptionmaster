from __future__ import annotations
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def flatten_chain(oc_resp: Dict[str, Any]) -> Tuple[pd.DataFrame, float]:
    """
    Returns:
      df: one row per strike with CE/PE fields
      spot: underlying last_price
    """
    data = oc_resp.get("data", {})
    spot = float(data.get("last_price", np.nan))
    oc = data.get("oc", {}) or {}

    rows: List[Dict[str, Any]] = []
    for strike_str, sides in oc.items():
        try:
            strike = float(strike_str)
        except Exception:
            continue

        ce = (sides or {}).get("ce") or {}
        pe = (sides or {}).get("pe") or {}

        def g(side: Dict[str, Any], k: str):
            return side.get(k)

        def gr(side: Dict[str, Any], k: str):
            return (side.get("greeks") or {}).get(k)

        rows.append({
            "strike": strike,

            "ce_ltp": g(ce, "last_price"),
            "ce_oi": g(ce, "oi"),
            "ce_oi_chg": g(ce, "oi_change"),
            "ce_vol": g(ce, "volume"),
            "ce_iv": g(ce, "implied_volatility"),
            "ce_delta": gr(ce, "delta"),
            "ce_gamma": gr(ce, "gamma"),
            "ce_theta": gr(ce, "theta"),
            "ce_vega": gr(ce, "vega"),

            "pe_ltp": g(pe, "last_price"),
            "pe_oi": g(pe, "oi"),
            "pe_oi_chg": g(pe, "oi_change"),
            "pe_vol": g(pe, "volume"),
            "pe_iv": g(pe, "implied_volatility"),
            "pe_delta": gr(pe, "delta"),
            "pe_gamma": gr(pe, "gamma"),
            "pe_theta": gr(pe, "theta"),
            "pe_vega": gr(pe, "vega"),
        })

    df = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)

    # numeric cleanup
    for c in df.columns:
        if c != "strike":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df, spot


def atm_strike(spot: float, step: int = 50) -> float:
    if np.isnan(spot):
        return np.nan
    return round(spot / step) * step


def pcr(df: pd.DataFrame) -> float:
    pe = df["pe_oi"].sum(skipna=True)
    ce = df["ce_oi"].sum(skipna=True)
    return float(pe / ce) if ce and ce != 0 else np.nan


def oi_walls(df: pd.DataFrame, spot: float, top_n: int = 5) -> Dict[str, pd.DataFrame]:
    """
    Support: strongest Put OI (and/or ΔOI) below spot
    Resistance: strongest Call OI (and/or ΔOI) above spot
    """
    below = df[df["strike"] <= spot].copy()
    above = df[df["strike"] >= spot].copy()

    support_oi = below.nlargest(top_n, "pe_oi")[["strike", "pe_oi", "pe_oi_chg", "pe_iv", "pe_ltp"]]
    resist_oi = above.nlargest(top_n, "ce_oi")[["strike", "ce_oi", "ce_oi_chg", "ce_iv", "ce_ltp"]]

    support_chg = below.nlargest(top_n, "pe_oi_chg")[["strike", "pe_oi", "pe_oi_chg", "pe_iv", "pe_ltp"]]
    resist_chg = above.nlargest(top_n, "ce_oi_chg")[["strike", "ce_oi", "ce_oi_chg", "ce_iv", "ce_ltp"]]

    return {
        "support_by_oi": support_oi,
        "resistance_by_oi": resist_oi,
        "support_by_oi_change": support_chg,
        "resistance_by_oi_change": resist_chg,
    }


def buildup_summary(df: pd.DataFrame, spot: float, band: float = 500) -> Dict[str, Any]:
    """
    Simple “buildup” heuristic around ATM:
    - net ΔOI (puts - calls)
    - volume imbalance
    - IV skew near ATM
    """
    if np.isnan(spot):
        return {"score": np.nan, "bias": "Unknown"}

    win = df[(df["strike"] >= spot - band) & (df["strike"] <= spot + band)].copy()
    if win.empty:
        return {"score": np.nan, "bias": "Unknown"}

    put_chg = win["pe_oi_chg"].sum(skipna=True)
    call_chg = win["ce_oi_chg"].sum(skipna=True)
    net_chg = put_chg - call_chg

    put_vol = win["pe_vol"].sum(skipna=True)
    call_vol = win["ce_vol"].sum(skipna=True)
    net_vol = put_vol - call_vol

    # crude IV skew: average put IV - call IV
    put_iv = win["pe_iv"].mean(skipna=True)
    call_iv = win["ce_iv"].mean(skipna=True)
    iv_skew = (put_iv - call_iv) if (pd.notna(put_iv) and pd.notna(call_iv)) else np.nan

    # Score: weighted combination
    score = 0.0
    score += np.tanh(net_chg / 200000) * 2.0   # tune divisor by typical OI scale
    score += np.tanh(net_vol / 200000) * 1.0
    if pd.notna(iv_skew):
        score += np.tanh(iv_skew / 5.0) * 1.0  # skew in vol pts

    # Interpret score
    if score >= 1.0:
        bias = "Bullish"
    elif score <= -1.0:
        bias = "Bearish"
    else:
        bias = "Sideways / Mixed"

    return {
        "score": float(score),
        "bias": bias,
        "net_oi_change_put_minus_call": float(net_chg),
        "net_volume_put_minus_call": float(net_vol),
        "iv_skew_put_minus_call": float(iv_skew) if pd.notna(iv_skew) else np.nan,
    }
