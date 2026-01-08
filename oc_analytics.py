from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

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
        def gr(x, k): return (x.get("greeks") or {}).get(k)

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
    return round(pe / ce, 2) if ce else np.nan


# ---------- support / resistance logic ----------

@dataclass
class Zone:
    lo: float
    hi: float
    center: float
    score: float


def _robust_z(x: pd.Series) -> pd.Series:
    med = x.median()
    iqr = x.quantile(0.75) - x.quantile(0.25)
    if iqr == 0 or pd.isna(iqr):
        return x * 0
    return (x - med) / iqr


def _infer_step(df: pd.DataFrame) -> int:
    diffs = np.diff(np.sort(df["strike"].unique()))
    diffs = diffs[diffs > 0]
    return int(pd.Series(diffs).mode().iloc[0]) if len(diffs) else 50


def support_resistance_zones(
    df: pd.DataFrame,
    spot: float,
    band: int = 800
) -> Dict[str, Zone]:

    step = _infer_step(df)
    win = df[(df["strike"] >= spot - band) & (df["strike"] <= spot + band)].copy()

    if win.empty:
        return {}

    win["z_pe"] = (
        _robust_z(win["pe_oi"]) +
        1.2 * _robust_z(win["pe_oi_chg"]) +
        0.5 * _robust_z(win["pe_vol"])
    )

    win["z_ce"] = (
        _robust_z(win["ce_oi"]) +
        1.2 * _robust_z(win["ce_oi_chg"]) +
        0.5 * _robust_z(win["ce_vol"])
    )

    sup = win[win["strike"] <= spot].nlargest(8, "z_pe")
    res = win[win["strike"] >= spot].nlargest(8, "z_ce")

    def make_zone(x, col):
        if x.empty:
            return None
        lo, hi = x["strike"].min(), x["strike"].max()
        center = np.average(x["strike"], weights=x[col].clip(lower=0.1))
        return Zone(lo, hi, center, x[col].sum())

    return {
        "support": make_zone(sup, "z_pe"),
        "resistance": make_zone(res, "z_ce"),
    }


# ---------- directional bias ----------

def buildup_summary(df: pd.DataFrame, spot: float, band: int = 500) -> Dict:
    win = df[(df["strike"] >= spot - band) & (df["strike"] <= spot + band)]
    if win.empty:
        return {"bias": "Unknown", "score": 0}

    net_oi = win["pe_oi_chg"].sum() - win["ce_oi_chg"].sum()
    net_vol = win["pe_vol"].sum() - win["ce_vol"].sum()

    score = np.tanh(net_oi / 2e5) + 0.5 * np.tanh(net_vol / 2e5)

    if score > 0.6:
        bias = "Bullish"
    elif score < -0.6:
        bias = "Bearish"
    else:
        bias = "Sideways"

    return {"bias": bias, "score": round(score, 2)}
