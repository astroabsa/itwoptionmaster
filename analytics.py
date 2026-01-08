from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd


@dataclass
class Zone:
    side: str                 # "support" or "resistance"
    lo: float
    hi: float
    center: float
    score: float
    strikes: List[float]


def _robust_z(x: pd.Series) -> pd.Series:
    """Robust scaling: (x - median) / IQR. Safer than standard z for heavy tails."""
    x = pd.to_numeric(x, errors="coerce")
    med = x.median(skipna=True)
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = (q3 - q1) if pd.notna(q3) and pd.notna(q1) and (q3 - q1) != 0 else np.nan
    if pd.isna(iqr):
        return (x - med).fillna(0.0)
    return ((x - med) / iqr).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _make_zones(strikes_scores: pd.DataFrame, step: float, max_gap_steps: int = 1) -> List[Zone]:
    """
    Group adjacent strikes into zones. max_gap_steps=1 means cluster strikes within <= 1*step gap.
    Expects columns: strike, score, side
    """
    strikes_scores = strikes_scores.sort_values("strike").reset_index(drop=True)
    zones: List[Zone] = []
    if strikes_scores.empty:
        return zones

    current = [float(strikes_scores.loc[0, "strike"])]
    current_scores = [float(strikes_scores.loc[0, "score"])]
    side = str(strikes_scores.loc[0, "side"])

    for i in range(1, len(strikes_scores)):
        s = float(strikes_scores.loc[i, "strike"])
        sc = float(strikes_scores.loc[i, "score"])
        prev = current[-1]

        if (s - prev) <= (max_gap_steps * step):
            current.append(s)
            current_scores.append(sc)
        else:
            lo, hi = min(current), max(current)
            center = float(np.average(current, weights=np.maximum(current_scores, 1e-6)))
            zones.append(Zone(side=side, lo=lo, hi=hi, center=center,
                              score=float(np.sum(current_scores)), strikes=current.copy()))
            current = [s]
            current_scores = [sc]

    lo, hi = min(current), max(current)
    center = float(np.average(current, weights=np.maximum(current_scores, 1e-6)))
    zones.append(Zone(side=side, lo=lo, hi=hi, center=center,
                      score=float(np.sum(current_scores)), strikes=current.copy()))
    return zones


def infer_step(df: pd.DataFrame, fallback: float = 50) -> float:
    diffs = np.diff(np.sort(df["strike"].dropna().unique()))
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return fallback
    return float(pd.Series(diffs).mode().iloc[0])


def support_resistance_zones(
    df: pd.DataFrame,
    spot: float,
    band: float = 800,
    top_k: int = 1,
    max_gap_steps: int = 1,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Zone]:
    """
    Returns best support + best resistance zone.
    """
    if weights is None:
        weights = {
            "oi": 1.0,
            "oi_chg": 1.2,
            "vol": 0.6,
            "iv": 0.2,
        }

    step = infer_step(df, fallback=50)

    win = df[(df["strike"] >= spot - band) & (df["strike"] <= spot + band)].copy()
    if win.empty:
        return {}

    # Build support candidates (puts, below spot)
    sup = win[win["strike"] <= spot].copy()
    res = win[win["strike"] >= spot].copy()

    # If oi_change is missing/None from API, it becomes NaN. Treat as 0.
    for c in ["pe_oi", "pe_oi_chg", "pe_vol", "pe_iv", "ce_oi", "ce_oi_chg", "ce_vol", "ce_iv"]:
        if c in sup.columns:
            win[c] = pd.to_numeric(win[c], errors="coerce")
            sup[c] = pd.to_numeric(sup.get(c), errors="coerce")
            res[c] = pd.to_numeric(res.get(c), errors="coerce")

    sup["pe_oi_chg"] = sup["pe_oi_chg"].fillna(0)
    res["ce_oi_chg"] = res["ce_oi_chg"].fillna(0)

    # Robust-normalize within window
    z_pe_oi = _robust_z(win["pe_oi"])
    z_pe_chg = _robust_z(win["pe_oi_chg"])
    z_pe_vol = _robust_z(win["pe_vol"])
    z_pe_iv = _robust_z(win["pe_iv"])

    z_ce_oi = _robust_z(win["ce_oi"])
    z_ce_chg = _robust_z(win["ce_oi_chg"])
    z_ce_vol = _robust_z(win["ce_vol"])
    z_ce_iv = _robust_z(win["ce_iv"])

    # Map back to sup/res by index alignment
    win = win.reset_index(drop=True)
    sup = sup.reset_index(drop=True)
    res = res.reset_index(drop=True)

    # Rebuild sup/res with the normalized columns by merging on strike (safe)
    zmap = win[["strike"]].copy()
    zmap["z_pe_oi"] = z_pe_oi.values
    zmap["z_pe_chg"] = z_pe_chg.values
    zmap["z_pe_vol"] = z_pe_vol.values
    zmap["z_pe_iv"] = z_pe_iv.values
    zmap["z_ce_oi"] = z_ce_oi.values
    zmap["z_ce_chg"] = z_ce_chg.values
    zmap["z_ce_vol"] = z_ce_vol.values
    zmap["z_ce_iv"] = z_ce_iv.values

    sup = sup.merge(zmap, on="strike", how="left")
    res = res.merge(zmap, on="strike", how="left")

    # Strike-level scores
    sup["score"] = (
        weights["oi"] * sup["z_pe_oi"]
        + weights["oi_chg"] * sup["z_pe_chg"]
        + weights["vol"] * sup["z_pe_vol"]
        + weights["iv"] * sup["z_pe_iv"]
    )
    res["score"] = (
        weights["oi"] * res["z_ce_oi"]
        + weights["oi_chg"] * res["z_ce_chg"]
        + weights["vol"] * res["z_ce_vol"]
        + weights["iv"] * res["z_ce_iv"]
    )

    # Keep only positive contributions (ignore weak strikes)
    sup_scores = sup[["strike", "score"]].copy()
    res_scores = res[["strike", "score"]].copy()
    sup_scores = sup_scores[sup_scores["score"] > 0].sort_values("score", ascending=False)
    res_scores = res_scores[res_scores["score"] > 0].sort_values("score", ascending=False)

    # Take a handful of top strikes before clustering
    sup_scores = sup_scores.head(12).assign(side="support")
    res_scores = res_scores.head(12).assign(side="resistance")

    sup_zones = _make_zones(sup_scores, step=step, max_gap_steps=max_gap_steps)
    res_zones = _make_zones(res_scores, step=step, max_gap_steps=max_gap_steps)

    sup_best = sorted(sup_zones, key=lambda z: z.score, reverse=True)[:top_k]
    res_best = sorted(res_zones, key=lambda z: z.score, reverse=True)[:top_k]

    out = {}
    if sup_best:
        out["support"] = sup_best[0]
    if res_best:
        out["resistance"] = res_best[0]
    return out
