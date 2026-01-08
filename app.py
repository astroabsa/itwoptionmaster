import pathlib
import datetime as dt

import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

from dhan_client import expiry_list, nearest_expiry, option_chain, NIFTY, SENSEX
from oc_analytics import (
    flatten_chain,
    atm_strike,
    pcr,
    support_resistance_zones,
    buildup_summary,
    breakout_breakdown_probability,
    wall_shift,
    detect_wall_events,
    option_buy_recommendation,
)

st.set_page_config(page_title="Live Option Chain Analyzer", layout="wide")
st.title("📈 Live Option Chain Analyzer")


# -----------------------------
# Helpers for 15-min snapshots + UI cards
# -----------------------------
def floor_to_15min(ts: dt.datetime) -> dt.datetime:
    m = (ts.minute // 15) * 15
    return ts.replace(minute=m, second=0, microsecond=0)

def arrow_fmt(value: float, unit: str = "") -> str:
    if value is None or pd.isna(value):
        return "—"
    if abs(value) < 1e-9:
        return f"→ 0{unit}"
    return f"{'↑' if value > 0 else '↓'} {abs(value):.0f}{unit}"

def pct_arrow_fmt(value: float) -> str:
    if value is None or pd.isna(value):
        return "—"
    if abs(value) < 1e-9:
        return "→ 0%"
    return f"{'↑' if value > 0 else '↓'} {abs(value):.0f}%"

def shift_color_html(label: str, value_str: str, tone: str) -> str:
    bg = {"bull": "#e9f7ef", "bear": "#fdecea", "neutral": "#f4f6f8"}[tone]
    fg = {"bull": "#1e7e34", "bear": "#c62828", "neutral": "#334155"}[tone]
    return f"""
    <div style="padding:14px;border-radius:12px;background:{bg};color:{fg};">
      <div style="font-size:13px;opacity:0.85;">{label}</div>
      <div style="font-size:28px;font-weight:800;line-height:1.2;">{value_str}</div>
    </div>
    """


# -----------------------------
# Load FnO watchlist from CSV
# -----------------------------
APP_DIR = pathlib.Path(__file__).parent
WATCHLIST_PATH = APP_DIR / "stock_watchlist.csv"

@st.cache_data(show_spinner=False)
def load_watchlist(csv_path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["SEM_TRADING_SYMBOL"] = df["SEM_TRADING_SYMBOL"].astype(str).str.upper().str.strip()
    df["SEM_INSTRUMENT_NAME"] = df["SEM_INSTRUMENT_NAME"].astype(str).str.strip()
    df["SEM_SMST_SECURITY_ID"] = pd.to_numeric(df["SEM_SMST_SECURITY_ID"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["SEM_TRADING_SYMBOL", "SEM_SMST_SECURITY_ID"])
    df = df.drop_duplicates(subset=["SEM_TRADING_SYMBOL"], keep="first")
    return df

if not WATCHLIST_PATH.exists():
    st.error(f"Missing file: {WATCHLIST_PATH.name}. Place it in the same folder as app.py.")
    st.stop()

watch_df = load_watchlist(WATCHLIST_PATH)

fno_symbols = watch_df["SEM_TRADING_SYMBOL"].tolist()
fno_symbols = [s for s in fno_symbols if s not in {"NIFTY", "SENSEX"}]
sidebar_list = ["NIFTY", "SENSEX"] + fno_symbols


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Controls")
    selected = st.selectbox("Select Instrument", sidebar_list, index=0)

    mode = st.selectbox("Mode", ["Scalp", "Intraday", "Positional"], index=1)
    preset_pct = {"Scalp": 1.5, "Intraday": 3.0, "Positional": 6.0}[mode]

    refresh = st.slider("Refresh (seconds)", 3, 30, 5)
    band_pct = st.slider(
        "Analysis range (%)",
        min_value=1.0,
        max_value=10.0,
        value=float(preset_pct),
        step=0.5,
        help="Window around spot used for option-chain analysis"
    )
    auto = st.toggle("Auto refresh", True)

if auto:
    st_autorefresh(interval=refresh * 1000, key="refresh")


# -----------------------------
# Build underlying mapping for API
# -----------------------------
def get_underlying(selected_symbol: str) -> dict:
    sym = selected_symbol.upper().strip()
    if sym == "NIFTY":
        return {"name": "NIFTY", "UnderlyingScrip": NIFTY["UnderlyingScrip"], "UnderlyingSeg": "IDX_I"}
    if sym == "SENSEX":
        return {"name": "SENSEX", "UnderlyingScrip": SENSEX["UnderlyingScrip"], "UnderlyingSeg": "IDX_I"}

    row = watch_df.loc[watch_df["SEM_TRADING_SYMBOL"] == sym].iloc[0]
    return {
        "name": sym,
        "display_name": str(row.get("SEM_INSTRUMENT_NAME", sym)),
        "UnderlyingScrip": int(row["SEM_SMST_SECURITY_ID"]),
        "UnderlyingSeg": "NSE_EQ",
    }

under = get_underlying(selected)


# -----------------------------
# Fetch option chain for nearest expiry
# -----------------------------
@st.cache_data(ttl=3, show_spinner=False)
def load_chain(underlying: dict):
    expiries = expiry_list(underlying["UnderlyingScrip"], underlying["UnderlyingSeg"])
    exp = nearest_expiry(expiries)
    oc = option_chain(underlying["UnderlyingScrip"], underlying["UnderlyingSeg"], exp)
    (df, spot) = flatten_chain(oc)
    meta = {
        "symbol": underlying.get("display_name", underlying["name"]),
        "trading_symbol": underlying["name"],
        "expiry": exp,
        "UnderlyingScrip": underlying["UnderlyingScrip"],
        "UnderlyingSeg": underlying["UnderlyingSeg"],
    }
    return df, spot, meta

df, spot, meta = load_chain(under)

# step sizes (simple defaults)
step = 50 if meta["trading_symbol"] in {"NIFTY"} else (100 if meta["trading_symbol"] in {"SENSEX"} else 10)
atm = atm_strike(spot, step)

# % → points + adaptive minimum ±6 strikes
band_points = int(round(spot * band_pct / 100.0)) if pd.notna(spot) else 0
band_points = max(band_points, 6 * step)
st.caption(f"Analysis window: ±{band_pct:.1f}%  (≈ ±{band_points} points)")

# zones + bias
zones_now = support_resistance_zones(df, spot, band=band_points)
support_now = zones_now["support"]
res_now = zones_now["resistance"]

bias_band = int(min(round(spot * 0.02), band_points)) if pd.notna(spot) else 0
bias = buildup_summary(df, spot, band=bias_band)

# probabilities
probs = {"breakout": 0.33, "breakdown": 0.33, "range": 0.34}
if support_now and res_now:
    probs = breakout_breakdown_probability(
        spot=spot,
        support=support_now,
        resistance=res_now,
        bias_score=bias["score"],
        step=step
    )


# -----------------------------
# Auto-snapshots every 15 minutes + rolling history
# -----------------------------
if "snapshots" not in st.session_state:
    st.session_state["snapshots"] = {}

hist_key = f"{meta['trading_symbol']}_history"
series_key = f"{meta['trading_symbol']}_series"

if hist_key not in st.session_state["snapshots"]:
    st.session_state["snapshots"][hist_key] = {}
if series_key not in st.session_state["snapshots"]:
    st.session_state["snapshots"][series_key] = []

now = dt.datetime.now()
bucket = floor_to_15min(now)
bucket_key = bucket.isoformat(timespec="minutes")

# snapshot object (bucket)
st.session_state["snapshots"][hist_key][bucket_key] = {
    "zones": zones_now,
    "spot": float(spot),
    "pcr": float(pcr(df)) if pd.notna(pcr(df)) else float("nan"),
    "bias_score": float(bias["score"]),
    "expiry": meta["expiry"],
}

# determine previous bucket
all_keys = sorted(st.session_state["snapshots"][hist_key].keys())
prev_key = None
for k in reversed(all_keys):
    if k < bucket_key:
        prev_key = k
        break
prev_snapshot = st.session_state["snapshots"][hist_key].get(prev_key) if prev_key else None

# upsert time-series record
support_center = float(support_now.center) if support_now else float("nan")
res_center = float(res_now.center) if res_now else float("nan")
support_score = float(support_now.score) if support_now else 0.0
res_score = float(res_now.score) if res_now else 0.0

record = {
    "t": bucket,
    "spot": float(spot),
    "pcr": float(pcr(df)) if pd.notna(pcr(df)) else float("nan"),
    "bias_score": float(bias["score"]),
    "support_center": support_center,
    "resistance_center": res_center,
    "support_score": support_score,
    "resistance_score": res_score,
    "p_breakout": float(probs["breakout"]),
    "p_breakdown": float(probs["breakdown"]),
    "p_range": float(probs["range"]),
}

lst = st.session_state["snapshots"][series_key]
if len(lst) and lst[-1]["t"] == bucket:
    lst[-1] = record
else:
    lst.append(record)

# keep last 288 points (~3 days of 15-min buckets)
MAX_POINTS = 288
if len(lst) > MAX_POINTS:
    st.session_state["snapshots"][series_key] = lst[-MAX_POINTS:]


# -----------------------------
# Header metrics
# -----------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Instrument", meta["symbol"])
c2.metric("Spot", f"{spot:.2f}")
c3.metric("ATM", f"{int(atm)}" if pd.notna(atm) else "-")
c4.metric("PCR", f"{pcr(df):.2f}" if pd.notna(pcr(df)) else "-")
c5.metric("Nearest Expiry", meta["expiry"])

st.caption(
    f"Auto-snapshots: **every 15 min** | Current bucket: **{bucket_key}**"
    + (f" | Comparing with: **{prev_key}**" if prev_key else " | No previous bucket yet.")
)

st.divider()


# -----------------------------
# Final conclusion
# -----------------------------
st.subheader("📌 Final Conclusion (Nearest Expiry)")
left, right = st.columns(2)

if support_now:
    left.success(f"Support Zone: {int(support_now.lo)} – {int(support_now.hi)}")
    left.caption(f"Strength (relative): {support_now.score:.2f} | Center: {int(support_now.center)}")
else:
    left.warning("Support: Not clear")

if res_now:
    right.error(f"Resistance Zone: {int(res_now.lo)} – {int(res_now.hi)}")
    right.caption(f"Strength (relative): {res_now.score:.2f} | Center: {int(res_now.center)}")
else:
    right.warning("Resistance: Not clear")

st.caption(f"Bias: **{bias['bias']}** (score {bias['score']:.2f})")
st.divider()


# -----------------------------
# Breakout / Breakdown probability
# -----------------------------
st.subheader("📊 Breakout / Breakdown Probability (heuristic)")
p1, p2, p3 = st.columns(3)
p1.metric("Breakout", f"{probs['breakout']*100:.0f}%")
p2.metric("Breakdown", f"{probs['breakdown']*100:.0f}%")
p3.metric("Range", f"{probs['range']*100:.0f}%")
st.caption("Heuristic probabilities reflect wall strength + distance + bias (not guaranteed).")
st.divider()


# -----------------------------
# Buy-only recommendation (no selling)
# -----------------------------
st.subheader("🧠 Buy-only Idea (rule-based)")
rec = option_buy_recommendation(
    df=df,
    spot=spot,
    step=step,
    support=support_now,
    resistance=res_now,
    probs=probs,
    bias=bias,
)

if rec["action"] == "NO_TRADE":
    st.info(f"**No trade**: {rec.get('reason', '')}")
else:
    action = rec["action"].replace("_", " ")
    strike = int(rec["strike"])
    conf = float(rec.get("confidence", 0.0))
    box_color = "#e9f7ef" if "CALL" in rec["action"] else "#fdecea"
    text_color = "#1e7e34" if "CALL" in rec["action"] else "#c62828"

    st.markdown(
        f"""
        <div style="padding:16px;border-radius:14px;background:{box_color};color:{text_color};">
          <div style="font-size:13px;opacity:0.85;">Suggested (educational, not advice)</div>
          <div style="font-size:28px;font-weight:800;line-height:1.2;">{action} @ {strike}</div>
          <div style="margin-top:8px;font-size:14px;">
            Style: <b>{rec.get('style','')}</b> &nbsp;|&nbsp;
            Confidence: <b>{conf*100:.0f}%</b>
          </div>
          <div style="margin-top:8px;font-size:13px;opacity:0.9;">
            {rec.get('reason','')}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
st.caption("Buy-only guidance. Always validate premium/IV and use predefined risk management.")
st.divider()


# -----------------------------
# Wall Shift (arrows + color coding) vs previous 15-min bucket
# -----------------------------
st.subheader("🔁 Wall Shift vs Previous 15-min Snapshot")

if prev_snapshot and prev_snapshot.get("zones"):
    shifts = wall_shift(prev_snapshot["zones"], zones_now)

    sup_center = shifts.get("support_center_shift")
    sup_strength = shifts.get("support_strength_change_pct")
    res_center = shifts.get("resistance_center_shift")
    res_strength = shifts.get("resistance_strength_change_pct")

    bull_points = 0
    bull_points += 1 if pd.notna(sup_center) and sup_center > 0 else 0
    bull_points += 1 if pd.notna(sup_strength) and sup_strength > 0 else 0
    bull_points += 1 if pd.notna(res_center) and res_center > 0 else 0
    bull_points += 1 if pd.notna(res_strength) and res_strength < 0 else 0

    bear_points = 0
    bear_points += 1 if pd.notna(sup_center) and sup_center < 0 else 0
    bear_points += 1 if pd.notna(sup_strength) and sup_strength < 0 else 0
    bear_points += 1 if pd.notna(res_center) and res_center < 0 else 0
    bear_points += 1 if pd.notna(res_strength) and res_strength > 0 else 0

    if bull_points >= 3:
        overall_tone, overall_label = "bull", "Bullish Shift"
    elif bear_points >= 3:
        overall_tone, overall_label = "bear", "Bearish Shift"
    else:
        overall_tone, overall_label = "neutral", "Neutral / No clear shift"

    st.markdown(shift_color_html("Overall", overall_label, overall_tone), unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        shift_color_html(
            "Support Center",
            arrow_fmt(sup_center, " pts"),
            "bull" if pd.notna(sup_center) and sup_center > 0 else ("bear" if pd.notna(sup_center) and sup_center < 0 else "neutral")
        ),
        unsafe_allow_html=True,
    )
    c2.markdown(
        shift_color_html(
            "Support Strength",
            pct_arrow_fmt(sup_strength),
            "bull" if pd.notna(sup_strength) and sup_strength > 0 else ("bear" if pd.notna(sup_strength) and sup_strength < 0 else "neutral")
        ),
        unsafe_allow_html=True,
    )
    c3.markdown(
        shift_color_html(
            "Resistance Center",
            arrow_fmt(res_center, " pts"),
            "bull" if pd.notna(res_center) and res_center > 0 else ("bear" if pd.notna(res_center) and res_center < 0 else "neutral")
        ),
        unsafe_allow_html=True,
    )
    c4.markdown(
        shift_color_html(
            "Resistance Strength",
            pct_arrow_fmt(res_strength),
            "bull" if pd.notna(res_strength) and res_strength < 0 else ("bear" if pd.notna(res_strength) and res_strength > 0 else "neutral")
        ),
        unsafe_allow_html=True,
    )

else:
    st.info("No previous 15-min snapshot yet. Leave the app running until the next 15-min bucket.")

st.divider()


# -----------------------------
# Wall events (auto detected)
# -----------------------------
st.subheader("🚨 Wall Events (auto-detected)")
prev_record = None
if len(st.session_state["snapshots"][series_key]) >= 2:
    prev_record = st.session_state["snapshots"][series_key][-2]

events = detect_wall_events(prev_record, record, step=step)
if not events:
    st.caption("No major wall events detected in the last 15-min change.")
else:
    for e in events:
        st.warning(e)

st.divider()


# -----------------------------
# OI chart (Calls=Red, Puts=Green)
# -----------------------------
st.subheader("📈 Open Interest around Spot")
win = df[(df["strike"] >= spot - band_points) & (df["strike"] <= spot + band_points)].copy()

plot_df = win.melt(
    id_vars="strike",
    value_vars=["ce_oi", "pe_oi"],
    var_name="Side",
    value_name="OI"
)
plot_df["Side"] = plot_df["Side"].replace({"ce_oi": "CALL OI", "pe_oi": "PUT OI"})

fig = px.bar(
    plot_df,
    x="strike",
    y="OI",
    color="Side",
    barmode="group",
    title="Open Interest around Spot",
    color_discrete_map={"CALL OI": "red", "PUT OI": "green"},
)
st.plotly_chart(fig, use_container_width=True, key=f"oi_{meta['trading_symbol']}")

st.divider()


# -----------------------------
# History charts
# -----------------------------
st.subheader("📉 History (15-min buckets)")
hist = st.session_state["snapshots"].get(series_key, [])

if len(hist) < 2:
    st.info("Not enough history yet. Leave the app running to build 15-min history.")
else:
    hdf = pd.DataFrame(hist).sort_values("t")
    hdf["t"] = pd.to_datetime(hdf["t"])

    fig1 = px.line(hdf, x="t", y=["spot", "support_center", "resistance_center"],
                   title="Spot vs Support/Resistance Centers")
    st.plotly_chart(fig1, use_container_width=True, key=f"hist_centers_{meta['trading_symbol']}")

    fig2 = px.line(hdf, x="t", y=["support_score", "resistance_score"],
                   title="Wall Strength (relative)")
    st.plotly_chart(fig2, use_container_width=True, key=f"hist_strength_{meta['trading_symbol']}")

    fig3 = px.line(hdf, x="t", y=["p_breakout", "p_breakdown", "p_range"],
                   title="Breakout / Breakdown / Range (heuristic)")
    st.plotly_chart(fig3, use_container_width=True, key=f"hist_probs_{meta['trading_symbol']}")
