import pathlib
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
)

st.set_page_config(page_title="Live Option Chain Analyzer", layout="wide")
st.title("📈 Live Option Chain Analyzer")

# -----------------------------
# Load FnO watchlist from CSV
# -----------------------------
APP_DIR = pathlib.Path(__file__).parent
WATCHLIST_PATH = APP_DIR / "stock_watchlist.csv"

@st.cache_data(show_spinner=False)
def load_watchlist(csv_path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # expected columns:
    # SEM_TRADING_SYMBOL, SEM_SMST_SECURITY_ID, SEM_INSTRUMENT_NAME
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

# Sidebar list: top 2 fixed, rest from file
fno_symbols = watch_df["SEM_TRADING_SYMBOL"].tolist()
fno_symbols = [s for s in fno_symbols if s not in {"NIFTY", "SENSEX"}]
sidebar_list = ["NIFTY", "SENSEX"] + fno_symbols

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Controls")
    selected = st.selectbox("Select Instrument", sidebar_list, index=0)

    refresh = st.slider("Refresh (seconds)", 3, 30, 5)

    # ✅ Percent-based analysis range
    band_pct = st.slider(
        "Analysis range (%)",
        min_value=1.0,
        max_value=10.0,
        value=3.0,
        step=0.5,
        help="Window around spot used for option-chain analysis"
    )

    auto = st.toggle("Auto refresh", True)

    st.divider()
    st.subheader("Snapshots")
    compare_yesterday = st.toggle("Compare with Yesterday", True)
    save_snapshot = st.button("Save current as 'Yesterday'")

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
        "UnderlyingSeg": "NSE_EQ",  # underlying is equity cash for stock options
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

# Step size for ATM rounding (simple defaults)
step = 50 if meta["trading_symbol"] in {"NIFTY"} else (100 if meta["trading_symbol"] in {"SENSEX"} else 10)
atm = atm_strike(spot, step)

# ✅ Percent → points + adaptive minimum (±6 strikes)
band_points = int(round(spot * band_pct / 100.0)) if pd.notna(spot) else 0
band_points = max(band_points, 6 * step)  # adaptive minimum window
st.caption(f"Analysis window: ±{band_pct:.1f}%  (≈ ±{band_points} points)")

zones_now = support_resistance_zones(df, spot, band=band_points)
support_now = zones_now["support"]
res_now = zones_now["resistance"]

# Bias window should be tighter than full analysis window (use min(2% of spot, band_points))
bias_band = int(min(round(spot * 0.02), band_points)) if pd.notna(spot) else 0
bias = buildup_summary(df, spot, band=bias_band)

# -----------------------------
# Snapshot handling (session-only)
# -----------------------------
if "snapshots" not in st.session_state:
    st.session_state["snapshots"] = {}

snap_key = f"{meta['trading_symbol']}_yesterday"

if save_snapshot:
    st.session_state["snapshots"][snap_key] = {
        "zones": zones_now,
        "spot": spot,
        "pcr": pcr(df),
        "bias_score": bias["score"],
        "expiry": meta["expiry"],
    }
    st.success("Saved current zones as 'Yesterday' snapshot (session memory).")

y_snapshot = st.session_state["snapshots"].get(snap_key)

# -----------------------------
# Header metrics
# -----------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Instrument", meta["symbol"])
c2.metric("Spot", f"{spot:.2f}")
c3.metric("ATM", f"{int(atm)}" if pd.notna(atm) else "-")
c4.metric("PCR", f"{pcr(df):.2f}" if pd.notna(pcr(df)) else "-")
c5.metric("Nearest Expiry", meta["expiry"])

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

# -----------------------------
# Breakout / Breakdown probability
# -----------------------------
st.subheader("📊 Breakout / Breakdown Probability (heuristic)")
if support_now and res_now:
    probs = breakout_breakdown_probability(
        spot=spot,
        support=support_now,
        resistance=res_now,
        bias_score=bias["score"],
        step=step
    )
    p1, p2, p3 = st.columns(3)
    p1.metric("Breakout", f"{probs['breakout']*100:.0f}%")
    p2.metric("Breakdown", f"{probs['breakdown']*100:.0f}%")
    p3.metric("Range", f"{probs['range']*100:.0f}%")
else:
    st.info("Need both support and resistance zones to compute probabilities.")

# -----------------------------
# Compare vs yesterday snapshot
# -----------------------------
if compare_yesterday:
    st.subheader("🔁 Wall Shift vs Yesterday (snapshot)")
    if y_snapshot and y_snapshot.get("zones"):
        shifts = wall_shift(y_snapshot["zones"], zones_now)
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Support Center Shift", f"{shifts['support_center_shift']:+.0f} pts")
        s2.metric("Support Strength Change", f"{shifts['support_strength_change_pct']:+.0f}%")
        s3.metric("Resistance Center Shift", f"{shifts['resistance_center_shift']:+.0f} pts")
        s4.metric("Resistance Strength Change", f"{shifts['resistance_strength_change_pct']:+.0f}%")
        st.caption("Save a snapshot once to compare. This snapshot is stored in session memory (resets if app restarts).")
    else:
        st.warning("No snapshot saved yet. Click **Save current as 'Yesterday'** once.")

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
