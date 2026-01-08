import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

from dhan_client import NIFTY, SENSEX, fetch_chain_for_symbol
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

with st.sidebar:
    symbol = st.selectbox("Index", [NIFTY["name"], SENSEX["name"]])
    refresh = st.slider("Refresh (seconds)", 3, 30, 5)
    band = st.slider("Analysis range (points)", 300, 2000, 800, 50)
    auto = st.toggle("Auto refresh", True)

    st.divider()
    st.subheader("Snapshots")
    compare_yesterday = st.toggle("Compare with Yesterday", True)
    save_snapshot = st.button("Save current as 'Yesterday'")

if auto:
    st_autorefresh(interval=refresh * 1000, key="refresh")

sym = NIFTY if symbol == "NIFTY" else SENSEX

@st.cache_data(ttl=3)
def load_chain():
    oc = fetch_chain_for_symbol(sym)
    (df, spot) = flatten_chain(oc)
    meta = oc.get("_meta", {})
    return df, spot, meta

df, spot, meta = load_chain()
step = 50 if symbol == "NIFTY" else 100
atm = atm_strike(spot, step)

zones_now = support_resistance_zones(df, spot, band=band)
support_now = zones_now["support"]
res_now = zones_now["resistance"]

bias = buildup_summary(df, spot, band=min(500, band))

# Save snapshot as "yesterday" (stored in Streamlit session_state)
if "snapshots" not in st.session_state:
    st.session_state["snapshots"] = {}

if save_snapshot:
    st.session_state["snapshots"][f"{symbol}_yesterday"] = {
        "zones": zones_now,
        "spot": spot,
        "pcr": pcr(df),
        "bias_score": bias["score"],
    }
    st.success("Saved current zones as 'Yesterday' snapshot (in-memory).")

y_key = f"{symbol}_yesterday"
y_snapshot = st.session_state["snapshots"].get(y_key)

# ---------- Header Metrics ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Spot", f"{spot:.2f}")
c2.metric("ATM", f"{int(atm)}" if pd.notna(atm) else "-")
c3.metric("PCR", f"{pcr(df):.2f}" if pd.notna(pcr(df)) else "-")
c4.metric("Bias", bias["bias"])

st.divider()

# ---------- Final Conclusion ----------
st.subheader("📌 Final Conclusion")

left, right = st.columns(2)

if support_now:
    left.success(f"Support Zone: {int(support_now.lo)} – {int(support_now.hi)}")
    left.caption(f"Strength (relative): {support_now.score:.2f}  | Center: {int(support_now.center)}")
else:
    left.warning("Support: Not clear")

if res_now:
    right.error(f"Resistance Zone: {int(res_now.lo)} – {int(res_now.hi)}")
    right.caption(f"Strength (relative): {res_now.score:.2f}  | Center: {int(res_now.center)}")
else:
    right.warning("Resistance: Not clear")

# ---------- Breakout / Breakdown probability ----------
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
    st.caption("These are model-derived heuristics (not guaranteed). They reflect wall strength + distance + bias.")
else:
    st.info("Need both support and resistance zones to compute probabilities.")

# ---------- Compare with Yesterday ----------
if compare_yesterday:
    st.subheader("🔁 Wall Shift vs Yesterday")

    if y_snapshot and y_snapshot.get("zones"):
        shifts = wall_shift(y_snapshot["zones"], zones_now)

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Support Center Shift", f"{shifts['support_center_shift']:+.0f} pts")
        s2.metric("Support Strength Change", f"{shifts['support_strength_change_pct']:+.0f}%")
        s3.metric("Resistance Center Shift", f"{shifts['resistance_center_shift']:+.0f} pts")
        s4.metric("Resistance Strength Change", f"{shifts['resistance_strength_change_pct']:+.0f}%")

        st.caption(
            "Center shift shows where the 'wall' moved. Strength change shows how much the wall built up or unwound."
        )
    else:
        st.warning("No 'Yesterday' snapshot saved yet. Click **Save current as 'Yesterday'** once.")

st.divider()

# ---------- Chart (Green puts, Red calls) ----------
st.subheader("📈 Open Interest around Spot")

win = df[(df["strike"] >= spot - band) & (df["strike"] <= spot + band)].copy()

plot_df = win.melt(
    id_vars="strike",
    value_vars=["ce_oi", "pe_oi"],
    var_name="Side",
    value_name="OI"
)

# nicer labels
plot_df["Side"] = plot_df["Side"].replace({"ce_oi": "CALL OI", "pe_oi": "PUT OI"})

fig = px.bar(
    plot_df,
    x="strike",
    y="OI",
    color="Side",
    barmode="group",
    title="Open Interest around Spot",
    color_discrete_map={
        "CALL OI": "red",
        "PUT OI": "green",
    },
)

st.plotly_chart(fig, use_container_width=True, key=f"oi_chart_{symbol}")
