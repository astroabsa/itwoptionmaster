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
)

st.set_page_config(page_title="Option Chain Analyzer", layout="wide")
st.title("📈 Live Option Chain Analyzer")

with st.sidebar:
    symbol = st.selectbox("Index", [NIFTY["name"], SENSEX["name"]])
    refresh = st.slider("Refresh (seconds)", 3, 30, 5)
    band = st.slider("Analysis range", 300, 1500, 800, 50)
    auto = st.toggle("Auto refresh", True)

if auto:
    st_autorefresh(interval=refresh * 1000, key="refresh")

sym = NIFTY if symbol == "NIFTY" else SENSEX

@st.cache_data(ttl=3)
def load():
    oc = fetch_chain_for_symbol(sym)
    return flatten_chain(oc), oc["_meta"]

(df, spot), meta = load()

atm = atm_strike(spot, 50 if symbol == "NIFTY" else 100)
zones = support_resistance_zones(df, spot, band)
bias = buildup_summary(df, spot)

# ---------- summary header ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Spot", f"{spot:.2f}")
c2.metric("ATM", atm)
c3.metric("PCR", pcr(df))
c4.metric("Bias", bias["bias"])

st.divider()

# ---------- final conclusion ----------
st.subheader("📌 Final Conclusion")

c1, c2 = st.columns(2)

if zones.get("support"):
    s = zones["support"]
    c1.success(f"Support Zone: {int(s.lo)} – {int(s.hi)}\n\nStrength: {s.score:.2f}")
else:
    c1.warning("Support: Not clear")

if zones.get("resistance"):
    r = zones["resistance"]
    c2.error(f"Resistance Zone: {int(r.lo)} – {int(r.hi)}\n\nStrength: {r.score:.2f}")
else:
    c2.warning("Resistance: Not clear")

st.divider()

# ---------- visualization ----------
win = df[(df["strike"] >= spot - band) & (df["strike"] <= spot + band)]

fig = px.bar(
    win.melt(
        id_vars="strike",
        value_vars=["ce_oi", "pe_oi"],
        var_name="Side",
        value_name="OI"
    ),
    x="strike",
    y="OI",
    color="Side",
    barmode="group",
    title="Open Interest around Spot"
)

st.plotly_chart(fig, use_container_width=True)
