import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

from dhan_client import NIFTY, SENSEX, fetch_chain_for_symbol
from analytics import flatten_chain, atm_strike, pcr, oi_walls, buildup_summary

st.set_page_config(page_title="Live Option Chain Analyzer", layout="wide")
st.title("📈 Live Option Chain Analyzer (DhanHQ)")

with st.sidebar:
    st.header("Controls")
    symbol = st.selectbox("Index", [NIFTY["name"], SENSEX["name"]], index=0)
    refresh = st.slider("Refresh interval (seconds)", 3, 30, 5, step=1)
    band = st.slider("ATM analysis band (points)", 200, 1500, 500, step=50)
    top_n = st.slider("Top support/resistance levels", 3, 10, 5, step=1)
    auto = st.toggle("Auto refresh", value=True)
    st.caption("Dhan option chain is rate-limited; keep refresh ≥ 3s.")

# ✅ IMPORTANT: rerun the whole script safely (NO while loop)
if auto:
    st_autorefresh(interval=refresh * 1000, key="auto_refresh")

symbol_obj = NIFTY if symbol == NIFTY["name"] else SENSEX

@st.cache_data(ttl=3, show_spinner=False)
def load_data(sym):
    oc = fetch_chain_for_symbol(sym)
    df, spot = flatten_chain(oc)
    meta = oc.get("_meta", {})
    return df, spot, meta

def render():
    df, spot, meta = load_data(symbol_obj)
    expiry = meta.get("expiry", "-")
    atm = atm_strike(spot, step=50 if symbol == "NIFTY" else 100)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Index", meta.get("symbol", symbol))
    col2.metric("Spot", f"{spot:.2f}" if pd.notna(spot) else "-")
    col3.metric("Nearest Expiry", expiry)
    col4.metric("PCR (OI)", f"{pcr(df):.2f}" if pd.notna(pcr(df)) else "-")

    summary = buildup_summary(df, spot, band=band)
    bcol1, bcol2, bcol3 = st.columns([1, 1, 2])
    bcol1.metric("Bias", summary["bias"])
    bcol2.metric("Score", f"{summary['score']:.2f}" if pd.notna(summary["score"]) else "-")
    bcol3.write({
        "net_oi_change_put_minus_call": summary["net_oi_change_put_minus_call"],
        "net_volume_put_minus_call": summary["net_volume_put_minus_call"],
        "iv_skew_put_minus_call": summary["iv_skew_put_minus_call"],
    })

    st.divider()

    left, right = st.columns(2)

    chart_df = df.copy()
    chart_df["distance"] = chart_df["strike"] - spot
    chart_df = chart_df[(chart_df["distance"] >= -band) & (chart_df["distance"] <= band)]

    oi_plot = chart_df.melt(
        id_vars=["strike"],
        value_vars=["ce_oi", "pe_oi"],
        var_name="side",
        value_name="oi",
    )
    fig_oi = px.bar(
        oi_plot, x="strike", y="oi", color="side", barmode="group",
        title=f"OI around ATM (±{band}) | ATM≈{atm}"
    )
    left.plotly_chart(fig_oi, use_container_width=True, key=f"oi_chart_{symbol}")

    doi_plot = chart_df.melt(
        id_vars=["strike"],
        value_vars=["ce_oi_chg", "pe_oi_chg"],
        var_name="side",
        value_name="oi_change",
    )
    fig_doi = px.bar(
        doi_plot, x="strike", y="oi_change", color="side", barmode="group",
        title="Change in OI (ΔOI) around ATM"
    )
    right.plotly_chart(fig_doi, use_container_width=True, key=f"doi_chart_{symbol}")

    st.divider()

    walls = oi_walls(df, spot, top_n=top_n)
    t1, t2, t3, t4 = st.columns(4)
    t1.subheader("🟢 Support (Put OI)")
    t1.dataframe(walls["support_by_oi"], use_container_width=True, hide_index=True, key=f"s_oi_{symbol}")
    t2.subheader("🔴 Resistance (Call OI)")
    t2.dataframe(walls["resistance_by_oi"], use_container_width=True, hide_index=True, key=f"r_oi_{symbol}")
    t3.subheader("🟢 Support (Put ΔOI)")
    t3.dataframe(walls["support_by_oi_change"], use_container_width=True, hide_index=True, key=f"s_doi_{symbol}")
    t4.subheader("🔴 Resistance (Call ΔOI)")
    t4.dataframe(walls["resistance_by_oi_change"], use_container_width=True, hide_index=True, key=f"r_doi_{symbol}")

    st.divider()
    st.subheader("Option Chain (flattened)")
    st.dataframe(df, use_container_width=True, height=420, key=f"chain_{symbol}")

render()
